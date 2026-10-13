# Databricks notebook source
# DBTITLE 1,Daily Batch Inference - Next 24 Hours
# # Daily Batch Inference - Electricity Demand Forecasting
#
# ## Purpose
#
# This notebook generates **next-day hourly demand forecasts** (24 hours ahead) using the production model from Unity Catalog.
#
# **What it does:**
# 1. Loads latest available demand data (for lag features)
# 2. Loads next-day weather forecast
# 3. Applies same feature engineering as training
# 4. Loads registered production model from Unity Catalog
# 5. Generates 24-hour demand predictions
# 6. Writes forecasts to Unity Catalog for Power BI
#
# **What it does NOT do:**
# * Train or retrain models
# * Run hyperparameter tuning
# * Update model registry
# * Touch validation or test sets
#
# ## Daily Schedule
#
# * **Runs**: Every day at 2:00 AM UTC
# * **Duration**: ~5 minutes (scoring only)
# * **Input**: Latest demand + next-day weather forecast
# * **Output**: 24 hourly forecasts to `dev.stlf_gold.demand_predictions`
#
# ---

# COMMAND ----------

# DBTITLE 1,Configuration and Model Loading
# --- Configuration and Model Loading ---
import json, pickle, tempfile, os, math, time, requests, holidays
import numpy as np, pandas as pd, torch, torch.nn as nn, mlflow
from datetime import datetime, timedelta, timezone
from pyspark.sql import functions as F
from delta.tables import DeltaTable

CATALOG = "dev"
SCHEMA = "stlf_gold"
GOLD_TABLE = f"{CATALOG}.{SCHEMA}.demand_features_hourly"
PREDICTIONS_TABLE = f"{CATALOG}.{SCHEMA}.demand_predictions"
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.demand_forecast_gru"
RESPONDENT = "ERCO"
FORECAST_HORIZON = 24
HORIZON = 24
BASE_C = 18.33
LOCAL_TZ = "America/Chicago"

CITIES = [
    {"city": "Houston", "lat": 29.7604, "lon": -95.3698, "load_weight": 0.38},
    {"city": "Dallas", "lat": 32.7767, "lon": -96.7970, "load_weight": 0.28},
    {"city": "San Antonio", "lat": 29.4241, "lon": -98.4936, "load_weight": 0.19},
    {"city": "Austin", "lat": 30.2672, "lon": -97.7431, "load_weight": 0.15},
]

WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "cloud_cover", "wind_speed_10m",
    "shortwave_radiation", "direct_normal_irradiance",
]

WEATHER_MAP = {
    "temperature_2m": "temp_c", "apparent_temperature": "apparent_temp_c",
    "dew_point_2m": "dew_point_c", "relative_humidity_2m": "rel_humidity",
    "wind_speed_10m": "wind_speed_ms", "cloud_cover": "cloud_cover",
    "shortwave_radiation": "shortwave_rad", "direct_normal_irradiance": "direct_normal_rad",
}

# Load registered GRU model from Unity Catalog
mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()
mv = client.get_model_version_by_alias(UC_MODEL_NAME, "Champion")
run_id = mv.run_id
model_version = mv.version

artifact_dir = tempfile.mkdtemp()
client.download_artifacts(run_id, "model", artifact_dir)
_artifacts = os.path.join(artifact_dir, "model", "artifacts")

with open(os.path.join(_artifacts, "features.json")) as f:
    FEATURE_LIST = json.load(f)
with open(os.path.join(_artifacts, "config.json")) as f:
    MODEL_CONFIG = json.load(f)
with open(os.path.join(_artifacts, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden=64, n_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers=n_layers, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gru_model = GRUModel(
    len(FEATURE_LIST),
    hidden=MODEL_CONFIG["hidden"],
    n_layers=MODEL_CONFIG["n_layers"],
    dropout=MODEL_CONFIG["dropout"],
)
gru_model.load_state_dict(
    torch.load(os.path.join(_artifacts, "gru_weights.pt"), map_location=device, weights_only=True)
)
gru_model.to(device)
gru_model.eval()

print(f"Model: {UC_MODEL_NAME} v{model_version}")
print(f"Run ID: {run_id}")
print(f"Features: {len(FEATURE_LIST)}")
print(f"Sequence length: {MODEL_CONFIG['seq_len']}")
print(f"Device: {device}")

# COMMAND ----------

# DBTITLE 1,Data Loading and Feature Engineering
# --- Load Gold data and fetch weather forecast ---
# Load enough Gold history for all lags (max lag 336 + seq_len 24 + buffer)
HISTORY_HOURS = 400

df_gold = (
    spark.table(GOLD_TABLE)
    .filter(F.col("respondent") == RESPONDENT)
    .orderBy(F.col("ts_utc"))
    .toPandas()
)
df_gold = df_gold.sort_values("ts_utc").reset_index(drop=True)
latest_demand_ts = df_gold.loc[df_gold["demand_mw"].notna(), "ts_local"].max()

if len(df_gold) < HISTORY_HOURS:
    raise ValueError(f"Insufficient Gold history: {len(df_gold)} rows, need >= {HISTORY_HOURS}")

# Fetch weather forecast from Open-Meteo API
forecast_start = pd.Timestamp(latest_demand_ts, tz=LOCAL_TZ).tz_convert("UTC") + timedelta(hours=1)
forecast_end = forecast_start + timedelta(hours=FORECAST_HORIZON - 1)

forecast_frames = []
for c in CITIES:
    params = {
        "latitude": c["lat"], "longitude": c["lon"],
        "hourly": ",".join(WEATHER_VARS),
        "timezone": "UTC", "wind_speed_unit": "ms",
        "forecast_days": 3,
    }
    resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=120)
    resp.raise_for_status()
    fdf = pd.DataFrame(resp.json()["hourly"])
    fdf["time"] = pd.to_datetime(fdf["time"])
    fdf["load_weight"] = c["load_weight"]
    forecast_frames.append(fdf)
    time.sleep(0.4)

forecast_all = pd.concat(forecast_frames, ignore_index=True)
forecast_all = forecast_all[
    (forecast_all["time"] >= forecast_start.floor("h").tz_localize(None)) &
    (forecast_all["time"] <= forecast_end.ceil("h").tz_localize(None))
]

# Aggregate weather by load weight (same as Silver notebook)
for src, tgt in WEATHER_MAP.items():
    forecast_all[tgt] = forecast_all[src] * forecast_all["load_weight"]
weather_agg = forecast_all.groupby("time").agg(
    {tgt: "sum" for tgt in WEATHER_MAP.values()}
).reset_index().rename(columns={"time": "ts_utc"})

if len(weather_agg) < FORECAST_HORIZON:
    raise ValueError(f"Insufficient forecast hours: {len(weather_agg)}, need {FORECAST_HORIZON}")

# Create future rows matching Gold table structure
future_rows = []
US_HOLIDAYS = {d.strftime("%Y-%m-%d") for d in holidays.US(years=range(2019, 2028))}

for _, w in weather_agg.head(FORECAST_HORIZON).iterrows():
    ts_utc = pd.Timestamp(w["ts_utc"])
    ts_local = ts_utc.tz_localize("UTC").tz_convert(LOCAL_TZ).replace(tzinfo=None)
    hour = ts_local.hour
    dow = ts_local.dayofweek
    doy = ts_local.dayofyear
    day = ts_local.day

    future_rows.append({
        "ts_utc": ts_utc,
        "ts_local": ts_local,
        "respondent": RESPONDENT,
        "demand_mw": np.nan,
        "eia_dayahead_forecast_mw": np.nan,
        "sin_hour": math.sin(2 * math.pi * hour / 24),
        "cos_hour": math.cos(2 * math.pi * hour / 24),
        "sin_dow": math.sin(2 * math.pi * dow / 7),
        "cos_dow": math.cos(2 * math.pi * dow / 7),
        "sin_doy": math.sin(2 * math.pi * doy / 365.25),
        "cos_doy": math.cos(2 * math.pi * doy / 365.25),
        "is_holiday": int(ts_local.strftime("%Y-%m-%d") in US_HOLIDAYS),
        "hour_day_interaction": f"{hour}_{dow}",
        "temp_c": w["temp_c"],
        "apparent_temp_c": w["apparent_temp_c"],
        "dew_point_c": w["dew_point_c"],
        "rel_humidity": w["rel_humidity"],
        "wind_speed_ms": w["wind_speed_ms"],
        "cloud_cover": w["cloud_cover"],
        "shortwave_rad": w["shortwave_rad"],
        "direct_normal_rad": w["direct_normal_rad"],
        "cdd_c": max(0.0, w["temp_c"] - BASE_C),
        "hdd_c": max(0.0, BASE_C - w["temp_c"]),
    })

df_future = pd.DataFrame(future_rows)

# Combine historical + future, keeping only needed columns
cols_to_keep = list(df_gold.columns)
for c in df_future.columns:
    if c not in cols_to_keep:
        cols_to_keep.append(c)
df_future = df_future.reindex(columns=cols_to_keep)

df = pd.concat([df_gold[cols_to_keep], df_future[cols_to_keep]], ignore_index=True)
df = df.drop_duplicates(subset=["ts_utc"], keep="last")
df = df.sort_values("ts_utc").reset_index(drop=True)
df["is_holiday"] = df["is_holiday"].astype(int)

# --- Feature engineering (must match training notebook cells 14-19 exactly) ---
# Recompute temp_roll_mean_24 on combined data (backward-looking 24h rolling mean)
df["temp_roll_mean_24"] = df.groupby("respondent")["temp_c"].transform(
    lambda x: x.rolling(24, min_periods=1).mean()
)

# Gold base demand features (lags + rolling, shift >= 24)
for lag in [24, 48, 168]:
    df[f"lag_{lag}"] = df.groupby("respondent")["demand_mw"].shift(lag)

df["roll_mean_24"] = df.groupby("respondent")["demand_mw"].transform(
    lambda x: x.shift(HORIZON).rolling(24, min_periods=1).mean())
df["roll_std_24"] = df.groupby("respondent")["demand_mw"].transform(
    lambda x: x.shift(HORIZON).rolling(24, min_periods=1).std())
df["roll_mean_168"] = df.groupby("respondent")["demand_mw"].transform(
    lambda x: x.shift(HORIZON).rolling(168, min_periods=1).mean())

# Notebook-level lags (cell 14)
for lag in [72, 167, 169]:
    df[f"lag_{lag}"] = df.groupby("respondent")["demand_mw"].shift(lag)

# Rolling stats (cell 15, shift 24)
for window in [3, 6, 12, 48, 336]:
    df[f"roll_mean_{window}"] = df.groupby("respondent")["demand_mw"].transform(
        lambda x: x.shift(HORIZON).rolling(window, min_periods=1).mean())
    df[f"roll_std_{window}"] = df.groupby("respondent")["demand_mw"].transform(
        lambda x: x.shift(HORIZON).rolling(window, min_periods=1).std())
    df[f"roll_min_{window}"] = df.groupby("respondent")["demand_mw"].transform(
        lambda x: x.shift(HORIZON).rolling(window, min_periods=1).min())
    df[f"roll_max_{window}"] = df.groupby("respondent")["demand_mw"].transform(
        lambda x: x.shift(HORIZON).rolling(window, min_periods=1).max())

# Weather interactions (cell 16)
df["temp_x_sin_hour"] = df["temp_c"] * df["sin_hour"]
df["temp_x_cos_hour"] = df["temp_c"] * df["cos_hour"]
df["temp_x_sin_dow"] = df["temp_c"] * df["sin_dow"]
df["temp_x_cos_dow"] = df["temp_c"] * df["cos_dow"]
df["temp_anomaly"] = df["temp_c"] - df["temp_roll_mean_24"]
df["temp_anomaly_x_sin_hour"] = df["temp_anomaly"] * df["sin_hour"]
df["temp_anomaly_x_cos_hour"] = df["temp_anomaly"] * df["cos_hour"]
df["hdd_x_sin_hour"] = df["hdd_c"] * df["sin_hour"]
df["hdd_x_cos_hour"] = df["hdd_c"] * df["cos_hour"]
df["cdd_x_sin_hour"] = df["cdd_c"] * df["sin_hour"]
df["cdd_x_cos_hour"] = df["cdd_c"] * df["cos_hour"]

# Demand patterns (cell 17)
df["demand_trend_weekly"] = (df["lag_24"] - df["lag_168"]) / (df["lag_168"] + 1e-6)
df["demand_acceleration"] = df["lag_24"] - df["lag_48"]
df["demand_diff_24_48"] = df["lag_24"] - df["lag_48"]
df["demand_diff_168_336"] = df["lag_168"] - df.groupby("respondent")["demand_mw"].shift(336)
df["demand_volatility_3h"] = df.groupby("respondent")["demand_mw"].transform(
    lambda x: x.shift(HORIZON).rolling(3, min_periods=1).std())

# Calendar features (cell 18)
ts_local_dt = pd.to_datetime(df["ts_local"])
df["hour"] = ts_local_dt.dt.hour
df["dow"] = ts_local_dt.dt.dayofweek
df["day"] = ts_local_dt.dt.day
df["month"] = ts_local_dt.dt.month
df["is_weekend"] = (df["dow"] >= 5).astype(int)
df["is_working_hour"] = ((df["hour"] >= 9) & (df["hour"] <= 17) & (df["dow"] < 5)).astype(int)
df["is_peak_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19, 20]).astype(int)
df["is_month_start"] = (df["day"] <= 7).astype(int)
df["is_month_end"] = (df["day"] >= 24).astype(int)
df["week_of_month"] = (df["day"] - 1) // 7 + 1

# Weather trends (cell 19)
df["temp_diff_24"] = df["temp_c"] - df["temp_roll_mean_24"]
df["humidity_change_24"] = df["rel_humidity"] - df.groupby("respondent")["rel_humidity"].shift(24)
df["temp_momentum"] = df["temp_c"] - df.groupby("respondent")["temp_c"].shift(3)
df["wind_change_24"] = df["wind_speed_ms"] - df.groupby("respondent")["wind_speed_ms"].shift(24)

print(f"Gold data: {len(df_gold)} rows, latest demand: {latest_demand_ts}")
print(f"Future rows: {len(df_future)}")
print(f"Combined: {len(df)} rows, {df.shape[1]} columns")
print(f"Forecast period: {df_future['ts_local'].iloc[0]} to {df_future['ts_local'].iloc[-1]}")

# COMMAND ----------

# DBTITLE 1,Feature Validation and Prediction
# --- Feature validation, sequence construction, and prediction ---

# Select features in the exact order stored in the model
X_all = df[FEATURE_LIST].copy()

# Validate feature count and names
assert len(FEATURE_LIST) == 56, f"Expected 56 features, got {len(FEATURE_LIST)}"
assert X_all.shape[1] == len(FEATURE_LIST), f"Feature count mismatch: {X_all.shape[1]} vs {len(FEATURE_LIST)}"
assert list(X_all.columns) == FEATURE_LIST, "Feature order mismatch with model"

# Identify forecast rows (future hours with no demand)
forecast_mask = df["demand_mw"].isna() & (df["ts_utc"] > df.loc[df["demand_mw"].notna(), "ts_utc"].max())
forecast_indices = df.index[forecast_mask].tolist()

if len(forecast_indices) > FORECAST_HORIZON:
    print(f"WARNING: Found {len(forecast_indices)} forecast rows, trimming to {FORECAST_HORIZON}")
    forecast_indices = forecast_indices[:FORECAST_HORIZON]
elif len(forecast_indices) < FORECAST_HORIZON:
    raise ValueError(f"Expected {FORECAST_HORIZON} forecast rows, found {len(forecast_indices)}")

# Check for NaN features in the rows needed for sequence construction
seq_len = MODEL_CONFIG["seq_len"]
first_needed_idx = forecast_indices[0] - seq_len
if first_needed_idx < 0:
    raise ValueError(f"Insufficient history for sequence construction: need index >= {seq_len}, have {first_needed_idx}")

needed_features = X_all.iloc[first_needed_idx:forecast_indices[-1] + 1]
nan_cols = needed_features.columns[needed_features.isna().any()].tolist()
if nan_cols:
    print(f"WARNING: NaN features in sequence range: {nan_cols}")
    for col in nan_cols:
        df[col] = df[col].fillna(df[col].median())

# Re-select after fillna
X_all = df[FEATURE_LIST].copy()

# Apply scaler (loaded from model artifacts, NOT refitted)
X_scaled = scaler.transform(X_all.values).astype(np.float32)

# Build sequences: for each forecast hour, use preceding seq_len hours as input
sequences = []
for idx in forecast_indices:
    seq = X_scaled[idx - seq_len:idx]
    assert seq.shape == (seq_len, len(FEATURE_LIST)), f"Bad sequence shape at index {idx}: {seq.shape}"
    sequences.append(seq)

X_predict = np.array(sequences)
print(f"Prediction sequences: {X_predict.shape}")

# Run GRU model
X_t = torch.FloatTensor(X_predict).to(device)
with torch.no_grad():
    predictions = gru_model(X_t).squeeze(-1).cpu().numpy()

# Convert from scaled target back to MW
predicted_demand = predictions * MODEL_CONFIG["target_scale"]

print(f"Predictions generated: {len(predicted_demand)} hours")
print(f"Min: {predicted_demand.min():.0f} MW")
print(f"Max: {predicted_demand.max():.0f} MW")
print(f"Mean: {predicted_demand.mean():.0f} MW")

# COMMAND ----------

# DBTITLE 1,Prediction Validation and Write
# --- Prediction validation and idempotent write ---

# Build predictions dataframe
forecast_rows = df.loc[forecast_indices, ["ts_utc", "ts_local"]].copy()
forecast_rows["respondent"] = RESPONDENT
forecast_rows["predicted_demand_mw"] = predicted_demand.astype(float)
forecast_rows["actual_demand_mw"] = np.nan
forecast_rows["absolute_error_mw"] = np.nan
forecast_rows["percent_error"] = np.nan
forecast_rows["residual_mw"] = np.nan
forecast_rows["temp_c"] = df.loc[forecast_indices, "temp_c"].values
forecast_rows["rel_humidity"] = df.loc[forecast_indices, "rel_humidity"].values
forecast_rows["wind_speed_ms"] = df.loc[forecast_indices, "wind_speed_ms"].values
forecast_rows["model_name"] = "GRU"
forecast_rows["model_version"] = str(model_version)
forecast_rows["prediction_timestamp"] = datetime.now(timezone.utc)
forecast_rows["mlflow_run_id"] = run_id

# --- Sanity checks ---
# 1. No NaN predictions
assert not forecast_rows["predicted_demand_mw"].isna().any(), "NaN predictions detected"

# 2. No negative demand
neg_count = (forecast_rows["predicted_demand_mw"] < 0).sum()
if neg_count > 0:
    raise ValueError(f"{neg_count} negative demand predictions detected")

# 3. Reasonable demand magnitudes (ERCOT ranges roughly 20,000-80,000 MW)
pred_min = forecast_rows["predicted_demand_mw"].min()
pred_max = forecast_rows["predicted_demand_mw"].max()
if pred_min < 15000:
    raise ValueError(f"Prediction minimum {pred_min:.0f} MW is below 15,000 MW threshold")
if pred_max > 100000:
    raise ValueError(f"Prediction maximum {pred_max:.0f} MW exceeds 100,000 MW threshold")

# 4. No duplicate timestamps
if forecast_rows["ts_utc"].duplicated().any():
    raise ValueError("Duplicate forecast timestamps detected")

# 5. Exactly 24 hours
if len(forecast_rows) != FORECAST_HORIZON:
    raise ValueError(f"Expected {FORECAST_HORIZON} predictions, got {len(forecast_rows)}")

# 6. Timestamps are contiguous
ts_diffs = forecast_rows["ts_utc"].diff().dropna()
if not all(ts_diffs == timedelta(hours=1)):
    raise ValueError("Forecast timestamps are not contiguous (gaps detected)")

print(f"All prediction validations passed")
print(f"Forecast period: {forecast_rows['ts_local'].iloc[0]} to {forecast_rows['ts_local'].iloc[-1]}")
print(f"Predictions: {len(forecast_rows)} hours")
print(f"Min: {pred_min:.0f} MW, Max: {pred_max:.0f} MW, Mean: {forecast_rows['predicted_demand_mw'].mean():.0f} MW")

# --- Idempotent write to predictions table ---
# Delete existing predictions for this forecast period, then insert
forecast_start_utc = forecast_rows["ts_utc"].min()
forecast_end_utc = forecast_rows["ts_utc"].max()

spark.sql(f"""
    DELETE FROM {PREDICTIONS_TABLE}
    WHERE ts_utc >= '{forecast_start_utc}'
      AND ts_utc <= '{forecast_end_utc}'
      AND respondent = '{RESPONDENT}'
      AND model_name = 'GRU'
""")

pred_sdf = spark.createDataFrame(forecast_rows)
pred_sdf = pred_sdf.withColumn("ts_utc", F.col("ts_utc").cast("timestamp")) \
    .withColumn("ts_local", F.col("ts_local").cast("timestamp")) \
    .withColumn("prediction_timestamp", F.col("prediction_timestamp").cast("timestamp"))

pred_sdf.write.mode("append").saveAsTable(PREDICTIONS_TABLE)

written = spark.sql(f"""
    SELECT COUNT(*) as cnt, MIN(ts_utc) as min_ts, MAX(ts_utc) as max_ts
    FROM {PREDICTIONS_TABLE}
    WHERE ts_utc >= '{forecast_start_utc}' AND model_name = 'GRU'
""")
written.show()

print(f"\nDeployment summary:")
print(f"  Model: {UC_MODEL_NAME} v{model_version}")
print(f"  Training data end: {latest_demand_ts}")
print(f"  Feature count: {len(FEATURE_LIST)}")
print(f"  Sequence length: {MODEL_CONFIG['seq_len']}")
print(f"  Forecast date: {forecast_rows['ts_local'].iloc[0].date()}")
print(f"  Predictions: {len(forecast_rows)} hours")
print(f"  Min prediction: {pred_min:.0f} MW")
print(f"  Max prediction: {pred_max:.0f} MW")
print(f"  Mean prediction: {forecast_rows['predicted_demand_mw'].mean():.0f} MW")
print(f"  Latest demand used: {latest_demand_ts}")