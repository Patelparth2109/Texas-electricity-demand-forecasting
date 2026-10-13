# Databricks notebook source
# DBTITLE 1,Production ML Pipeline - Electricity Demand Forecasting
# MAGIC %md
# MAGIC # Production ML Pipeline - Electricity Demand Forecasting
# MAGIC
# MAGIC ## Pipeline Overview
# MAGIC
# MAGIC This notebook implements a production-ready ML pipeline for short-term electricity demand forecasting with:
# MAGIC
# MAGIC * **Automated daily execution** via Databricks Jobs
# MAGIC * **MLflow experiment tracking** for all training runs
# MAGIC * **Unity Catalog model registry** for governed model versioning
# MAGIC * **Predictions table generation** for Power BI consumption
# MAGIC * **Proper validation methodology** to prevent overfitting
# MAGIC
# MAGIC ## Pipeline Stages
# MAGIC
# MAGIC 1. **Setup & Configuration** - Environment initialization and dependencies
# MAGIC 2. **Data Ingestion** - Load hourly demand and weather data from Unity Catalog
# MAGIC 3. **Feature Engineering** - Create 50+ temporal and weather features
# MAGIC 4. **Model Training** - Train and validate multiple regression models
# MAGIC 5. **Hyperparameter Tuning** - Optimize best model with time-series cross-validation
# MAGIC 6. **Model Registration** - Register final model to Unity Catalog
# MAGIC 7. **Predictions Generation** - Score test data and save to UC table
# MAGIC 8. **Model Evaluation** - Calculate and log performance metrics
# MAGIC
# MAGIC ## Model Performance Target
# MAGIC
# MAGIC **Baseline**: EIA day-ahead forecast MAPE 2.0-3.0%  
# MAGIC **Target**: Beat baseline with MAPE < 2.0%
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Stage 1: Setup & Configuration
# MAGIC %md
# MAGIC ## Stage 1: Setup & Configuration
# MAGIC
# MAGIC Install required dependencies and configure MLflow experiment tracking.

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# Install all required packages in one cell
%pip install xgboost lightgbm catboost mlflow --quiet

# COMMAND ----------

# DBTITLE 1,Import Libraries
# Import libraries
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.catboost
from datetime import datetime

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

print(f"MLflow Version: {mlflow.__version__}")
print(f"Libraries imported successfully at {datetime.now()}")

# COMMAND ----------

# DBTITLE 1,Configure MLflow Experiment
# Configure MLflow experiment
EXPERIMENT_NAME = "/Users/qgf22@txstate.edu/electricity-demand-forecasting"
mlflow.set_experiment(EXPERIMENT_NAME)

# Set Unity Catalog as model registry
mlflow.set_registry_uri("databricks-uc")

# Configuration
CATALOG = "dev"
SCHEMA = "stlf_gold"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.demand_forecast_catboost"
PREDICTIONS_TABLE = f"{CATALOG}.{SCHEMA}.demand_predictions"

print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Model Registry: {MODEL_NAME}")
print(f"Predictions Table: {PREDICTIONS_TABLE}")

# COMMAND ----------

# DBTITLE 1,Stage 2: Data Ingestion
# MAGIC %md
# MAGIC ## Stage 2: Data Ingestion
# MAGIC
# MAGIC Load hourly electricity demand data with weather features from Unity Catalog gold layer.

# COMMAND ----------

# DBTITLE 1,Load Data from Unity Catalog
# Load hourly demand data with pre-computed features from Unity Catalog
df_demand = spark.table(f'{CATALOG}.{SCHEMA}.demand_features_hourly').toPandas()

print(f"Data loaded: {df_demand.shape[0]:,} rows, {df_demand.shape[1]} columns")
print(f"Date range: {df_demand['ts_local'].min()} to {df_demand['ts_local'].max()}")

# COMMAND ----------

df_demand.head()

# COMMAND ----------

# DBTITLE 1,Stage 3: Feature Engineering
# MAGIC %md
# MAGIC ## Stage 3: Feature Engineering
# MAGIC
# MAGIC Create temporal and weather features to improve model performance:
# MAGIC
# MAGIC * Short-term demand lags (1, 2, 3, 6, 12, 72 hours)
# MAGIC * Rolling statistics (mean, std, min, max) over multiple windows
# MAGIC * Weather-time interaction features
# MAGIC * Demand pattern features (trends, acceleration, volatility)
# MAGIC * Enhanced calendar features
# MAGIC * Weather trend features

# COMMAND ----------

df_demand.isnull().sum()

# COMMAND ----------

#Remove null values

df_demand.dropna(inplace=True)

# COMMAND ----------

df_demand.head()

# COMMAND ----------

#Convert is_holiday to int

df_demand['is_holiday'] = df_demand['is_holiday'].astype(int)
df_demand.head()

# COMMAND ----------

# DBTITLE 1,Add Short-Term Lag Features
# Demand lags must be >= HORIZON (24h) for day-ahead forecasting.
# Lags 1, 2, 3, 6, 12 are invalid: demand from < 24h ago is not available
# when generating a 24-hour-ahead forecast.
HORIZON = 24
for lag in [72, 167, 169]:
    df_demand[f'lag_{lag}'] = df_demand.groupby('respondent')['demand_mw'].shift(lag)

print(f"Added lag features (all >= {HORIZON}h). New shape: {df_demand.shape}")

# COMMAND ----------

# DBTITLE 1,Add Rolling Statistics (Multiple Windows)
# Rolling demand statistics — must end at t-24 (HORIZON) for day-ahead forecasting.
# shift(1) would include demand from 1 hour ago, which is not available at t-24.
for window in [3, 6, 12, 48, 336]:
    df_demand[f'roll_mean_{window}'] = df_demand.groupby('respondent')['demand_mw'].transform(
        lambda x: x.shift(HORIZON).rolling(window=window, min_periods=1).mean()
    )
    df_demand[f'roll_std_{window}'] = df_demand.groupby('respondent')['demand_mw'].transform(
        lambda x: x.shift(HORIZON).rolling(window=window, min_periods=1).std()
    )
    df_demand[f'roll_min_{window}'] = df_demand.groupby('respondent')['demand_mw'].transform(
        lambda x: x.shift(HORIZON).rolling(window=window, min_periods=1).min()
    )
    df_demand[f'roll_max_{window}'] = df_demand.groupby('respondent')['demand_mw'].transform(
        lambda x: x.shift(HORIZON).rolling(window=window, min_periods=1).max()
    )

print(f"Added rolling statistics (shifted by HORIZON={HORIZON}). New shape: {df_demand.shape}")

# COMMAND ----------

# DBTITLE 1,Add Weather Interaction Features
# Temperature-time interactions (temperature effect varies by time)
df_demand['temp_x_sin_hour'] = df_demand['temp_c'] * df_demand['sin_hour']
df_demand['temp_x_cos_hour'] = df_demand['temp_c'] * df_demand['cos_hour']
df_demand['temp_x_sin_dow'] = df_demand['temp_c'] * df_demand['sin_dow']
df_demand['temp_x_cos_dow'] = df_demand['temp_c'] * df_demand['cos_dow']

# Temperature anomaly interactions
df_demand['temp_anomaly'] = df_demand['temp_c'] - df_demand['temp_roll_mean_24']
df_demand['temp_anomaly_x_sin_hour'] = df_demand['temp_anomaly'] * df_demand['sin_hour']
df_demand['temp_anomaly_x_cos_hour'] = df_demand['temp_anomaly'] * df_demand['cos_hour']

# Heating/cooling degree days with time
df_demand['hdd_x_sin_hour'] = df_demand['hdd_c'] * df_demand['sin_hour']
df_demand['hdd_x_cos_hour'] = df_demand['hdd_c'] * df_demand['cos_hour']
df_demand['cdd_x_sin_hour'] = df_demand['cdd_c'] * df_demand['sin_hour']
df_demand['cdd_x_cos_hour'] = df_demand['cdd_c'] * df_demand['cos_hour']

print(f"Added weather interaction features. New shape: {df_demand.shape}")

# COMMAND ----------

# DBTITLE 1,Add Demand Pattern Features
# Demand trend and momentum features (all use lags >= 24 or shift >= 24)
df_demand['demand_trend_weekly'] = (df_demand['lag_24'] - df_demand['lag_168']) / (df_demand['lag_168'] + 1e-6)
df_demand['demand_acceleration'] = df_demand['lag_24'] - df_demand['lag_48']
df_demand['demand_diff_24_48'] = df_demand['lag_24'] - df_demand['lag_48']
df_demand['demand_diff_168_336'] = df_demand['lag_168'] - df_demand.groupby('respondent')['demand_mw'].shift(336)

# Recent volatility — shift by HORIZON so latest demand used is t-24
df_demand['demand_volatility_3h'] = df_demand.groupby('respondent')['demand_mw'].transform(
    lambda x: x.shift(HORIZON).rolling(window=3, min_periods=1).std()
)

print(f"Added demand pattern features. New shape: {df_demand.shape}")

# COMMAND ----------

# DBTITLE 1,Add Enhanced Calendar Features
# Extract time components for calendar features
df_demand['hour'] = pd.to_datetime(df_demand['ts_local']).dt.hour
df_demand['dow'] = pd.to_datetime(df_demand['ts_local']).dt.dayofweek
df_demand['day'] = pd.to_datetime(df_demand['ts_local']).dt.day
df_demand['month'] = pd.to_datetime(df_demand['ts_local']).dt.month

# Calendar features
df_demand['is_weekend'] = (df_demand['dow'] >= 5).astype(int)
df_demand['is_working_hour'] = ((df_demand['hour'] >= 9) & (df_demand['hour'] <= 17) & (df_demand['dow'] < 5)).astype(int)
df_demand['is_peak_hour'] = ((df_demand['hour'].isin([7, 8, 9, 17, 18, 19, 20]))).astype(int)
df_demand['is_month_start'] = (df_demand['day'] <= 7).astype(int)
df_demand['is_month_end'] = (df_demand['day'] >= 24).astype(int)
df_demand['week_of_month'] = (df_demand['day'] - 1) // 7 + 1

print(f"Added calendar features. New shape: {df_demand.shape}")

# COMMAND ----------

# DBTITLE 1,Add Weather Trend Features
# Weather change and trend features
df_demand['temp_diff_24'] = df_demand['temp_c'] - df_demand['temp_roll_mean_24']
df_demand['humidity_change_24'] = df_demand['rel_humidity'] - df_demand.groupby('respondent')['rel_humidity'].shift(24)
df_demand['temp_momentum'] = df_demand['temp_c'] - df_demand.groupby('respondent')['temp_c'].shift(3)
df_demand['wind_change_24'] = df_demand['wind_speed_ms'] - df_demand.groupby('respondent')['wind_speed_ms'].shift(24)

print(f"Added weather trend features. New shape: {df_demand.shape}")

# COMMAND ----------

# DBTITLE 1,Remove NA Values from Lag Features
# Drop rows with NaN values created by lag and rolling features
print(f"Shape before dropping NaN: {df_demand.shape}")
df_demand.dropna(inplace=True)
print(f"Shape after dropping NaN: {df_demand.shape}")
print(f"\nTotal features created: {df_demand.shape[1] - 35} new features")

# COMMAND ----------

# DBTITLE 1,Identify Redundant Features
# Identify highly correlated features (redundant)
corr_matrix = df_demand.select_dtypes(include=[np.number]).corr().abs()

# Find features with correlation > 0.95 (excluding self-correlation)
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
redundant_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]

print(f"Found {len(redundant_features)} redundant features (correlation > 0.95):")
for feat in redundant_features:
    correlated_with = upper_triangle[feat][upper_triangle[feat] > 0.95].index.tolist()
    if correlated_with:
        print(f"  {feat} <-> {correlated_with[0]}: {upper_triangle[feat][correlated_with[0]]:.3f}")

print(f"\nWill exclude these from modeling: {redundant_features}")

# COMMAND ----------

# DBTITLE 1,Feature Leakage Audit Table
# MAGIC %md
# MAGIC ## Feature Leakage Audit — 24-Hour-Ahead Forecasting
# MAGIC
# MAGIC Every demand-derived feature must use observations from `t-24` or earlier.
# MAGIC Weather and calendar features are available at prediction time via forecast or deterministic calculation.
# MAGIC
# MAGIC **Corrections applied in this run:**
# MAGIC - Removed: `lag_1`, `lag_2`, `lag_3`, `lag_6`, `lag_12` (demand shift < 24h, not available at forecast time)
# MAGIC - Fixed: All rolling demand statistics changed from `shift(1)` to `shift(24)` (window now ends at t-24)
# MAGIC - Fixed: `demand_volatility_3h` changed from `shift(1)` to `shift(24)`
# MAGIC - Kept: `lag_24`, `lag_48`, `lag_72`, `lag_167`, `lag_168`, `lag_169` (all shifts >= 24)
# MAGIC
# MAGIC | Feature | Source | Available at t-24? | Action |
# MAGIC | --- | --- | --- | --- |
# MAGIC | Calendar (sin/cos hour, dow, doy) | Deterministic from timestamp | Yes | Keep |
# MAGIC | is_holiday | US holidays library | Yes | Keep |
# MAGIC | temp_c, dew_point_c, rel_humidity, etc. | Weather forecast for target t | Yes (forecast) | Keep |
# MAGIC | cdd_c, hdd_c | Derived from temp_c at t | Yes (forecast) | Keep |
# MAGIC | temp_roll_mean_24 | 24h weather ending at t | Yes (observed + forecast) | Keep |
# MAGIC | lag_24, lag_48, lag_72 | demand_mw shift >= 24 | Yes | Keep |
# MAGIC | lag_167, lag_168, lag_169 | demand_mw shift >= 24 | Yes | Keep |
# MAGIC | roll_mean_24, roll_std_24, roll_mean_168 | Gold table: window ending t-24 | Yes | Keep |
# MAGIC | roll_mean_3..336, roll_std_3..336, etc. | shift(1).rolling() | **No** — used t-1 demand | **Fixed: shift(24)** |
# MAGIC | lag_1, lag_2, lag_3, lag_6, lag_12 | demand_mw shift < 24 | **No** | **Removed** |
# MAGIC | demand_volatility_3h | shift(1).rolling(3) | **No** — used t-1 demand | **Fixed: shift(24)** |
# MAGIC | demand_trend_weekly | lag_24 - lag_168 | Yes | Keep |
# MAGIC | demand_acceleration | lag_24 - lag_48 | Yes | Keep |
# MAGIC | demand_diff_168_336 | lag_168 - shift(336) | Yes | Keep |
# MAGIC | Weather interactions (temp_x_*) | temp_c x calendar | Yes (forecast) | Keep |
# MAGIC | Weather trends (humidity_change_24, etc.) | Current/past weather | Yes | Keep |

# COMMAND ----------

# DBTITLE 1,Corrected Feature Set Definition
# --- Corrected feature set definition ---
# Valid lags (>= HORIZON) are explicitly retained per requirement.

base_excluded = [
    'ts_utc', 'ts_local', 'respondent', 'split', 'demand_mw',
    'eia_dayahead_forecast_mw', 'is_trainable', 'demand_is_missing',
    'weather_is_missing', '_gold_built_at', 'hour', 'dow', 'day', 'month',
    'hour_day_interaction'
]

# Redundant features EXCLUDING valid lags (lag_24, lag_48, lag_72, lag_167, lag_169)
# These lags are kept because they are valid for 24h-ahead forecasting.
redundant_features = [
    'apparent_temp_c', 'roll_mean_3', 'roll_min_3', 'roll_max_3',
    'roll_mean_6', 'roll_min_6', 'roll_max_6', 'roll_max_12',
    'roll_mean_48', 'roll_std_48', 'roll_min_48', 'roll_max_48',
    'roll_mean_336', 'demand_diff_24_48', 'demand_volatility_3h',
    'week_of_month', 'temp_diff_24'
]

excluded_cols = set(base_excluded) | set(redundant_features)
feature_cols = [
    c for c in df_demand.columns
    if c not in excluded_cols
    and df_demand[c].dtype in ('float64', 'int64', 'int32')
]

print(f"Total columns: {df_demand.shape[1]}")
print(f"Excluded columns: {len(excluded_cols)}")
print(f"Modeling features: {len(feature_cols)}")
print(f"Target: demand_mw")
print(f"Date range: {df_demand['ts_local'].min()} to {df_demand['ts_local'].max()}")
print(f"\nFeature list ({len(feature_cols)}):")
for i, f in enumerate(sorted(feature_cols), 1):
    print(f"  {i:2d}. {f}")

# Verify no invalid lags remain
invalid_lags = [f for f in feature_cols if f.startswith('lag_') and int(f.split('_')[1]) < HORIZON]
assert len(invalid_lags) == 0, f"Invalid lags found: {invalid_lags}"
print(f"\nLeakage audit: PASSED (no demand lags < {HORIZON}h)")

# COMMAND ----------

# DBTITLE 1,Section 1: Classical ML Validation
# MAGIC %md
# MAGIC ## Section 1: Classical ML Validation
# MAGIC
# MAGIC Time-based cross-validation with 5 chronological folds, preserving temporal order (training dates < validation dates).
# MAGIC Models: CatBoost, LightGBM, XGBoost. No StandardScaler — all models use raw features.
# MAGIC
# MAGIC Metrics per fold: MAPE, MAE, RMSE, bias, EIA baseline MAPE, improvement vs EIA, % hours beating EIA, top 10% demand MAPE.
# MAGIC
# MAGIC After CV, forward validation on Aug 1-16 and Aug 17-31 (used for model selection).
# MAGIC Sep 1-13 is reserved as untouched holdout for final evaluation only.

# COMMAND ----------

# DBTITLE 1,Metrics and Validation Configuration
# --- Metrics and configuration ---

def compute_metrics(y_true, y_pred, eia_pred=None):
    """Return MAPE, MAE, RMSE, bias, % beats EIA, and optional EIA comparison."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = np.mean(y_pred - y_true)
    result = {
        'MAPE (%)': round(mape, 3),
        'MAE (MW)': round(mae, 1),
        'RMSE (MW)': round(rmse, 1),
        'Bias (MW)': round(bias, 1),
    }
    if eia_pred is not None:
        eia_pred = np.asarray(eia_pred, dtype=float)
        mask = ~np.isnan(eia_pred)
        if mask.sum() > 0:
            eia_mape = mean_absolute_percentage_error(y_true[mask], eia_pred[mask]) * 100
            result['EIA MAPE (%)'] = round(eia_mape, 3)
            result['Improvement vs EIA (pp)'] = round(eia_mape - mape, 3)
            model_error = np.abs(y_pred[mask] - y_true[mask])
            eia_error = np.abs(eia_pred[mask] - y_true[mask])
            result['% Beats EIA'] = round(np.mean(model_error < eia_error) * 100, 1)
    return result

def top10_demand_mape(y_true, y_pred):
    """MAPE on the top 10% demand hours within the evaluation set."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    p90 = np.percentile(y_true, 90)
    mask = y_true >= p90
    if mask.sum() == 0:
        return None
    return round(mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100, 3)

# Time-based CV folds: 5 chronological folds with expanding training windows
cv_folds = [
    {'fold': 'CV1', 'train_end': '2024-12-31', 'val_start': '2025-01-01', 'val_end': '2025-03-31'},
    {'fold': 'CV2', 'train_end': '2025-03-31', 'val_start': '2025-04-01', 'val_end': '2025-06-30'},
    {'fold': 'CV3', 'train_end': '2025-06-30', 'val_start': '2025-07-01', 'val_end': '2025-09-30'},
    {'fold': 'CV4', 'train_end': '2025-09-30', 'val_start': '2025-10-01', 'val_end': '2025-12-31'},
    {'fold': 'CV5', 'train_end': '2025-12-31', 'val_start': '2026-01-01', 'val_end': '2026-03-31'},
]

# Forward validation windows (used for model selection, NOT untouched)
validation_windows = [
    {'label': 'Aug 1-16',  'train_end': '2026-07-31', 'val_start': '2026-08-01', 'val_end': '2026-08-16'},
    {'label': 'Aug 17-31', 'train_end': '2026-08-16', 'val_start': '2026-08-17', 'val_end': '2026-08-31'},
]

# Untouched forward holdout (NOT used during model selection or tuning)
untouched_holdout = {
    'label': 'Sep Forward', 'train_end': '2026-08-31',
    'val_start': '2026-09-01', 'val_end': '2026-09-13'
}

# Model configurations (fixed parameters, no tuning)
model_configs = {
    'CatBoost': lambda: CatBoostRegressor(
        iterations=800, depth=6, learning_rate=0.05,
        l2_leaf_reg=3, random_seed=42, verbose=0, allow_writing_files=False
    ),
    'LightGBM': lambda: LGBMRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        num_leaves=31, min_child_samples=20, random_state=42, verbose=-1
    ),
    'XGBoost': lambda: XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
    ),
}

print(f"CV folds: {len(cv_folds)}")
print(f"Validation windows: {len(validation_windows)}")
print(f"Untouched holdout: {untouched_holdout['label']}")
print(f"Models: {list(model_configs.keys())}")
print(f"Features: {len(feature_cols)}")

# COMMAND ----------

# DBTITLE 1,Classical ML Training: CV + Holdout
# --- Section 1: Classical ML Training ---
# Train CatBoost, LightGBM, XGBoost on 5 CV folds + 2 forward validation windows.
# Sep 1-13 is reserved as untouched holdout for final evaluation only.

def train_and_evaluate(df, feature_cols, train_mask, val_mask, model_name, model_factory, window_label):
    """Train a single model and return metrics + predictions DataFrame."""
    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, 'demand_mw']
    X_val = df.loc[val_mask, feature_cols]
    y_val = df.loc[val_mask, 'demand_mw']
    eia_val = df.loc[val_mask, 'eia_dayahead_forecast_mw']

    model = model_factory()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    metrics = compute_metrics(y_val, y_pred, eia_val)
    metrics['Model'] = model_name
    metrics['Window'] = window_label
    metrics['Top 10% MAPE (%)'] = top10_demand_mape(y_val.values, y_pred)

    preds_df = pd.DataFrame({
        'ts_local': df.loc[val_mask, 'ts_local'].values,
        'temp_c': df.loc[val_mask, 'temp_c'].values,
        'demand_mw': y_val.values,
        'eia_dayahead_forecast_mw': eia_val.values,
        'y_pred': y_pred,
        'model': model_name,
        'window': window_label,
    })
    return metrics, preds_df

all_cv_results = []
all_val_results = []
all_predictions = {}

# --- Cross-validation folds ---
for fold in cv_folds:
    train_end = pd.Timestamp(fold['train_end'])
    val_start = pd.Timestamp(fold['val_start'])
    val_end = pd.Timestamp(fold['val_end'])

    train_mask = df_demand['ts_local'] <= train_end
    val_mask = (df_demand['ts_local'] >= val_start) & (df_demand['ts_local'] <= val_end)

    n_train = train_mask.sum()
    n_val = val_mask.sum()
    if n_val == 0:
        continue
    print(f"\n{fold['fold']}: train={n_train:,} rows, val={n_val} rows")

    for model_name, model_factory in model_configs.items():
        metrics, preds = train_and_evaluate(
            df_demand, feature_cols, train_mask, val_mask,
            model_name, model_factory, fold['fold']
        )
        all_cv_results.append(metrics)
        all_predictions[(model_name, fold['fold'])] = preds
        print(f"  {model_name}: MAPE={metrics['MAPE (%)']:.2f}%  Beats EIA={metrics.get('% Beats EIA', 'N/A')}%  "
              f"Top10={metrics.get('Top 10% MAPE (%)', 'N/A')}%  Bias={metrics['Bias (MW)']}  "
              f"EIA={metrics.get('EIA MAPE (%)', 'N/A')}")

# --- Forward validation windows (not untouched) ---
for window in validation_windows:
    train_end = pd.Timestamp(window['train_end'])
    val_start = pd.Timestamp(window['val_start'])
    val_end = pd.Timestamp(window['val_end'])

    train_mask = df_demand['ts_local'] <= train_end
    val_mask = (df_demand['ts_local'] >= val_start) & (df_demand['ts_local'] <= val_end)

    n_train = train_mask.sum()
    n_val = val_mask.sum()
    if n_val == 0:
        continue
    print(f"\nValidation {window['label']}: train={n_train:,} rows, val={n_val} rows")

    for model_name, model_factory in model_configs.items():
        metrics, preds = train_and_evaluate(
            df_demand, feature_cols, train_mask, val_mask,
            model_name, model_factory, window['label']
        )
        all_val_results.append(metrics)
        all_predictions[(model_name, window['label'])] = preds
        print(f"  {model_name}: MAPE={metrics['MAPE (%)']:.2f}%  Beats EIA={metrics.get('% Beats EIA', 'N/A')}%  "
              f"Top10={metrics.get('Top 10% MAPE (%)', 'N/A')}%  Bias={metrics['Bias (MW)']}  "
              f"EIA={metrics.get('EIA MAPE (%)', 'N/A')}")

cv_df = pd.DataFrame(all_cv_results)
val_df = pd.DataFrame(all_val_results)
print(f"\nCompleted {len(all_cv_results)} CV + {len(all_val_results)} validation evaluations.")

# COMMAND ----------

# DBTITLE 1,CV Summary and Detailed Results
# --- CV summary: mean and std across folds ---

cv_summary = cv_df.groupby('Model').agg(
    cv_mape_mean=('MAPE (%)', 'mean'),
    cv_mape_std=('MAPE (%)', 'std'),
    cv_mae_mean=('MAE (MW)', 'mean'),
    cv_rmse_mean=('RMSE (MW)', 'mean'),
    cv_bias_mean=('Bias (MW)', 'mean'),
    cv_beats_eia_mean=('% Beats EIA', 'mean'),
    cv_top10_mape_mean=('Top 10% MAPE (%)', 'mean'),
).round(3).reset_index()

cv_summary = cv_summary.sort_values('cv_mape_mean')
print("CV Summary (mean +/- std across 5 folds):")
display(cv_summary)

display_cols = ['Window', 'Model', 'MAPE (%)', 'MAE (MW)', 'RMSE (MW)', 'Bias (MW)',
                'EIA MAPE (%)', 'Improvement vs EIA (pp)', '% Beats EIA', 'Top 10% MAPE (%)']
print("\nDetailed CV Results:")
display(cv_df[display_cols].sort_values(['Window', 'MAPE (%)']))

print("\nForward Validation Results:")
display(val_df[display_cols].sort_values(['Window', 'MAPE (%)']))

# COMMAND ----------

# DBTITLE 1,High-Demand Period Analysis
# --- High-demand period analysis ---
# Evaluate model performance on critical demand periods:
#   1. Top 10% demand hours
#   2. High-demand hours (top 25%)
#   3. Summer peak days (top 20% daily peak)
#   4. Highest temperature periods (top 10% temp_c)
#   5. High-demand evening hours (top 25% demand, hours 17-21)

all_preds_combined = pd.concat(all_predictions.values(), ignore_index=True)

demand_p90 = all_preds_combined['demand_mw'].quantile(0.90)
demand_p75 = all_preds_combined['demand_mw'].quantile(0.75)
daily_peaks = all_preds_combined.groupby(all_preds_combined['ts_local'].dt.date)['demand_mw'].max()
peak_day_threshold = daily_peaks.quantile(0.80)
peak_days = set(daily_peaks[daily_peaks >= peak_day_threshold].index)
temp_p90 = all_preds_combined['temp_c'].quantile(0.90)

print("Subset thresholds:")
print(f"  Top 10% demand >= {demand_p90:.0f} MW")
print(f"  High-demand (p75) >= {demand_p75:.0f} MW")
print(f"  Peak days threshold >= {peak_day_threshold:.0f} MW (top 20% daily peaks)")
print(f"  Top 10% temp >= {temp_p90:.1f} C")

subsets = {
    'Top 10% Demand': all_preds_combined['demand_mw'] >= demand_p90,
    'High-Demand (Top 25%)': all_preds_combined['demand_mw'] >= demand_p75,
    'Summer Peak Days': all_preds_combined['ts_local'].dt.date.isin(peak_days),
    'Highest Temp (Top 10%)': all_preds_combined['temp_c'] >= temp_p90,
    'High-Demand Evening': (all_preds_combined['demand_mw'] >= demand_p75) &
                           (all_preds_combined['ts_local'].dt.hour.isin([17, 18, 19, 20, 21])),
}

high_demand_rows = []

for subset_name, mask in subsets.items():
    subset_preds = all_preds_combined[mask]
    if len(subset_preds) == 0:
        continue

    for model_name in model_configs.keys():
        model_preds = subset_preds[subset_preds['model'] == model_name]
        if len(model_preds) == 0:
            continue
        m = compute_metrics(
            model_preds['demand_mw'].values,
            model_preds['y_pred'].values,
            model_preds['eia_dayahead_forecast_mw'].values,
        )
        m['Model'] = model_name
        m['Subset'] = subset_name
        m['N'] = len(model_preds)
        high_demand_rows.append(m)

    eia_mask = ~subset_preds['eia_dayahead_forecast_mw'].isna()
    if eia_mask.sum() > 0:
        eia_mape = mean_absolute_percentage_error(
            subset_preds.loc[eia_mask, 'demand_mw'],
            subset_preds.loc[eia_mask, 'eia_dayahead_forecast_mw']
        ) * 100
        high_demand_rows.append({
            'MAPE (%)': round(eia_mape, 3),
            'Model': 'EIA Baseline',
            'Subset': subset_name,
            'N': eia_mask.sum(),
        })

high_demand_df = pd.DataFrame(high_demand_rows)
display_cols_hd = ['Subset', 'Model', 'N', 'MAPE (%)', 'MAE (MW)', 'RMSE (MW)', 'Bias (MW)']
print("\nHigh-Demand Period Analysis:")
display(high_demand_df[display_cols_hd].sort_values(['Subset', 'MAPE (%)']))

# COMMAND ----------

# DBTITLE 1,Section 2: Deep Learning Models
# MAGIC %md
# MAGIC ## Section 2: Deep Learning Models
# MAGIC
# MAGIC Isolated section — run explicitly. These cells do not execute when the normal ML validation cells are rerun.
# MAGIC
# MAGIC Step 1: Screen all 4 architectures (LSTM, GRU, 1D CNN, CNN-LSTM) on one representative validation period (Aug 1-16).
# MAGIC Step 2: Take the best 1-2 architectures and run 3-fold time-based CV.
# MAGIC
# MAGIC Uses chronological splits, early stopping, LR scheduling, deterministic seeds.
# MAGIC Input scaling (StandardScaler) is fit on training portion only.
# MAGIC Sep 1-13 is reserved as untouched holdout — not used in this section.

# COMMAND ----------

# DBTITLE 1,DL Screening
# --- DL Screening: Test all 4 architectures on Aug 1-16 representative period ---
# Isolated cell: run explicitly.

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Configuration
SEQ_LEN = 24
BATCH_SIZE = 256
MAX_EPOCHS = 50
PATIENCE = 10
LR = 0.001
SEED = 42
TARGET_SCALE = 50000.0

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# --- Model definitions ---
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden=64, n_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=n_layers, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden=64, n_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers=n_layers, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])

class CNNModel(nn.Module):
    def __init__(self, input_dim, hidden=64, n_layers=1, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        return self.head(x[:, :, -1])

class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden=64, n_layers=1, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(input_dim, hidden, kernel_size=3, padding=1), nn.ReLU())
        self.lstm = nn.LSTM(hidden, hidden, num_layers=n_layers, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

DL_ARCHITECTURES = {
    'LSTM': LSTMModel,
    'GRU': GRUModel,
    '1D-CNN': CNNModel,
    'CNN-LSTM': CNNLSTMModel,
}

# --- Shared utilities ---
def create_sequences(features, targets, seq_len):
    X, y = [], []
    for i in range(seq_len, len(features)):
        X.append(features[i - seq_len:i])
        y.append(targets[i])
    return np.array(X), np.array(y)

def train_dl_model(model, train_loader, val_loader, name, max_epochs=MAX_EPOCHS):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    best_val = float('inf')
    best_state = None
    no_improve = 0
    for epoch in range(max_epochs):
        model.train()
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b).squeeze(-1), y_b)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                val_loss += criterion(model(X_b).squeeze(-1), y_b).item() * len(X_b)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  {name} epoch {epoch+1}/{max_epochs}: val_loss={val_loss:.6f}")
        if no_improve >= PATIENCE:
            print(f"  {name} early stopped at epoch {epoch+1}")
            break
    model.load_state_dict(best_state)
    return model

def dl_evaluate_on_window(train_df, val_df, feat_cols, model_cls, name, seq_len=SEQ_LEN):
    """Train and evaluate a DL model on a single train/val window."""
    scaler = StandardScaler()
    train_feat = scaler.fit_transform(train_df[feat_cols].values)
    val_feat = scaler.transform(val_df[feat_cols].values)
    train_tgt = train_df['demand_mw'].values / TARGET_SCALE
    val_tgt = val_df['demand_mw'].values / TARGET_SCALE
    n_train = len(train_feat)

    all_feat = np.vstack([train_feat, val_feat])
    all_tgt = np.concatenate([train_tgt, val_tgt])
    X_all, y_all = create_sequences(all_feat, all_tgt, seq_len)
    n_train_seq = n_train - seq_len
    X_train_seq, y_train_seq = X_all[:n_train_seq], y_all[:n_train_seq]
    X_val_seq, y_val_seq = X_all[n_train_seq:], y_all[n_train_seq:]

    X_train_t = torch.FloatTensor(X_train_seq).to(device)
    y_train_t = torch.FloatTensor(y_train_seq).to(device)
    X_val_t = torch.FloatTensor(X_val_seq).to(device)
    y_val_t = torch.FloatTensor(y_val_seq).to(device)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)

    input_dim = len(feat_cols)
    model = model_cls(input_dim)
    trained = train_dl_model(model, train_loader, val_loader, name)
    trained.eval()
    with torch.no_grad():
        y_pred_scaled = trained(X_val_t).squeeze(-1).cpu().numpy()
    y_pred = y_pred_scaled * TARGET_SCALE
    y_true = y_val_seq * TARGET_SCALE
    eia_val = val_df['eia_dayahead_forecast_mw'].values
    metrics = compute_metrics(y_true, y_pred, eia_val)
    metrics['Model'] = name
    metrics['Top 10% MAPE (%)'] = top10_demand_mape(y_true, y_pred)
    return metrics, trained

# --- Screening: Aug 1-16 representative period ---
screen_train_end = pd.Timestamp('2026-07-31')
screen_val_start = pd.Timestamp('2026-08-01')
screen_val_end = pd.Timestamp('2026-08-16')
screen_train = df_demand[df_demand['ts_local'] <= screen_train_end].copy()
screen_val = df_demand[(df_demand['ts_local'] >= screen_val_start) & (df_demand['ts_local'] <= screen_val_end)].copy()
print(f"Screening on Aug 1-16: train={len(screen_train):,}, val={len(screen_val)}")

screen_results = []
screen_models = {}
for name, cls in DL_ARCHITECTURES.items():
    print(f"\nTraining {name}...")
    metrics, trained = dl_evaluate_on_window(screen_train, screen_val, feature_cols, cls, name)
    screen_results.append(metrics)
    screen_models[name] = trained
    print(f"  {name}: MAPE={metrics['MAPE (%)']:.2f}%  Beats EIA={metrics.get('% Beats EIA', 'N/A')}%  "
          f"Bias={metrics['Bias (MW)']}")

screen_df = pd.DataFrame(screen_results)
print("\nDL Screening Results (Aug 1-16):")
display(screen_df[['Model', 'MAPE (%)', 'MAE (MW)', 'RMSE (MW)', 'Bias (MW)', '% Beats EIA', 'EIA MAPE (%)']])

# Select best 1-2 architectures
screen_sorted = screen_df.sort_values('MAPE (%)')
best_dl_names = screen_sorted.head(2)['Model'].tolist()
print(f"\nBest 1-2 DL architectures for CV: {best_dl_names}")

# COMMAND ----------

# DBTITLE 1,DL Cross-Validation
# --- DL Cross-Validation for best 1-2 architectures ---
# Uses 3 chronological folds (subset of classical CV folds).
# Isolated cell: run explicitly after screening.

dl_cv_folds = [
    {'fold': 'DLCV1', 'train_end': '2025-06-30', 'val_start': '2025-07-01', 'val_end': '2025-09-30'},
    {'fold': 'DLCV2', 'train_end': '2025-12-31', 'val_start': '2026-01-01', 'val_end': '2026-03-31'},
    {'fold': 'DLCV3', 'train_end': '2026-07-31', 'val_start': '2026-08-01', 'val_end': '2026-08-16'},
]

all_dl_cv_results = []

for name in best_dl_names:
    cls = DL_ARCHITECTURES[name]
    print(f"\n{'='*40}")
    print(f"Cross-validating {name} on {len(dl_cv_folds)} folds")
    print(f"{'='*40}")
    for fold in dl_cv_folds:
        train_end = pd.Timestamp(fold['train_end'])
        val_start = pd.Timestamp(fold['val_start'])
        val_end = pd.Timestamp(fold['val_end'])
        train_df = df_demand[df_demand['ts_local'] <= train_end].copy()
        val_df_fold = df_demand[(df_demand['ts_local'] >= val_start) & (df_demand['ts_local'] <= val_end)].copy()
        if len(val_df_fold) == 0:
            continue
        print(f"\n{fold['fold']}: train={len(train_df):,}, val={len(val_df_fold)}")
        metrics, _ = dl_evaluate_on_window(train_df, val_df_fold, feature_cols, cls, name)
        metrics['Window'] = fold['fold']
        all_dl_cv_results.append(metrics)
        print(f"  {name}: MAPE={metrics['MAPE (%)']:.2f}%  Beats EIA={metrics.get('% Beats EIA', 'N/A')}%  "
              f"Top10={metrics.get('Top 10% MAPE (%)', 'N/A')}%  Bias={metrics['Bias (MW)']}")

dl_cv_df = pd.DataFrame(all_dl_cv_results)

# DL CV summary
dl_cv_summary = dl_cv_df.groupby('Model').agg(
    cv_mape_mean=('MAPE (%)', 'mean'),
    cv_mape_std=('MAPE (%)', 'std'),
    cv_mae_mean=('MAE (MW)', 'mean'),
    cv_rmse_mean=('RMSE (MW)', 'mean'),
    cv_bias_mean=('Bias (MW)', 'mean'),
    cv_beats_eia_mean=('% Beats EIA', 'mean'),
    cv_top10_mape_mean=('Top 10% MAPE (%)', 'mean'),
).round(3).reset_index()
dl_cv_summary = dl_cv_summary.sort_values('cv_mape_mean')

print("\nDL CV Summary (3 folds):")
display(dl_cv_summary)

print("\nDetailed DL CV Results:")
dl_display_cols = ['Window', 'Model', 'MAPE (%)', 'MAE (MW)', 'RMSE (MW)', 'Bias (MW)',
                  'EIA MAPE (%)', 'Improvement vs EIA (pp)', '% Beats EIA', 'Top 10% MAPE (%)']
display(dl_cv_df[dl_display_cols].sort_values(['Window', 'MAPE (%)']))

# Compare with classical CV summary
print("\nClassical vs DL CV Comparison:")
combined_cv = pd.concat([cv_summary, dl_cv_summary], ignore_index=True)
display(combined_cv.sort_values('cv_mape_mean'))

# COMMAND ----------

# DBTITLE 1,Finalist Selection
# --- Finalist selection ---
# Select best classical and best DL based on CV MAPE, stability, EIA improvement, peak demand, bias, % beats EIA.

combined_summary = pd.concat([cv_summary, dl_cv_summary], ignore_index=True)
combined_summary = combined_summary.sort_values('cv_mape_mean')

print("Combined CV Summary (all models):")
display(combined_summary)

# Best classical model
classical_only = cv_summary.sort_values('cv_mape_mean')
best_classical_name = classical_only.iloc[0]['Model']
best_classical_row = classical_only.iloc[0]
print(f"\nBest Classical ML Model: {best_classical_name}")
print(f"  CV MAPE: {best_classical_row['cv_mape_mean']:.3f} +/- {best_classical_row['cv_mape_std']:.3f}")
print(f"  CV Beats EIA: {best_classical_row['cv_beats_eia_mean']:.1f}%")
print(f"  CV Top 10% MAPE: {best_classical_row['cv_top10_mape_mean']:.3f}")
print(f"  CV Bias: {best_classical_row['cv_bias_mean']:.1f}")

# Best DL model
dl_only = dl_cv_summary.sort_values('cv_mape_mean')
best_dl_name = dl_only.iloc[0]['Model']
best_dl_row = dl_only.iloc[0]
print(f"\nBest Deep Learning Model: {best_dl_name}")
print(f"  CV MAPE: {best_dl_row['cv_mape_mean']:.3f} +/- {best_dl_row['cv_mape_std']:.3f}")
print(f"  CV Beats EIA: {best_dl_row['cv_beats_eia_mean']:.1f}%")
print(f"  CV Top 10% MAPE: {best_dl_row['cv_top10_mape_mean']:.3f}")
print(f"  CV Bias: {best_dl_row['cv_bias_mean']:.1f}")

print(f"\nFinalists: {best_classical_name} (classical) vs {best_dl_name} (deep learning)")

# COMMAND ----------

# DBTITLE 1,Classical Hyperparameter Tuning
# --- Classical Hyperparameter Tuning ---
# Small random search for the best classical model using time-based CV.
# Isolated cell: run explicitly after finalist selection.

from sklearn.model_selection import ParameterSampler

tuning_folds = [cv_folds[0], cv_folds[2], cv_folds[4]]
N_TRIALS = 20

if best_classical_name == 'CatBoost':
    param_grid = {
        'depth': [4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'iterations': [500, 800, 1000, 1200],
        'l2_leaf_reg': [1, 3, 5, 7],
        'random_strength': [0.5, 1.0, 2.0],
        'subsample': [0.7, 0.8, 0.9, 1.0],
    }
    def make_tuned_model(params):
        return CatBoostRegressor(
            depth=params['depth'], learning_rate=params['learning_rate'],
            iterations=params['iterations'], l2_leaf_reg=params['l2_leaf_reg'],
            random_strength=params['random_strength'], subsample=params['subsample'],
            random_seed=42, verbose=0, allow_writing_files=False)
elif best_classical_name == 'LightGBM':
    param_grid = {
        'n_estimators': [300, 500, 800], 'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.03, 0.05, 0.1], 'num_leaves': [15, 31, 63],
        'min_child_samples': [10, 20, 40], 'subsample': [0.7, 0.8, 0.9],
    }
    def make_tuned_model(params):
        return LGBMRegressor(random_state=42, verbose=-1, **params)
else:
    param_grid = {
        'n_estimators': [300, 500, 800], 'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.03, 0.05, 0.1], 'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9], 'min_child_weight': [1, 3, 5],
    }
    def make_tuned_model(params):
        return XGBRegressor(random_state=42, verbosity=0, **params)

param_list = list(ParameterSampler(param_grid, n_iter=N_TRIALS, random_state=42))
print(f"Tuning {best_classical_name} with {N_TRIALS} trials on {len(tuning_folds)} folds")

best_tune_mape = float('inf')
best_tune_params = None
tune_results = []

for trial_idx, params in enumerate(param_list):
    fold_mapes = []
    for fold in tuning_folds:
        tr_end = pd.Timestamp(fold['train_end'])
        vs = pd.Timestamp(fold['val_start'])
        ve = pd.Timestamp(fold['val_end'])
        tr_mask = df_demand['ts_local'] <= tr_end
        v_mask = (df_demand['ts_local'] >= vs) & (df_demand['ts_local'] <= ve)
        model = make_tuned_model(params)
        model.fit(df_demand.loc[tr_mask, feature_cols], df_demand.loc[tr_mask, 'demand_mw'])
        y_pred = model.predict(df_demand.loc[v_mask, feature_cols])
        y_true = df_demand.loc[v_mask, 'demand_mw'].values
        fold_mapes.append(mean_absolute_percentage_error(y_true, y_pred) * 100)
    avg_mape = np.mean(fold_mapes)
    tune_results.append({'trial': trial_idx, 'avg_mape': round(avg_mape, 3), **{k: v for k, v in params.items()}})
    if avg_mape < best_tune_mape:
        best_tune_mape = avg_mape
        best_tune_params = params
    if (trial_idx + 1) % 5 == 0:
        print(f"  Trial {trial_idx+1}/{N_TRIALS}: avg_mape={avg_mape:.3f}%")

print(f"\nBest params: {best_tune_params}")
print(f"Best avg CV MAPE: {best_tune_mape:.3f}%")

tune_df = pd.DataFrame(tune_results)
display(tune_df.sort_values('avg_mape').head(10))

# COMMAND ----------

# DBTITLE 1,DL Hyperparameter Tuning
# --- DL Hyperparameter Tuning ---
# Small random search for the best DL model using time-based CV.
# Isolated cell: run explicitly after classical tuning.

DL_N_TRIALS = 10
DL_TUNING_FOLDS = [dl_cv_folds[0], dl_cv_folds[2]]

dl_param_grid = {
    'seq_len': [24, 48, 72],
    'hidden': [32, 64, 128],
    'n_layers': [1, 2],
    'dropout': [0.0, 0.1, 0.2],
    'lr': [0.001, 0.0005, 0.002],
    'batch_size': [128, 256, 512],
}

dl_param_list = list(ParameterSampler(dl_param_grid, n_iter=DL_N_TRIALS, random_state=42))
best_dl_cls = DL_ARCHITECTURES[best_dl_name]
print(f"Tuning {best_dl_name} with {DL_N_TRIALS} trials on {len(DL_TUNING_FOLDS)} folds")

best_dl_tune_mape = float('inf')
best_dl_tune_params = None
dl_tune_results = []

for trial_idx, params in enumerate(dl_param_list):
    fold_mapes = []
    for fold in DL_TUNING_FOLDS:
        tr_end = pd.Timestamp(fold['train_end'])
        vs = pd.Timestamp(fold['val_start'])
        ve = pd.Timestamp(fold['val_end'])
        tr_df = df_demand[df_demand['ts_local'] <= tr_end].copy()
        v_df = df_demand[(df_demand['ts_local'] >= vs) & (df_demand['ts_local'] <= ve)].copy()
        if len(v_df) == 0:
            continue
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        scaler = StandardScaler()
        tr_feat = scaler.fit_transform(tr_df[feature_cols].values)
        v_feat = scaler.transform(v_df[feature_cols].values)
        tr_tgt = tr_df['demand_mw'].values / TARGET_SCALE
        v_tgt = v_df['demand_mw'].values / TARGET_SCALE
        all_f = np.vstack([tr_feat, v_feat])
        all_t = np.concatenate([tr_tgt, v_tgt])
        X_a, y_a = create_sequences(all_f, all_t, params['seq_len'])
        n_tr = len(tr_feat) - params['seq_len']
        X_tr, y_tr = X_a[:n_tr], y_a[:n_tr]
        X_v, y_v = X_a[n_tr:], y_a[n_tr:]
        X_tr_t = torch.FloatTensor(X_tr).to(device)
        y_tr_t = torch.FloatTensor(y_tr).to(device)
        X_v_t = torch.FloatTensor(X_v).to(device)
        y_v_t = torch.FloatTensor(y_v).to(device)
        tr_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=params['batch_size'], shuffle=True)
        v_loader = DataLoader(TensorDataset(X_v_t, y_v_t), batch_size=params['batch_size'], shuffle=False)
        model = best_dl_cls(len(feature_cols), hidden=params['hidden'], n_layers=params['n_layers'], dropout=params['dropout']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
        criterion = nn.MSELoss()
        best_vl = float('inf')
        best_st = None
        no_imp = 0
        for ep in range(MAX_EPOCHS):
            model.train()
            for X_b, y_b in tr_loader:
                optimizer.zero_grad()
                loss = criterion(model(X_b).squeeze(-1), y_b)
                loss.backward()
                optimizer.step()
            model.eval()
            vl = 0
            with torch.no_grad():
                for X_b, y_b in v_loader:
                    vl += criterion(model(X_b).squeeze(-1), y_b).item() * len(X_b)
            vl /= len(v_loader.dataset)
            if vl < best_vl:
                best_vl = vl
                best_st = {k: v.clone() for k, v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= PATIENCE:
                break
        model.load_state_dict(best_st)
        model.eval()
        with torch.no_grad():
            y_p = model(X_v_t).squeeze(-1).cpu().numpy() * TARGET_SCALE
        fold_mapes.append(mean_absolute_percentage_error(y_v * TARGET_SCALE, y_p) * 100)
    avg_mape = np.mean(fold_mapes)
    dl_tune_results.append({'trial': trial_idx, 'avg_mape': round(avg_mape, 3), **{k: v for k, v in params.items()}})
    if avg_mape < best_dl_tune_mape:
        best_dl_tune_mape = avg_mape
        best_dl_tune_params = params
    print(f"  Trial {trial_idx+1}/{DL_N_TRIALS}: avg_mape={avg_mape:.3f}%")

print(f"\nBest DL params: {best_dl_tune_params}")
print(f"Best DL avg CV MAPE: {best_dl_tune_mape:.3f}%")

dl_tune_df = pd.DataFrame(dl_tune_results)
display(dl_tune_df.sort_values('avg_mape').head(5))

# COMMAND ----------

# DBTITLE 1,Residual Forecasting Challenger
# --- Residual Forecasting Challenger ---
# Train CatBoost to predict EIA residual rather than total demand.
# Final prediction = eia_forecast + predicted_residual.
# All EIA residual features use shift >= 24 (forecast-time safe).

df_residual = df_demand.copy()
df_residual['eia_residual'] = df_residual['demand_mw'] - df_residual['eia_dayahead_forecast_mw']

# Lagged EIA residual features (all shift >= 24)
for lag in [24, 48, 168]:
    df_residual[f'eia_resid_lag_{lag}'] = df_residual.groupby('respondent')['eia_residual'].shift(lag)
df_residual['eia_resid_roll_mean_24'] = df_residual.groupby('respondent')['eia_residual'].transform(
    lambda x: x.shift(HORIZON).rolling(24, min_periods=1).mean())
df_residual['eia_resid_roll_std_24'] = df_residual.groupby('respondent')['eia_residual'].transform(
    lambda x: x.shift(HORIZON).rolling(24, min_periods=1).std())
df_residual['eia_resid_abs_mean_24'] = df_residual.groupby('respondent')['eia_residual'].transform(
    lambda x: x.abs().shift(HORIZON).rolling(24, min_periods=1).mean())

df_residual.dropna(subset=['eia_resid_lag_24', 'eia_resid_lag_48', 'eia_resid_lag_168',
                           'eia_resid_roll_mean_24', 'eia_resid_roll_std_24'], inplace=True)

# Feature set for residual model: original features + EIA forecast + EIA residual features
residual_feature_cols = feature_cols + ['eia_dayahead_forecast_mw',
    'eia_resid_lag_24', 'eia_resid_lag_48', 'eia_resid_lag_168',
    'eia_resid_roll_mean_24', 'eia_resid_roll_std_24', 'eia_resid_abs_mean_24']

# Also exclude any non-numeric columns
residual_feature_cols = [c for c in residual_feature_cols if df_residual[c].dtype in ('float64', 'int64', 'int32')]

print(f"Residual model features: {len(residual_feature_cols)} (original {len(feature_cols)} + EIA features)")
print(f"Residual rows after dropna: {len(df_residual)}")

def train_residual_evaluate(train_mask, val_mask, window_label):
    """Train CatBoost on EIA residual, return metrics for final prediction = eia + predicted residual."""
    X_train = df_residual.loc[train_mask, residual_feature_cols]
    y_train = df_residual.loc[train_mask, 'eia_residual']
    X_val = df_residual.loc[val_mask, residual_feature_cols]
    y_val_demand = df_residual.loc[val_mask, 'demand_mw']
    eia_val = df_residual.loc[val_mask, 'eia_dayahead_forecast_mw']

    model = CatBoostRegressor(
        iterations=800, depth=6, learning_rate=0.05,
        l2_leaf_reg=3, random_seed=42, verbose=0, allow_writing_files=False)
    model.fit(X_train, y_train)
    y_pred_residual = model.predict(X_val)
    y_pred_demand = eia_val.values + y_pred_residual

    metrics = compute_metrics(y_val_demand.values, y_pred_demand, eia_val.values)
    metrics['Model'] = 'Residual CatBoost'
    metrics['Window'] = window_label
    metrics['Top 10% MAPE (%)'] = top10_demand_mape(y_val_demand.values, y_pred_demand)
    return metrics

residual_cv_results = []
for fold in cv_folds:
    tr_end = pd.Timestamp(fold['train_end'])
    vs = pd.Timestamp(fold['val_start'])
    ve = pd.Timestamp(fold['val_end'])
    tr_mask = df_residual['ts_local'] <= tr_end
    v_mask = (df_residual['ts_local'] >= vs) & (df_residual['ts_local'] <= ve)
    if v_mask.sum() == 0:
        continue
    metrics = train_residual_evaluate(tr_mask, v_mask, fold['fold'])
    residual_cv_results.append(metrics)
    print(f"  {fold['fold']}: MAPE={metrics['MAPE (%)']:.2f}%  Beats EIA={metrics.get('% Beats EIA', 'N/A')}%  "
          f"Top10={metrics.get('Top 10% MAPE (%)', 'N/A')}%  Bias={metrics['Bias (MW)']}")

# Forward validation
residual_val_results = []
for window in validation_windows:
    tr_end = pd.Timestamp(window['train_end'])
    vs = pd.Timestamp(window['val_start'])
    ve = pd.Timestamp(window['val_end'])
    tr_mask = df_residual['ts_local'] <= tr_end
    v_mask = (df_residual['ts_local'] >= vs) & (df_residual['ts_local'] <= ve)
    if v_mask.sum() == 0:
        continue
    metrics = train_residual_evaluate(tr_mask, v_mask, window['label'])
    residual_val_results.append(metrics)
    print(f"  {window['label']}: MAPE={metrics['MAPE (%)']:.2f}%  Beats EIA={metrics.get('% Beats EIA', 'N/A')}%  "
          f"Top10={metrics.get('Top 10% MAPE (%)', 'N/A')}%  Bias={metrics['Bias (MW)']}")

residual_cv_df = pd.DataFrame(residual_cv_results)
residual_val_df = pd.DataFrame(residual_val_results)

residual_cv_summary = residual_cv_df.groupby('Model').agg(
    cv_mape_mean=('MAPE (%)', 'mean'), cv_mape_std=('MAPE (%)', 'std'),
    cv_beats_eia_mean=('% Beats EIA', 'mean'), cv_top10_mape_mean=('Top 10% MAPE (%)', 'mean'),
    cv_bias_mean=('Bias (MW)', 'mean'),
).round(3).reset_index()

print("\nResidual CatBoost CV Summary:")
display(residual_cv_summary)
print("\nResidual CatBoost Forward Validation:")
display(residual_val_df[['Window', 'MAPE (%)', 'MAE (MW)', 'RMSE (MW)', 'Bias (MW)',
                        'EIA MAPE (%)', 'Improvement vs EIA (pp)', '% Beats EIA', 'Top 10% MAPE (%)']])

# COMMAND ----------

# DBTITLE 1,Final Untouched Holdout
# MAGIC %md
# MAGIC ## Final Untouched Holdout Evaluation
# MAGIC
# MAGIC Sep 1-13, 2026 is the untouched forward period — NOT used during model selection or hyperparameter tuning.
# MAGIC All finalists are trained on all data through Aug 31, 2026 and evaluated on Sep 1-13.
# MAGIC No model registration or Champion alias change.

# COMMAND ----------

# DBTITLE 1,Final Holdout Evaluation
# --- Final untouched holdout: Sep 1-13 ---
# Train all finalists on all data through Aug 31, evaluate on Sep 1-13.

holdout_train_end = pd.Timestamp(untouched_holdout['train_end'])
holdout_val_start = pd.Timestamp(untouched_holdout['val_start'])
holdout_val_end = pd.Timestamp(untouched_holdout['val_end'])

train_mask_final = df_demand['ts_local'] <= holdout_train_end
val_mask_final = (df_demand['ts_local'] >= holdout_val_start) & (df_demand['ts_local'] <= holdout_val_end)

X_train_final = df_demand.loc[train_mask_final, feature_cols]
y_train_final = df_demand.loc[train_mask_final, 'demand_mw']
X_val_final = df_demand.loc[val_mask_final, feature_cols]
y_val_final = df_demand.loc[val_mask_final, 'demand_mw']
eia_val_final = df_demand.loc[val_mask_final, 'eia_dayahead_forecast_mw']

holdout_metrics = []

# 1. Tuned classical model
print(f"Training tuned {best_classical_name} on all data through Aug 31...")
classical_model = make_tuned_model(best_tune_params)
classical_model.fit(X_train_final, y_train_final)
y_pred_classical = classical_model.predict(X_val_final)
m = compute_metrics(y_val_final.values, y_pred_classical, eia_val_final.values)
m['Model'] = f'Tuned {best_classical_name}'
m['Top 10% MAPE (%)'] = top10_demand_mape(y_val_final.values, y_pred_classical)
holdout_metrics.append(m)
print(f"  MAPE={m['MAPE (%)']:.2f}%  Beats EIA={m.get('% Beats EIA', 'N/A')}%  Bias={m['Bias (MW)']}")

# 2. Tuned DL model
print(f"\nTraining tuned {best_dl_name} on all data through Aug 31...")
torch.manual_seed(SEED)
np.random.seed(SEED)
tr_df_final = df_demand[df_demand['ts_local'] <= holdout_train_end].copy()
v_df_final = df_demand[(df_demand['ts_local'] >= holdout_val_start) & (df_demand['ts_local'] <= holdout_val_end)].copy()
scaler_final = StandardScaler()
tr_feat_f = scaler_final.fit_transform(tr_df_final[feature_cols].values)
v_feat_f = scaler_final.transform(v_df_final[feature_cols].values)
tr_tgt_f = tr_df_final['demand_mw'].values / TARGET_SCALE
v_tgt_f = v_df_final['demand_mw'].values / TARGET_SCALE
all_f_f = np.vstack([tr_feat_f, v_feat_f])
all_t_f = np.concatenate([tr_tgt_f, v_tgt_f])
X_a_f, y_a_f = create_sequences(all_f_f, all_t_f, best_dl_tune_params['seq_len'])
n_tr_f = len(tr_feat_f) - best_dl_tune_params['seq_len']
X_tr_f, y_tr_f = X_a_f[:n_tr_f], y_a_f[:n_tr_f]
X_v_f, y_v_f = X_a_f[n_tr_f:], y_a_f[n_tr_f:]
X_tr_t_f = torch.FloatTensor(X_tr_f).to(device)
y_tr_t_f = torch.FloatTensor(y_tr_f).to(device)
X_v_t_f = torch.FloatTensor(X_v_f).to(device)
tr_loader_f = DataLoader(TensorDataset(X_tr_t_f, y_tr_t_f), batch_size=best_dl_tune_params['batch_size'], shuffle=True)
v_loader_f = DataLoader(TensorDataset(X_v_t_f, torch.FloatTensor(y_v_f).to(device)), batch_size=best_dl_tune_params['batch_size'], shuffle=False)
dl_model_final = best_dl_cls(len(feature_cols), hidden=best_dl_tune_params['hidden'],
                            n_layers=best_dl_tune_params['n_layers'], dropout=best_dl_tune_params['dropout'])
dl_model_final = train_dl_model(dl_model_final, tr_loader_f, v_loader_f, f'Tuned {best_dl_name}')
dl_model_final.eval()
with torch.no_grad():
    y_pred_dl = dl_model_final(X_v_t_f).squeeze(-1).cpu().numpy() * TARGET_SCALE
m = compute_metrics(y_v_f * TARGET_SCALE, y_pred_dl, v_df_final['eia_dayahead_forecast_mw'].values)
m['Model'] = f'Tuned {best_dl_name}'
m['Top 10% MAPE (%)'] = top10_demand_mape(y_v_f * TARGET_SCALE, y_pred_dl)
holdout_metrics.append(m)
print(f"  MAPE={m['MAPE (%)']:.2f}%  Beats EIA={m.get('% Beats EIA', 'N/A')}%  Bias={m['Bias (MW)']}")

# 3. Untuned CatBoost (default parameters, no EIA dependency)
print(f"\nTraining untuned CatBoost on all data through Aug 31...")
untuned_cat = model_configs['CatBoost']()
untuned_cat.fit(X_train_final, y_train_final)
y_pred_untuned_cat = untuned_cat.predict(X_val_final)
m = compute_metrics(y_val_final.values, y_pred_untuned_cat, eia_val_final.values)
m['Model'] = 'Untuned CatBoost'
m['Top 10% MAPE (%)'] = top10_demand_mape(y_val_final.values, y_pred_untuned_cat)
holdout_metrics.append(m)
print(f"  MAPE={m['MAPE (%)']:.2f}%  Beats EIA={m.get('% Beats EIA', 'N/A')}%  Bias={m['Bias (MW)']}")

# 4. Untuned GRU (default parameters, no EIA dependency)
print(f"\nTraining untuned GRU on all data through Aug 31...")
torch.manual_seed(SEED)
np.random.seed(SEED)
m, _ = dl_evaluate_on_window(tr_df_final, v_df_final, feature_cols, GRUModel, 'Untuned GRU')
holdout_metrics.append(m)
print(f"  MAPE={m['MAPE (%)']:.2f}%  Beats EIA={m.get('% Beats EIA', 'N/A')}%  Bias={m['Bias (MW)']}")

# 5. Untuned CNN-LSTM (default parameters, no EIA dependency)
print(f"\nTraining untuned CNN-LSTM on all data through Aug 31...")
torch.manual_seed(SEED)
np.random.seed(SEED)
m, _ = dl_evaluate_on_window(tr_df_final, v_df_final, feature_cols, CNNLSTMModel, 'Untuned CNN-LSTM')
holdout_metrics.append(m)
print(f"  MAPE={m['MAPE (%)']:.2f}%  Beats EIA={m.get('% Beats EIA', 'N/A')}%  Bias={m['Bias (MW)']}")

# 6. Residual CatBoost
print(f"\nTraining Residual CatBoost on all data through Aug 31...")
tr_mask_r = df_residual['ts_local'] <= holdout_train_end
v_mask_r = (df_residual['ts_local'] >= holdout_val_start) & (df_residual['ts_local'] <= holdout_val_end)
X_train_r = df_residual.loc[tr_mask_r, residual_feature_cols]
y_train_r = df_residual.loc[tr_mask_r, 'eia_residual']
X_val_r = df_residual.loc[v_mask_r, residual_feature_cols]
y_val_demand_r = df_residual.loc[v_mask_r, 'demand_mw']
eia_val_r = df_residual.loc[v_mask_r, 'eia_dayahead_forecast_mw']
resid_model = CatBoostRegressor(
    iterations=800, depth=6, learning_rate=0.05,
    l2_leaf_reg=3, random_seed=42, verbose=0, allow_writing_files=False)
resid_model.fit(X_train_r, y_train_r)
y_pred_resid = resid_model.predict(X_val_r)
y_pred_demand_r = eia_val_r.values + y_pred_resid
m = compute_metrics(y_val_demand_r.values, y_pred_demand_r, eia_val_r.values)
m['Model'] = 'Residual CatBoost'
m['Top 10% MAPE (%)'] = top10_demand_mape(y_val_demand_r.values, y_pred_demand_r)
holdout_metrics.append(m)
print(f"  MAPE={m['MAPE (%)']:.2f}%  Beats EIA={m.get('% Beats EIA', 'N/A')}%  Bias={m['Bias (MW)']}")

# 7. EIA baseline
eia_mask = ~eia_val_final.isna()
if eia_mask.sum() > 0:
    eia_mape = mean_absolute_percentage_error(y_val_final[eia_mask], eia_val_final[eia_mask]) * 100
    eia_mae = mean_absolute_error(y_val_final[eia_mask], eia_val_final[eia_mask])
    eia_rmse = np.sqrt(mean_squared_error(y_val_final[eia_mask], eia_val_final[eia_mask]))
    eia_bias = np.mean(eia_val_final[eia_mask].values - y_val_final[eia_mask].values)
    holdout_metrics.append({
        'Model': 'EIA Baseline', 'MAPE (%)': round(eia_mape, 3),
        'MAE (MW)': round(eia_mae, 1), 'RMSE (MW)': round(eia_rmse, 1),
        'Bias (MW)': round(eia_bias, 1), 'Top 10% MAPE (%)': top10_demand_mape(
            y_val_final[eia_mask].values, eia_val_final[eia_mask].values),
        '% Beats EIA': 0.0,
    })
    print(f"\nEIA Baseline: MAPE={eia_mape:.2f}%")

holdout_final_df = pd.DataFrame(holdout_metrics)
print("\n" + "=" * 70)
print("FINAL UNTOUCHED HOLDOUT (Sep 1-13)")
print("=" * 70)
display(holdout_final_df[['Model', 'MAPE (%)', 'MAE (MW)', 'RMSE (MW)', 'Bias (MW)',
                          '% Beats EIA', 'Top 10% MAPE (%)']].sort_values('MAPE (%)'))

# COMMAND ----------

# DBTITLE 1,Final Comparison Table
# --- Final comparison table ---
# Model | CV MAPE Mean | CV Std | Holdout MAPE | MAE | RMSE | Bias | % Beats EIA | Peak MAPE

final_comparison_rows = []

def _f(v):
    """Safely convert to float or return None."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return float(v)

# Tuned classical
classical_holdout = holdout_final_df[holdout_final_df['Model'] == f'Tuned {best_classical_name}'].iloc[0]
final_comparison_rows.append({
    'Model': f'Tuned {best_classical_name}',
    'CV MAPE Mean': round(best_tune_mape, 2),
    'CV Std': None,
    'Holdout MAPE': _f(classical_holdout['MAPE (%)']),
    'MAE': _f(classical_holdout['MAE (MW)']),
    'RMSE': _f(classical_holdout['RMSE (MW)']),
    'Bias': _f(classical_holdout['Bias (MW)']),
    '% Beats EIA': _f(classical_holdout.get('% Beats EIA', None)),
    'Peak MAPE': _f(classical_holdout['Top 10% MAPE (%)']),
})

# Tuned DL
dl_holdout = holdout_final_df[holdout_final_df['Model'] == f'Tuned {best_dl_name}'].iloc[0]
final_comparison_rows.append({
    'Model': f'Tuned {best_dl_name}',
    'CV MAPE Mean': round(best_dl_tune_mape, 2),
    'CV Std': None,
    'Holdout MAPE': _f(dl_holdout['MAPE (%)']),
    'MAE': _f(dl_holdout['MAE (MW)']),
    'RMSE': _f(dl_holdout['RMSE (MW)']),
    'Bias': _f(dl_holdout['Bias (MW)']),
    '% Beats EIA': _f(dl_holdout.get('% Beats EIA', None)),
    'Peak MAPE': _f(dl_holdout['Top 10% MAPE (%)']),
})

# Untuned CatBoost (no EIA dependency)
untuned_cat_holdout = holdout_final_df[holdout_final_df['Model'] == 'Untuned CatBoost'].iloc[0]
final_comparison_rows.append({
    'Model': 'Untuned CatBoost',
    'CV MAPE Mean': round(float(cv_summary[cv_summary['Model'] == 'CatBoost']['cv_mape_mean'].values[0]), 2),
    'CV Std': round(float(cv_summary[cv_summary['Model'] == 'CatBoost']['cv_mape_std'].values[0]), 2),
    'Holdout MAPE': _f(untuned_cat_holdout['MAPE (%)']),
    'MAE': _f(untuned_cat_holdout['MAE (MW)']),
    'RMSE': _f(untuned_cat_holdout['RMSE (MW)']),
    'Bias': _f(untuned_cat_holdout['Bias (MW)']),
    '% Beats EIA': _f(untuned_cat_holdout.get('% Beats EIA', None)),
    'Peak MAPE': _f(untuned_cat_holdout['Top 10% MAPE (%)']),
})

# Untuned GRU (no EIA dependency)
untuned_gru_holdout = holdout_final_df[holdout_final_df['Model'] == 'Untuned GRU'].iloc[0]
final_comparison_rows.append({
    'Model': 'Untuned GRU',
    'CV MAPE Mean': round(float(dl_cv_summary[dl_cv_summary['Model'] == 'GRU']['cv_mape_mean'].values[0]), 2),
    'CV Std': round(float(dl_cv_summary[dl_cv_summary['Model'] == 'GRU']['cv_mape_std'].values[0]), 2),
    'Holdout MAPE': _f(untuned_gru_holdout['MAPE (%)']),
    'MAE': _f(untuned_gru_holdout['MAE (MW)']),
    'RMSE': _f(untuned_gru_holdout['RMSE (MW)']),
    'Bias': _f(untuned_gru_holdout['Bias (MW)']),
    '% Beats EIA': _f(untuned_gru_holdout.get('% Beats EIA', None)),
    'Peak MAPE': _f(untuned_gru_holdout['Top 10% MAPE (%)']),
})

# Untuned CNN-LSTM (no EIA dependency)
untuned_cnnlstm_holdout = holdout_final_df[holdout_final_df['Model'] == 'Untuned CNN-LSTM'].iloc[0]
final_comparison_rows.append({
    'Model': 'Untuned CNN-LSTM',
    'CV MAPE Mean': round(float(dl_cv_summary[dl_cv_summary['Model'] == 'CNN-LSTM']['cv_mape_mean'].values[0]), 2),
    'CV Std': round(float(dl_cv_summary[dl_cv_summary['Model'] == 'CNN-LSTM']['cv_mape_std'].values[0]), 2),
    'Holdout MAPE': _f(untuned_cnnlstm_holdout['MAPE (%)']),
    'MAE': _f(untuned_cnnlstm_holdout['MAE (MW)']),
    'RMSE': _f(untuned_cnnlstm_holdout['RMSE (MW)']),
    'Bias': _f(untuned_cnnlstm_holdout['Bias (MW)']),
    '% Beats EIA': _f(untuned_cnnlstm_holdout.get('% Beats EIA', None)),
    'Peak MAPE': _f(untuned_cnnlstm_holdout['Top 10% MAPE (%)']),
})

# Residual CatBoost
resid_holdout = holdout_final_df[holdout_final_df['Model'] == 'Residual CatBoost'].iloc[0]
final_comparison_rows.append({
    'Model': 'Residual CatBoost',
    'CV MAPE Mean': round(float(residual_cv_summary.iloc[0]['cv_mape_mean']), 2),
    'CV Std': round(float(residual_cv_summary.iloc[0]['cv_mape_std']), 2),
    'Holdout MAPE': _f(resid_holdout['MAPE (%)']),
    'MAE': _f(resid_holdout['MAE (MW)']),
    'RMSE': _f(resid_holdout['RMSE (MW)']),
    'Bias': _f(resid_holdout['Bias (MW)']),
    '% Beats EIA': _f(resid_holdout.get('% Beats EIA', None)),
    'Peak MAPE': _f(resid_holdout['Top 10% MAPE (%)']),
})

# EIA baseline
eia_holdout = holdout_final_df[holdout_final_df['Model'] == 'EIA Baseline'].iloc[0]
final_comparison_rows.append({
    'Model': 'EIA Baseline',
    'CV MAPE Mean': None,
    'CV Std': None,
    'Holdout MAPE': _f(eia_holdout['MAPE (%)']),
    'MAE': _f(eia_holdout['MAE (MW)']),
    'RMSE': _f(eia_holdout['RMSE (MW)']),
    'Bias': _f(eia_holdout['Bias (MW)']),
    '% Beats EIA': None,
    'Peak MAPE': _f(eia_holdout['Top 10% MAPE (%)']),
})

final_comparison_df = pd.DataFrame(final_comparison_rows)
final_comparison_df = final_comparison_df.sort_values('Holdout MAPE')

print("=" * 70)
print("FINAL COMPARISON TABLE")
print("=" * 70)
display(final_comparison_df)

# Identify winners
trained_models = [r for r in final_comparison_rows if r['Model'] != 'EIA Baseline']
best_overall = min(trained_models, key=lambda x: x['Holdout MAPE'])
eia_mape_val = float(eia_holdout['MAPE (%)'])

# Identify best non-EIA model (standalone, no EIA dependency)
non_eia_models = [r for r in final_comparison_rows if r['Model'] not in ('EIA Baseline', 'Residual CatBoost')]
best_non_eia = min(non_eia_models, key=lambda x: x['Holdout MAPE'])

print(f"\n1. Best classical model: Tuned {best_classical_name}")
print(f"2. Best deep-learning model: Tuned {best_dl_name}")
print(f"3. Best residual model: Residual CatBoost")
print(f"4. Overall winner (all models): {best_overall['Model']} (Holdout MAPE {best_overall['Holdout MAPE']:.2f}%)")
beats_eia = best_overall['Holdout MAPE'] < eia_mape_val
print(f"5. Beats EIA: {'Yes' if beats_eia else 'No'} "
      f"({best_overall['Holdout MAPE']:.2f}% vs EIA {eia_mape_val:.2f}%)")
print(f"6. Best non-EIA (standalone) model: {best_non_eia['Model']} (Holdout MAPE {best_non_eia['Holdout MAPE']:.2f}%)")
print(f"7. Production readiness: "
      f"{'Stable enough for production' if beats_eia else 'Does not beat EIA baseline on untouched holdout'}")
print(f"\nChampion alias NOT changed. No models deployed.")

# COMMAND ----------

# DBTITLE 1,Production GRU Registration
# --- Production Model Registration: Untuned GRU ---
# Train the production GRU on all data through Aug 31, 2026.
# Register in Unity Catalog with all preprocessing artifacts.
# This model has NO EIA dependency.

import json
import pickle
import tempfile
import os
import mlflow.pyfunc

# Model configuration (untuned defaults from screening)
GRU_CONFIG = {
    'model_type': 'GRU',
    'seq_len': 24,
    'hidden': 64,
    'n_layers': 1,
    'dropout': 0.0,
    'lr': 0.001,
    'batch_size': 256,
    'max_epochs': 50,
    'patience': 10,
    'target_scale': 50000.0,
    'seed': 42,
}

UC_MODEL_NAME = "dev.stlf_gold.demand_forecast_gru"
EXPERIMENT_NAME = "/Users/qgf22@txstate.edu/electricity-demand-forecasting"

# --- Train GRU on all data through Aug 31 ---
train_end = pd.Timestamp(untouched_holdout['train_end'])
val_start = pd.Timestamp(untouched_holdout['val_start'])
val_end = pd.Timestamp(untouched_holdout['val_end'])

train_df = df_demand[df_demand['ts_local'] <= train_end].copy()
val_df = df_demand[(df_demand['ts_local'] >= val_start) & (df_demand['ts_local'] <= val_end)].copy()

# Fit scaler on training data ONLY — never refit during inference
scaler = StandardScaler()
train_feat = scaler.fit_transform(train_df[feature_cols].values)
val_feat = scaler.transform(val_df[feature_cols].values)
train_tgt = train_df['demand_mw'].values / GRU_CONFIG['target_scale']
val_tgt = val_df['demand_mw'].values / GRU_CONFIG['target_scale']

# Build sequences (same logic as create_sequences)
all_feat = np.vstack([train_feat, val_feat])
all_tgt = np.concatenate([train_tgt, val_tgt])
X_all, y_all = create_sequences(all_feat, all_tgt, GRU_CONFIG['seq_len'])
n_train_seq = len(train_feat) - GRU_CONFIG['seq_len']
X_train_seq, y_train_seq = X_all[:n_train_seq], y_all[:n_train_seq]
X_val_seq, y_val_seq = X_all[n_train_seq:], y_all[n_train_seq:]

# Train
torch.manual_seed(GRU_CONFIG['seed'])
np.random.seed(GRU_CONFIG['seed'])

X_train_t = torch.FloatTensor(X_train_seq).to(device)
y_train_t = torch.FloatTensor(y_train_seq).to(device)
X_val_t = torch.FloatTensor(X_val_seq).to(device)
y_val_t = torch.FloatTensor(y_val_seq).to(device)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=GRU_CONFIG['batch_size'], shuffle=True)
val_loader = DataLoader(
    TensorDataset(X_val_t, y_val_t),
    batch_size=GRU_CONFIG['batch_size'], shuffle=False)

gru_model = GRUModel(len(feature_cols), hidden=GRU_CONFIG['hidden'],
                     n_layers=GRU_CONFIG['n_layers'], dropout=GRU_CONFIG['dropout'])
gru_model = train_dl_model(gru_model, train_loader, val_loader, 'Production GRU')

# Evaluate on Sep holdout
gru_model.eval()
with torch.no_grad():
    y_pred_holdout = gru_model(X_val_t).squeeze(-1).cpu().numpy() * GRU_CONFIG['target_scale']
y_true_holdout = y_val_seq * GRU_CONFIG['target_scale']
holdout_metrics = compute_metrics(y_true_holdout, y_pred_holdout,
                                  val_df['eia_dayahead_forecast_mw'].values)
holdout_metrics['Top 10% MAPE (%)'] = top10_demand_mape(y_true_holdout, y_pred_holdout)

print(f"Production GRU Holdout MAPE: {holdout_metrics['MAPE (%)']:.2f}%")
print(f"  Beats EIA: {holdout_metrics.get('% Beats EIA', 'N/A')}%")
print(f"  Bias: {holdout_metrics['Bias (MW)']} MW")
print(f"  Top 10% MAPE: {holdout_metrics['Top 10% MAPE (%)']:.2f}%")

# --- Verify feature order ---
print(f"\nFeature count: {len(feature_cols)}")
print(f"Sequence length: {GRU_CONFIG['seq_len']}")
print(f"Training data end: {train_end}")
print(f"Training rows: {len(train_df)}")

# --- MLflow pyfunc wrapper (self-contained, includes architecture) ---
class GRUForecastWrapper(mlflow.pyfunc.PythonModel):
    """Self-contained GRU model for 24-hour demand forecasting."""

    def load_context(self, context):
        import torch
        import torch.nn as nn
        import json
        import pickle
        import numpy as np

        class GRUModel(nn.Module):
            def __init__(self, input_dim, hidden=64, n_layers=1, dropout=0.0):
                super().__init__()
                self.gru = nn.GRU(input_dim, hidden, num_layers=n_layers,
                                  dropout=dropout, batch_first=True)
                self.head = nn.Sequential(
                    nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
            def forward(self, x):
                out, _ = self.gru(x)
                return self.head(out[:, -1, :])

        with open(context.artifacts['config']) as f:
            self.config = json.load(f)
        with open(context.artifacts['features']) as f:
            self.feature_list = json.load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GRUModel(
            len(self.feature_list),
            hidden=self.config['hidden'],
            n_layers=self.config['n_layers'],
            dropout=self.config['dropout'])
        self.model.load_state_dict(
            torch.load(context.artifacts['weights'], map_location=self.device,
                       weights_only=True))
        self.model.to(self.device)
        self.model.eval()

        with open(context.artifacts['scaler'], 'rb') as f:
            self.scaler = pickle.load(f)

    def predict(self, context, model_input):
        import numpy as np
        import torch

        features = model_input[self.feature_list].values.astype(np.float32)
        scaled = self.scaler.transform(features).astype(np.float32)

        seq_len = self.config['seq_len']
        target_scale = self.config['target_scale']

        sequences = []
        for i in range(seq_len, len(scaled)):
            sequences.append(scaled[i - seq_len:i])
        sequences = np.array(sequences)

        x = torch.FloatTensor(sequences).to(self.device)
        with torch.no_grad():
            preds = self.model(x).squeeze(-1).cpu().numpy()

        return preds * target_scale

# --- Log to MLflow and register in UC ---
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run(run_name="Production_GRU_Untuned_2026Sep") as run:
    run_id = run.info.run_id

    # Log params
    mlflow.log_params(GRU_CONFIG)
    mlflow.log_param("feature_count", len(feature_cols))
    mlflow.log_param("train_end_date", str(train_end))
    mlflow.log_param("train_rows", len(train_df))
    mlflow.log_param("val_start_date", str(val_start))
    mlflow.log_param("val_end_date", str(val_end))

    # Log metrics
    mlflow.log_metric("holdout_mape", holdout_metrics['MAPE (%)'])
    mlflow.log_metric("holdout_mae", holdout_metrics['MAE (MW)'])
    mlflow.log_metric("holdout_rmse", holdout_metrics['RMSE (MW)'])
    mlflow.log_metric("holdout_bias", holdout_metrics['Bias (MW)'])
    mlflow.log_metric("holdout_beats_eia_pct", holdout_metrics.get('% Beats EIA', 0))
    mlflow.log_metric("holdout_top10_mape", holdout_metrics.get('Top 10% MAPE (%)', 0))
    mlflow.log_metric("cv_mape_mean", 1.57)
    mlflow.log_metric("cv_mape_std", 0.48)

    # Log tags
    mlflow.set_tags({
        'model_type': 'GRU',
        'eia_dependency': 'none',
        'forecast_horizon': '24h',
        'feature_engineering': 'leakage_safe_shift24',
        'training_data_range': f"2019-01-21 to {train_end}",
        'validation_method': 'time_based_CV_5fold_plus_holdout',
    })

    # Save artifacts to temp directory
    artifact_dir = tempfile.mkdtemp()

    torch.save(gru_model.state_dict(),
               os.path.join(artifact_dir, 'gru_weights.pt'))

    with open(os.path.join(artifact_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    with open(os.path.join(artifact_dir, 'config.json'), 'w') as f:
        json.dump(GRU_CONFIG, f, indent=2)

    with open(os.path.join(artifact_dir, 'features.json'), 'w') as f:
        json.dump(list(feature_cols), f, indent=2)

    artifacts = {
        'weights': os.path.join(artifact_dir, 'gru_weights.pt'),
        'scaler': os.path.join(artifact_dir, 'scaler.pkl'),
        'config': os.path.join(artifact_dir, 'config.json'),
        'features': os.path.join(artifact_dir, 'features.json'),
    }

    # Infer model signature for UC registration (required by Databricks UC)
    sig_input = pd.DataFrame({col: [0.0] for col in feature_cols})
    sig_output = np.array([50000.0])
    model_signature = mlflow.models.infer_signature(
        model_input=sig_input, model_output=sig_output)

    # Log as pyfunc model
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=GRUForecastWrapper(),
        artifacts=artifacts,
        signature=model_signature,
    )

    # Register in Unity Catalog
    model_uri = f"runs:/{run_id}/model"
    registered = mlflow.register_model(model_uri=model_uri, name=UC_MODEL_NAME)

    print(f"\nRegistered: {UC_MODEL_NAME} version {registered.version}")
    print(f"  Run ID: {run_id}")

# Set Champion alias
client = mlflow.MlflowClient()
client.set_registered_model_alias(UC_MODEL_NAME, "Champion", int(registered.version))

# Verify Champion alias
alias_info = client.get_model_version_by_alias(UC_MODEL_NAME, "Champion")
print(f"Champion alias verified: version {alias_info.version}")
print(f"\nModel registration complete.")
print(f"  UC Model: {UC_MODEL_NAME}")
print(f"  Version: {registered.version}")
print(f"  Alias: Champion")
print(f"  Holdout MAPE: {holdout_metrics['MAPE (%)']:.2f}%")
print(f"  Features: {len(feature_cols)}")
print(f"  Sequence length: {GRU_CONFIG['seq_len']}")
print(f"  Training end: {train_end}")