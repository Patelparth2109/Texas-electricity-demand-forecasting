# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Daily Data Refresh - Operational Maintenance
# MAGIC %md
# MAGIC # Daily Data Refresh - Operational Maintenance
# MAGIC
# MAGIC **Purpose:** Keep data current for operational forecasting
# MAGIC
# MAGIC **Schedule:** Run daily at 2:00 AM UTC via Databricks Job
# MAGIC
# MAGIC **What it does:**
# MAGIC 1. Pull yesterday's actual demand from EIA-930
# MAGIC 2. Pull yesterday's weather observations  
# MAGIC 3. Refresh 24-hour weather forecast for operational predictions
# MAGIC 4. Data flows automatically to feature table via silver/gold layers
# MAGIC
# MAGIC **Run time:** ~2-3 minutes
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Configuration
from datetime import datetime, timedelta, timezone
import json, time, requests
import pandas as pd
from pyspark.sql import functions as F

# Unity Catalog
CATALOG = "dev"
BRONZE_SCHEMA = "stlf_bronze"
GOLD_SCHEMA = "stlf_gold"

DEMAND_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.eia930_demand_raw"
WEATHER_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.open_meteo_weather_raw"
WEATHER_FORECAST_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.open_meteo_forecast_weather"

BA = "ERCO"
# Read EIA API key from Databricks secret (preferred) or notebook parameter (fallback)
# To use secrets: create scope 'forecasting' with key 'eia_api_key' via Databricks UI > Secrets
# To use parameter: add 'eia_api_key' as a notebook parameter in the job task settings
try:
    EIA_API_KEY = dbutils.secrets.get(scope="forecasting", key="eia_api_key")
except Exception:
    try:
        EIA_API_KEY = dbutils.widgets.get("eia_api_key")
    except Exception:
        raise RuntimeError(
            "EIA API key not found. Either:\n"
            "1. Create Databricks secret scope 'forecasting' with key 'eia_api_key', or\n"
            "2. Add 'eia_api_key' as a notebook parameter in the job task settings"
        )

CITIES = [
    {"city": "Houston", "lat": 29.7604, "lon": -95.3698, "load_weight": 0.38},
    {"city": "Dallas", "lat": 32.7767, "lon": -96.7970, "load_weight": 0.28},
    {"city": "San Antonio", "lat": 29.4241, "lon": -98.4936, "load_weight": 0.19},
    {"city": "Austin", "lat": 30.2672, "lon": -97.7431, "load_weight": 0.15}
]

# Pull last 7 days to ensure no gaps
start_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT00")
end_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")

print(f"Daily refresh window: {start_date} to {end_date}")

# COMMAND ----------

# DBTITLE 1,Helper Functions
def _get_with_retry(url, *, headers=None, params=None, timeout=90, tries=4):
    last = None
    for attempt in range(tries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            last = e
            if attempt < tries - 1:
                time.sleep(2 ** attempt)
    raise last

# COMMAND ----------

# DBTITLE 1,Refresh EIA Demand (Last 7 Days)
print("\n1. Refreshing EIA demand data...")

EIA_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
params = {
    "api_key": EIA_API_KEY,
    "frequency": "hourly",
    "data[0]": "value",
    "facets[respondent][]": BA,
    "facets[type][]": ["D", "DF"],
    "start": start_date,
    "end": end_date,
    "sort[0][column]": "period",
    "sort[0][direction]": "asc",
    "offset": 0,
    "length": 5000,
}

resp = _get_with_retry(EIA_URL, params=params, timeout=120)
eia_rows = resp.json().get("response", {}).get("data", [])

if eia_rows:
    eia_pdf = pd.DataFrame(eia_rows)
    # Rename hyphenated columns to match Delta table schema (underscores)
    col_renames = {
        "type-name": "type_name",
        "respondent-name": "respondent_name",
        "value-units": "value_units",
    }
    eia_pdf = eia_pdf.rename(columns={k: v for k, v in col_renames.items() if k in eia_pdf.columns})
    
    eia_sdf = (
        spark.createDataFrame(eia_pdf)
        .withColumn("value", F.col("value").cast("double"))
        .withColumn("_source", F.lit("EIA-930 API"))
        .withColumn("_ingested_at", F.current_timestamp())
    )
    
    from delta.tables import DeltaTable
    target = DeltaTable.forName(spark, DEMAND_TABLE)
    target.alias("t").merge(
        eia_sdf.alias("s"),
        """t.period = s.period AND t.respondent = s.respondent AND 
           t.type = s.type AND t.type_name = s.type_name"""
    ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
    
    print(f"  Merged {len(eia_rows):,} demand rows")
else:
    print("   No new demand data found")

# COMMAND ----------

# DBTITLE 1,Refresh Weather Observations
print("\n2. Refreshing weather observations...")

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "precipitation", "rain", "snowfall",
    "cloud_cover", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "shortwave_radiation", "direct_normal_irradiance"
]

weather_start = start_date[:10]
weather_end = end_date[:10]

frames = []
for c in CITIES:
    params = {
        "latitude": c["lat"], "longitude": c["lon"],
        "start_date": weather_start, "end_date": weather_end,
        "hourly": ",".join(WEATHER_VARS),
        "timezone": "UTC", "wind_speed_unit": "ms"
    }
    resp = _get_with_retry(ARCHIVE_URL, params=params, timeout=120)
    wdf = pd.DataFrame(resp.json()["hourly"])
    wdf["city"] = c["city"]
    wdf["lat"] = c["lat"]
    wdf["lon"] = c["lon"]
    wdf["load_weight"] = c["load_weight"]
    frames.append(wdf)
    time.sleep(0.4)

weather_sdf = (
    spark.createDataFrame(pd.concat(frames, ignore_index=True))
    .withColumnRenamed("time", "period")
    .withColumn("_source", F.lit("Open-Meteo archive (ERA5)"))
    .withColumn("_ingested_at", F.current_timestamp())
)

from delta.tables import DeltaTable
target = DeltaTable.forName(spark, WEATHER_TABLE)
target.alias("t").merge(
    weather_sdf.alias("s"),
    "t.period = s.period AND t.city = s.city"
).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()

print(f"  Merged {weather_sdf.count():,} weather rows")

# COMMAND ----------

# DBTITLE 1,Refresh Weather Forecast
print("\n3. Refreshing weather forecast...")

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

forecast_frames = []
for c in CITIES:
    params = {
        "latitude": c["lat"], "longitude": c["lon"],
        "hourly": ",".join(["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "cloud_cover", "wind_speed_10m", "shortwave_radiation", "direct_normal_irradiance"]),
        "timezone": "UTC", "wind_speed_unit": "ms",
        "forecast_days": 2
    }
    resp = _get_with_retry(FORECAST_URL, params=params, timeout=120)
    fdf = pd.DataFrame(resp.json()["hourly"])
    fdf["time"] = pd.to_datetime(fdf["time"]).dt.strftime("%Y-%m-%dT%H:%M")
    fdf["city"] = c["city"]
    fdf["lat"] = c["lat"]
    fdf["lon"] = c["lon"]
    fdf["load_weight"] = c["load_weight"]
    fdf["_source"] = "Open-Meteo forecast"
    fdf["_ingested_at"] = pd.Timestamp.utcnow()
    forecast_frames.append(fdf)
    time.sleep(0.4)

# MERGE raw forecast into bronze table (same schema as historical weather)
forecast_sdf = spark.createDataFrame(pd.concat(forecast_frames, ignore_index=True))
forecast_sdf = forecast_sdf.withColumnRenamed("time", "period")

from delta.tables import DeltaTable
target = DeltaTable.forName(spark, WEATHER_FORECAST_TABLE)
target.alias("t").merge(
    forecast_sdf.alias("s"),
    "t.period = s.period AND t.city = s.city"
).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()

print(f"  Merged {forecast_sdf.count():,} forecast rows into {WEATHER_FORECAST_TABLE}")

# COMMAND ----------

# DBTITLE 1,Summary
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC Data is now current. Next steps:
# MAGIC 1. Silver/gold layers will automatically process new bronze data
# MAGIC 2. Run `daily_batch_inference` notebook to generate predictions
# MAGIC 3. Forecasts are written to `demand_predictions`
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Schedule as Databricks Job
# MAGIC
# MAGIC To automate:
# MAGIC 1. Create a new Job in Databricks
# MAGIC 2. Add this notebook as a task
# MAGIC 3. Schedule: Daily at 2:00 AM UTC
# MAGIC 4. Attach to: Serverless or small cluster
# MAGIC 5. Enable email notifications on failure