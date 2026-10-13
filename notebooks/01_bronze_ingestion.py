# Databricks notebook source
# # Bronze Layer: EIA-930 Demand and Open-Meteo Weather Ingestion
# One-time historical backfill into Unity Catalog bronze Delta tables.
# - **Demand and EIA day-ahead forecast:** EIA v2 API. Paginated (5,000-row cap), sorted for stable pages, values cast from string → double at write.
# - **Weather:** Open-Meteo historical archive (ERA5). Index-aligned arrays, no API key, with a guard against silently-dropped variables.
#
# Both pulls run on the **driver** (plain HTTP, not Spark), so a small single-node or serverless cluster is all this needs. Re-running the notebook is safe: writes overwrite, so a failed pull replaces cleanly.

# COMMAND ----------

# ## Config

# COMMAND ----------

from datetime import datetime, timedelta, timezone
 
# --- Unity Catalog target -----------------------------------------------------

CATALOG       = "dev"                 # Unity Catalog catalog (backfill runs in dev)
BRONZE_SCHEMA = "stlf_bronze"
DEMAND_TABLE  = f"{CATALOG}.{BRONZE_SCHEMA}.eia930_demand_raw"
WEATHER_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.open_meteo_weather_raw"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{BRONZE_SCHEMA}")

# COMMAND ----------

# --- Target balancing authority ----------------------------------------------

BA = "ERCO"   # ERCOT. Others: CISO, MISO, PJM, ISNE, NYIS, SWPP, ...

# COMMAND ----------

# --- EIA API key from a Databricks secret scope (never hardcode) -------------
# Register at https://www.eia.gov/opendata/, then:
#   databricks secrets create-scope forecasting
#   databricks secrets put-secret forecasting eia_api_key
EIA_API_KEY = dbutils.secrets.get(scope="forecasting", key="eia_api_key")

# COMMAND ----------

# --- History window in UTC. EIA-930 begins 2015-07-01. -----------------------

EIA_START = "2015-07-01T00"
EIA_END   = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")

# COMMAND ----------

# --- Weather: load-weighted ERCOT metros ------------------------------------
# load_weight approximates each metro's share of ERCOT load; weights sum to 1.0.
# Must match the weights used in jobs/daily_data_refresh.py.
CITIES = [
    {
        "city": "Houston", 
        "lat": 29.7604, "lon": -95.3698, 
        "load_weight": 0.38  # Coast zone industrial + high humidity load
    },
    {
        "city": "Dallas", 
        "lat": 32.7767, "lon": -96.7970, 
        "load_weight": 0.36  # North Central data center + residential load
    },
    {
        "city": "San Antonio", 
        "lat": 29.4241, "lon": -98.4936, 
        "load_weight": 0.13  # South Central metro hub
    },
    {
        "city": "Austin", 
        "lat": 30.2672, "lon": -97.7431, 
        "load_weight": 0.13  # South Central tech corridors
    },
]

WEATHER_VARS = [
    "temperature_2m",
    "apparent_temperature",
    "dew_point_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "cloud_cover",
    "shortwave_radiation",
    "direct_normal_irradiance" 
]

# ERA5 archive lags ~5 days; the recent gap is filled by the forecast API in the
# scoring path, so stop the historical backfill a few days short of "now".
WEATHER_START = "2015-07-01"
WEATHER_END = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%d")

# COMMAND ----------

# ## Shared imports

# COMMAND ----------

import json, time, requests
import pandas as pd
from pyspark.sql import functions as F

def _get_with_retry(url, *, headers=None, params=None, timeout=90, tries=4):
    """GET with simple exponential backoff for transient errors / rate limits."""
    last = None
    for attempt in range(tries):
        resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        if resp.status_code == 200:
            return resp
        last = resp
        time.sleep(2 ** attempt)
    last.raise_for_status()

# COMMAND ----------

# ## EIA-930 — demand (D) + day-ahead forecast (DF)

# COMMAND ----------

EIA_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
 
def eia_fetch_all(api_key, respondent, types, start, end, page=5000, pause=0.4):
    """
    Page through the 5,000-row cap.
 
    - sort by period ASC so pages don't shuffle across requests
    - `type` facet pulls D and DF together (long format; pivot in silver)
    - params ride in the X-Params header — cleaner than bracket-encoded query params
    """
    rows, offset = [], 0
    while True:
        params = {
            "frequency": "hourly",                 # hourly = UTC timestamps
            "data": ["value"],
            "facets": {"respondent": [respondent], "type": types},
            "start": start,
            "end": end,
            "sort": [{"column": "period", "direction": "asc"}],
            "offset": offset,
            "length": page,
        }
        resp = _get_with_retry(
            EIA_URL,
            headers={"X-Api-Key": api_key, "X-Params": json.dumps(params)},
        )
        payload = resp.json()["response"]
        batch = payload.get("data", [])
        rows.extend(batch)
        total = int(payload.get("total", 0))
        offset += page
        print(f"  {respondent}: {len(rows):>7,} / {total:,}")
        if offset >= total or not batch:
            break
        time.sleep(pause)
    return rows
 
 
print(f"Pulling EIA {BA} D + DF  {EIA_START} → {EIA_END}")
eia_rows = eia_fetch_all(EIA_API_KEY, BA, ["D", "DF"], EIA_START, EIA_END)
 
# --- to pandas: cast value string → numeric NOW to avoid a string-typed bronze ---
pdf = pd.DataFrame(eia_rows)
pdf["value"] = pd.to_numeric(pdf["value"], errors="coerce")
 
# --- to Spark: fix hyphenated column names, add ingestion metadata -------------
demand_sdf = (
    spark.createDataFrame(pdf)
    .withColumnRenamed("respondent-name", "respondent_name")
    .withColumnRenamed("type-name",       "type_name")
    .withColumnRenamed("value-units",     "value_units")
    .withColumn("_source",      F.lit("EIA v2 electricity/rto/region-data"))
    .withColumn("_ingested_at", F.current_timestamp())
)
# `period` stays a raw UTC string ("YYYY-MM-DDTHH") — timestamp parsing is silver's job.
 
(demand_sdf.write
    .mode("overwrite")                       # backfill is idempotent
    .option("overwriteSchema", "true")
    .saveAsTable(DEMAND_TABLE))
 
print(f"Wrote {demand_sdf.count():,} rows → {DEMAND_TABLE}")
 

# COMMAND ----------

# ## Open-Meteo — historical weather (ERA5)

# COMMAND ----------

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

def open_meteo_archive(lat, lon, start, end, variables):
    """
    One archive call per location. Response is index-aligned arrays:
    time[i] pairs with each variable[i].
    Open-Meteo silently DROPS variables it doesn't recognize, so we assert
    every requested column actually came back.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(variables),
        "timezone": "UTC", # keep bronze in UTC to match EIA
        "wind_speed_unit": "ms", # SI; convert downstream if you prefer
    }
    resp = _get_with_retry(ARCHIVE_URL, params=params, timeout=120)
    hourly = resp.json()["hourly"]
    df = pd.DataFrame(hourly) # columns: time + each variable
    missing = [v for v in variables if v not in df.columns]
    if missing:
        raise ValueError(f"Open-Meteo returned no data for: {missing}")
    return df

frames = []
for c in CITIES:
    print(f"Pulling weather: {c['city']} {WEATHER_START} → {WEATHER_END}")
    wdf = open_meteo_archive(c["lat"], c["lon"], WEATHER_START, WEATHER_END, WEATHER_VARS)
    wdf["city"] = c["city"]
    wdf["lat"] = c["lat"]
    wdf["lon"] = c["lon"]
    wdf["load_weight"] = c["load_weight"]
    frames.append(wdf)
    time.sleep(0.4)

weather_pdf = pd.concat(frames, ignore_index=True)
weather_sdf = (
    spark.createDataFrame(weather_pdf)
    .withColumnRenamed("time", "period") # note: Open-Meteo period is "YYYY-MM-DDTHH:MM"
    .withColumn("_source", F.lit("Open-Meteo archive (ERA5)"))
    .withColumn("_ingested_at", F.current_timestamp())
)
(weather_sdf.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(WEATHER_TABLE))
print(f"Wrote {weather_sdf.count():,} rows → {WEATHER_TABLE}")

# COMMAND ----------

# ## NOAA HRRR — Historical Day-Ahead Weather Forecasts
#
# **Purpose**: Pull what the weather forecast *predicted* 24 hours ahead (not what actually happened).
#
# **Why this matters for fair comparison**:
# - Current setup uses ERA5 (perfect hindsight weather observations)
# - Operational forecasting uses weather *forecasts* (which have errors)
# - HRRR gives us the 24h-ahead forecast that was issued at each historical time
#
# **HRRR Details**:
# - High-Resolution Rapid Refresh model from NOAA
# - 3km resolution, hourly forecasts up to 48 hours ahead
# - Archives available on AWS S3 back to 2014
# - Free public data, but GRIB2 format requires `cfgrib` + `xarray`
#
# **Dependencies:** `xarray`, `cfgrib` (see requirements.txt)
#
# **Trade-off**: GRIB2 parsing is slower than Open-Meteo's JSON API, but gives true forecast parity.

# COMMAND ----------

# GRIB2 parsing requires xarray with a GRIB engine (see requirements.txt)
import xarray as xr
from datetime import datetime, timedelta

# HRRR forecast archive on AWS S3 (public, no credentials needed)
HRRR_BASE = "s3://noaa-hrrr-bdp-pds"
HRRR_FORECAST_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.hrrr_forecast_weather"

def hrrr_24h_forecast(lat, lon, target_date):
    """
    Get the 24-hour-ahead HRRR forecast for a specific location.
    
    Args:
        lat, lon: Location coordinates
        target_date: The date we want to forecast (YYYY-MM-DD)
    
    Returns:
        DataFrame with hourly forecasts for target_date
    """
    # HRRR runs 4 times daily: 00z, 06z, 12z, 18z
    # For day-ahead: use the 00z run from the previous day (24h ahead)
    issue_date = datetime.strptime(target_date, "%Y-%m-%d") - timedelta(days=1)
    issue_hour = "00"  # 00z run for day-ahead forecast
    
    # HRRR file path format: hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfsfcfFF.grib2
    # FF = forecast hour (00-48)
    base_path = f"{HRRR_BASE}/hrrr.{issue_date.strftime('%Y%m%d')}/conus"
    
    hourly_data = []
    
    # Pull 24 forecast hours (24-47 for tomorrow)
    for fhour in range(24, 48):
        grib_file = f"{base_path}/hrrr.t{issue_hour}z.wrfsfcf{fhour:02d}.grib2"
        
        try:
            # Open GRIB2 file with xarray + cfgrib
            ds = xr.open_dataset(
                grib_file,
                engine='cfgrib',
                backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}}
            )
            
            # Extract nearest point to our coordinates
            point = ds.sel(latitude=lat, longitude=lon, method='nearest')
            
            # Map HRRR variables to our schema
            hourly_data.append({
                'time': issue_date + timedelta(hours=fhour),
                'temperature_2m': float(point['t2m'].values) - 273.15,  # K → C
                'apparent_temperature': float(point['t2m'].values) - 273.15,  # Simplified
                'dew_point_2m': float(point['d2m'].values) - 273.15 if 'd2m' in point else None,
                'relative_humidity_2m': float(point['r2'].values) if 'r2' in point else None,
                'wind_speed_10m': float(point['si10'].values) if 'si10' in point else None,
                'cloud_cover': None,  # HRRR has total cloud cover but different encoding
                'shortwave_radiation': float(point['dswrf'].values) if 'dswrf' in point else None,
                'direct_normal_irradiance': None,  # Not directly available
            })
            
            ds.close()
            
        except Exception as e:
            print(f"Warning: Could not load {grib_file}: {e}")
            continue
    
    return pd.DataFrame(hourly_data)

# HRRR INGESTION: Now active for realistic forecast training
# WARNING: This is SLOW - test on a small date range first!

# --- STEP 1: Test on a small sample first (1 week) ---
print("\n" + "="*80)
print("HRRR INGESTION - SAMPLE RUN (1 week test)")
print("="*80)

# Test period: Just pull 1 week to estimate runtime
TEST_START = "2025-12-31"
TEST_END = "2026-01-07"    # 1 week later

print(f"\nTest period: {TEST_START} to {TEST_END}")
print("If this works well, expand to full period (2015-2026)...\n")

frames = []
for c in CITIES:
    print(f"\nPulling HRRR forecasts for {c['city']}...")
    
    # Loop through each day in test period
    start_date = datetime.strptime(TEST_START, "%Y-%m-%d")
    end_date = datetime.strptime(TEST_END, "%Y-%m-%d")
    
    current_date = start_date
    day_count = 0
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        day_count += 1
        print(f"  Day {day_count}: {date_str}...", end=" ")
        
        try:
            wdf = hrrr_24h_forecast(c["lat"], c["lon"], date_str)
            if len(wdf) > 0:
                wdf["city"] = c["city"]
                wdf["lat"] = c["lat"]
                wdf["lon"] = c["lon"]
                wdf["load_weight"] = c["load_weight"]
                frames.append(wdf)
                print(f"{len(wdf)} hours")
            else:
                print("No data")
        except Exception as e:
            print(f"Failed: {str(e)[:50]}...")
        
        current_date += timedelta(days=1)
        time.sleep(0.1)  # Be nice to AWS S3

if frames:
    print(f"\n{'='*80}")
    print(f"Successfully pulled {len(frames)} city-days of HRRR forecasts")
    print(f"Writing to {HRRR_FORECAST_TABLE}...")
    
    hrrr_pdf = pd.concat(frames, ignore_index=True)
    hrrr_sdf = (
        spark.createDataFrame(hrrr_pdf)
        .withColumnRenamed("time", "period")
        .withColumn("_source", F.lit("NOAA HRRR 24h-ahead forecast"))
        .withColumn("_ingested_at", F.current_timestamp())
    )
    (hrrr_sdf.write
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(HRRR_FORECAST_TABLE))
    
    row_count = hrrr_sdf.count()
    print(f"Wrote {row_count:,} rows → {HRRR_FORECAST_TABLE}")
    print(f"\nEstimated full backfill time: ~{(row_count / 168 * 4000) / 3600:.1f} hours")
    print("(Adjust TEST_START/TEST_END above to run full historical period)")
    print("="*80)
else:
    print("\nNo HRRR data retrieved - check AWS S3 access and GRIB2 library install")

# COMMAND ----------

# ## Open-Meteo Forecast API (Operational Weather)
#
# For live forecasting (not historical backtesting), weather comes from Open-Meteo's **forecast** API:
#
# - Free, no API key needed
# - 16-day forecast horizon
# - Same JSON format as archive (easy to parse)
# - Same schema as the historical weather table
# - Only works for **future** periods (can't backtest historical)
#
# **When to use each**:
# 1. **ERA5 Archive** (current setup) = Perfect weather hindsight, good for model development
# 2. **HRRR Archive** (above) = Historical forecasts, good for realistic backtesting
# 3. **Open-Meteo Forecast** (below) = Operational deployment going forward

# COMMAND ----------

# DBTITLE 1,Operational Forecast Ingestion
# ============================================================================
# OPERATIONAL FORECAST INGESTION (Active for Production)
# ============================================================================
# This pulls FORECASTED weather (not historical) for operational predictions.
# Run this daily/hourly to keep fresh forecasts available.
# ============================================================================

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.open_meteo_forecast_weather"

def open_meteo_forecast(lat, lon, variables):
    """
    Get 16-day weather forecast from Open-Meteo.
    Use this for OPERATIONAL forecasting going forward.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(variables),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
        "forecast_days": 16,  # Max available (384 hours)
    }
    resp = _get_with_retry(FORECAST_URL, params=params, timeout=60)
    hourly = resp.json()["hourly"]
    df = pd.DataFrame(hourly)
    missing = [v for v in variables if v not in df.columns]
    if missing:
        raise ValueError(f"Open-Meteo forecast API returned no data for: {missing}")
    return df

print("\n" + "="*80)
print("OPERATIONAL FORECAST INGESTION")
print("="*80)
print(f"Pulling 16-day forecasts for {len(CITIES)} cities...\n")

frames = []
for c in CITIES:
    print(f"  {c['city']:15s}", end=" ")
    try:
        wdf = open_meteo_forecast(c["lat"], c["lon"], WEATHER_VARS)
        wdf["city"] = c["city"]
        wdf["lat"] = c["lat"]
        wdf["lon"] = c["lon"]
        wdf["load_weight"] = c["load_weight"]
        frames.append(wdf)
        print(f"{len(wdf)} hours")
    except Exception as e:
        print(f"Failed: {e}")
    time.sleep(0.4)

if frames:
    forecast_pdf = pd.concat(frames, ignore_index=True)
    forecast_sdf = (
        spark.createDataFrame(forecast_pdf)
        .withColumnRenamed("time", "period")
        .withColumn("_source", F.lit("Open-Meteo 16-day forecast"))
        .withColumn("_ingested_at", F.current_timestamp())
    )
    
    # OVERWRITE: Fresh forecast replaces old one
    (forecast_sdf.write
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(FORECAST_TABLE))
    
    first_period = forecast_pdf['time'].min()
    last_period = forecast_pdf['time'].max()
    
    print(f"\n{'='*80}")
    print(f"Wrote {forecast_sdf.count():,} rows → {FORECAST_TABLE}")
    print(f"  Coverage: {first_period} → {last_period}")
    print(f"  Source: Open-Meteo 16-day forecast (operational)")
    print("="*80)
else:
    print("\nNo forecast data retrieved")

# COMMAND ----------

# ## Sanity checks

# COMMAND ----------

 
# Demand: expect two types (D, DF), continuous hourly coverage from 2015-07
display(
    spark.table(DEMAND_TABLE)
    .groupBy("type", "type_name")
    .agg(F.min("period").alias("first"),
         F.max("period").alias("last"),
         F.count("*").alias("rows"),
         F.sum(F.col("value").isNull().cast("int")).alias("null_values"))
)
 
# Weather: expect one block per city, same span, no missing variable columns
display(
    spark.table(WEATHER_TABLE)
    .groupBy("city")
    .agg(F.min("period").alias("first"),
         F.max("period").alias("last"),
         F.count("*").alias("rows"))
)