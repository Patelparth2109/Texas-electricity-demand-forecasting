# Databricks notebook source
# # Gold Layer — Feature Engineering, Split, and Baseline
#
# The Gold layer transforms the conformed Silver table into a **leakage-safe, model-ready feature dataset** for day-ahead energy demand forecasting.
#
# ## Core Rule: Only Use Information Available at Forecast Time
#
# The forecasting horizon is **24 hours ahead**, so every feature must reflect information that would actually be available when the forecast is generated.
#
# ### Demand-Derived Features
#
# Demand lags and rolling statistics must use observations from **`t - 24` or earlier**.
#
# Examples:
#
# - `lag_24` → demand from the same hour one day earlier
# - `lag_168` → demand from the same hour one week earlier
# - Rolling statistics → windows ending at `t - 24`
#
# A rolling window ending at `t - 1` would leak demand observations from inside the 24-hour forecast horizon and therefore cannot be used.
#
# ### Weather Features
#
# Weather features may correspond to the **target hour `t`** because, in production, the model will receive a weather forecast for that hour.
#
# During development:
#
# - Train using ERA5 observed weather
# - Serve using forecast weather
# - Evaluate the final production model using forecast weather for a realistic performance estimate
#
# This train/serve difference is tracked as **R-4**.
#
# ### Calendar Features
#
# Calendar variables are known in advance and are therefore safe at any forecast horizon.
#
# Examples include:
#
# - Hour of day
# - Day of week
# - Weekend indicator
# - Month / season
# - Holiday indicators
# - Cyclical time encodings
#
# ## Why Row-Based Windows Are Safe
#
# Silver guarantees a **continuous, gap-free hourly UTC spine** where:
#
# `1 row = 1 hour`
#
# Therefore, Spark row-offset windows map directly to elapsed time.
#
# For example:
#
# `rowsBetween(-47, -24)`
#
# represents the **24-hour window ending exactly 24 hours before target hour `t`**.
#
# This allows lag and rolling-window features to be constructed deterministically while preserving the day-ahead leakage boundary.

# COMMAND ----------

# ## Config

# COMMAND ----------

import math
from pyspark.sql import functions as F
from pyspark.sql.window import Window
 
CATALOG      = "dev"
SILVER_TABLE = f"{CATALOG}.stlf_silver.demand_weather_hourly"
GOLD_TABLE   = f"{CATALOG}.stlf_gold.demand_features_hourly"
 
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.stlf_gold")
spark.conf.set("spark.sql.session.timeZone", "UTC")
 
HORIZON   = 24            # day-ahead. All demand features must be lagged >= this.
BASE_C    = 18.33         # 65 °F, standard HDD/CDD balance point for electricity
 
# Chronological split — NEVER random for time series. Contiguous blocks, test is
# strictly the most recent data (mimics forecasting the future you haven't seen).
TRAIN_END = "2025-08-31"  # train: through summer 2025 (includes 2025 peak season)
VAL_END   = "2025-12-31"  # val:   Sep-Dec 2025
                          # test:  2026 onward (most recent data)

# COMMAND ----------

# ## Calendar & cyclical features (leakage-free)

# COMMAND ----------

import holidays

holiday_strs = sorted({d.strftime("%Y-%m-%d") for d in holidays.US(years=range(2019, 2028))})
sv = spark.table(SILVER_TABLE)

def cyc(col, period):
    """sin/cos pair so the model sees hour 23 and hour 0 as adjacent."""
    ang = 2 * math.pi * F.col(col) / F.lit(period)
    return F.sin(ang), F.cos(ang)

sin_h, cos_h = cyc("hour_local", 24)

# dayofweek_local is 1..7 → shift to 0..6 before encoding
ang_dow = 2 * math.pi * (F.col("dayofweek_local") - 1) / F.lit(7)
sin_dy, cos_dy = cyc("dayofyear", 365.25)

g = (
    sv
    .withColumn("sin_hour", sin_h).withColumn("cos_hour", cos_h)
    .withColumn("sin_dow", F.sin(ang_dow)).withColumn("cos_dow", F.cos(ang_dow))
    .withColumn("sin_doy", sin_dy).withColumn("cos_doy", cos_dy)
    .withColumn("is_holiday", F.date_format("ts_local", "yyyy-MM-dd").isin(holiday_strs))
)

# COMMAND ----------

# ## Weather features (serve-available via the weather forecast)

# COMMAND ----------

# Windows ending at t are fine for weather: you have a forecast for the target hour.

w_wx24 = Window.orderBy("ts_utc").rowsBetween(-23, 0)
 
g = (
    g
    # Degree-days: the load-vs-temperature relationship is V-shaped, not linear.
    .withColumn("cdd_c", F.greatest(F.lit(0.0), F.col("temp_c") - F.lit(BASE_C)))
    .withColumn("hdd_c", F.greatest(F.lit(0.0), F.lit(BASE_C) - F.col("temp_c")))
    # 24h mean temp captures building thermal inertia (today's load leans on
    # yesterday's heat). Serve-available because it's all forecast weather.
    .withColumn("temp_roll_mean_24", F.avg("temp_c").over(w_wx24))
)

# COMMAND ----------

# ## Demand lags & rolling stats — all end at t − HORIZON

# COMMAND ----------

w_ord    = Window.orderBy("ts_utc")
# rolling windows OFFSET back by the horizon so nothing inside the 24h we're
# forecasting can leak in:
w_roll24  = Window.orderBy("ts_utc").rowsBetween(-(HORIZON + 23),  -HORIZON)   # 24h ending t-24
w_roll168 = Window.orderBy("ts_utc").rowsBetween(-(HORIZON + 167), -HORIZON)   # 7d  ending t-24
 
g = (
    g
    .withColumn("lag_24",  F.lag("demand_mw", 24).over(w_ord))    # same hour yesterday
    .withColumn("lag_48",  F.lag("demand_mw", 48).over(w_ord))    # two days ago
    .withColumn("lag_168", F.lag("demand_mw", 168).over(w_ord))   # same hour last week
    .withColumn("roll_mean_24",  F.avg("demand_mw").over(w_roll24))
    .withColumn("roll_std_24",   F.stddev("demand_mw").over(w_roll24))
    .withColumn("roll_mean_168", F.avg("demand_mw").over(w_roll168))
)
# Note: lags landing on a flagged missing hour are null; rolling avg/stddev skip
# nulls, so a few gaps degrade gracefully rather than poisoning the feature.

# COMMAND ----------

# ## Split, trainability flag, write

# COMMAND ----------

g = (
    g
    .withColumn("split",
        F.when(F.to_date("ts_utc") <= F.lit(TRAIN_END), "train")
         .when(F.to_date("ts_utc") <= F.lit(VAL_END),   "val")
         .otherwise("test"))
    # a row is usable only if it has a target AND its longest lag is populated
    # (drops the ~168h warm-up at the very start and any rows adjacent to gaps)
    .withColumn("is_trainable",
        F.col("demand_mw").isNotNull() & F.col("lag_168").isNotNull())
)
 
g = g.withColumn("hour_day_interaction", F.concat_ws("_", F.col("dayofweek_local"), F.col("hour_local")))

FEATURES = [
    "sin_hour",
    "cos_hour",
    "sin_dow",
    "cos_dow",
    "sin_doy",
    "cos_doy",
    "is_holiday",
    "hour_day_interaction",
    "temp_c",
    "apparent_temp_c",
    "dew_point_c",
    "rel_humidity",
    "wind_speed_ms",
    "cloud_cover",
    "shortwave_rad",
    "direct_normal_rad",
    "cdd_c",
    "hdd_c",
    "temp_roll_mean_24",
    "lag_24",
    "lag_48",
    "lag_168",
    "roll_mean_24",
    "roll_std_24",
    "roll_mean_168",
]

gold = g.select(
    "ts_utc",
    "ts_local",
    "respondent",
    "demand_mw",               # target
    "eia_dayahead_forecast_mw", # baseline to beat (kept out of FEATURES)
    *FEATURES,
    "split",
    "is_trainable",
    "demand_is_missing",
    "weather_is_missing",
    F.current_timestamp().alias("_gold_built_at"),
)

(gold.write
 .mode("overwrite").option("overwriteSchema", "true")
 .saveAsTable(GOLD_TABLE))

print(f"Wrote {gold.count():,} rows → {GOLD_TABLE}")
print("Feature columns:", len(FEATURES))

# IMPORTANT: scale/normalize in the modeling step, fitting the scaler on the TRAIN
# split ONLY, then applying to val/test. Fitting on all data leaks test statistics.

# COMMAND ----------

# ## Validation

# COMMAND ----------

gd = spark.table(GOLD_TABLE)
 
# 1) Split is chronological and non-overlapping
print("Split time ranges (should be strictly increasing, no overlap):")
gd.groupBy("split").agg(
    F.min("ts_utc").alias("first"), F.max("ts_utc").alias("last"),
    F.count("*").alias("rows"),
    F.sum(F.col("is_trainable").cast("int")).alias("trainable"),
).orderBy("first").show(truncate=False)
 
# 2) Warm-up / gap nulls in the longest lag (expected ~168 + a bit)
print("Rows with null lag_168 (warm-up + gap-adjacent):",
      gd.filter("lag_168 IS NULL").count())
 
# 3) Sanity: the workhorse features should correlate strongly with demand
print("corr(demand, lag_24)  =", round(gd.stat.corr("demand_mw", "lag_24"), 3))
print("corr(demand, lag_168) =", round(gd.stat.corr("demand_mw", "lag_168"), 3))
print("corr(demand, cdd_c)   =", round(gd.stat.corr("demand_mw", "cdd_c"), 3))

# 4) THE NUMBER TO BEAT — EIA's own day-ahead forecast on the test window.
#    The model must beat this MAPE to add value.
baseline = (gd
    .filter("split = 'test' AND NOT demand_is_missing "
            "AND eia_dayahead_forecast_mw IS NOT NULL")
    .select(
        (F.avg(F.abs((F.col("demand_mw") - F.col("eia_dayahead_forecast_mw"))
                     / F.col("demand_mw"))) * 100).alias("DF_MAPE_pct"),
        F.avg(F.abs(F.col("demand_mw") - F.col("eia_dayahead_forecast_mw")))
            .alias("DF_MAE_mw"),
    ))
print("EIA day-ahead forecast baseline on TEST:")
baseline.show()
 