# Databricks notebook source
# # Silver Layer — Conform, Join, Validate
#
# The Silver layer transforms the two raw Bronze tables into a single, analysis-ready **hourly dataset**.
#
# ## Core Design Decision: UTC Is the Time Spine
#
# **Every UTC hour exists exactly once.**
#
# Using UTC as the canonical time index avoids daylight saving time (DST) issues:
#
# - No missing hours during the spring-forward transition
# - No duplicated hours during the fall-back transition
# - No ambiguity when joining demand and weather data
#
# Demand and weather are joined using **UTC timestamps**. After the join, timestamps are converted to `America/Chicago` local time only for calendar-based features, since electricity demand follows local human activity patterns.
#
# This design eliminates the DST gap/duplicate risk because local time is never used as the primary time index.
#
# ## Silver Layer Responsibilities
#
# 1. **Pivot demand from long to wide format**
#    - `D` → `demand_mw`
#    - `DF` → `day_ahead_forecast_mw`
#
# 2. **Aggregate multi-city weather**
#    - Collapse weather observations from four cities into a single regional series
#    - Apply `load_weight` to create load-representative weather measures
#
# 3. **Create a continuous hourly UTC spine**
#    - Cover the overlapping time range between demand and weather
#    - Guarantee one row per UTC hour
#
# 4. **Join and validate**
#    - Left join demand and weather onto the UTC spine
#    - Flag hours with missing demand or weather observations
#
# 5. **Derive local calendar fields**
#    - Convert UTC → `America/Chicago`
#    - Create basic calendar columns needed downstream
#
# ## Layer Boundary
#
# The Silver layer remains intentionally focused on:
#
# **cleaning → conforming → joining → validating**
#
# More advanced forecasting features belong in the **Gold layer**, including:
#
# - Cyclical time encodings
# - Lag features
# - Rolling-window statistics
# - Heating Degree Days (HDD)
# - Cooling Degree Days (CDD)
# - Other model-specific engineered features

# COMMAND ----------

# ## Config
#

# COMMAND ----------

from pyspark.sql import functions as F
 
# Bronze inputs (from notebook 01)
CATALOG       = "dev"
BRONZE_SCHEMA = "stlf_bronze"
SILVER_SCHEMA = "stlf_silver"
DEMAND_BRONZE  = f"{CATALOG}.{BRONZE_SCHEMA}.eia930_demand_raw"

# Weather sources: Historical (ERA5) + Operational (Forecast)
WEATHER_HISTORICAL = f"{CATALOG}.{BRONZE_SCHEMA}.open_meteo_weather_raw"
WEATHER_FORECAST   = f"{CATALOG}.{BRONZE_SCHEMA}.open_meteo_forecast_weather"

SILVER_TABLE   = f"{CATALOG}.{SILVER_SCHEMA}.demand_weather_hourly"
 
BA        = "ERCO"
LOCAL_TZ  = "America/Chicago"   # ERCOT operates on Central time
 
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SILVER_SCHEMA}")
 
# Pin the session zone to UTC so string→timestamp parsing is unambiguous and
# stable regardless of cluster/user defaults. This is essential for the spine.
spark.conf.set("spark.sql.session.timeZone", "UTC")

# COMMAND ----------

# ## Demand: long → wide (D and DF into columns)

# COMMAND ----------

demand_raw = spark.table(DEMAND_BRONZE)
 
# Defensive de-dup: EIA occasionally republishes an hour. Collapse to one row per
# (period, type) so the pivot is clean, and report if any duplicates existed.

n_before = demand_raw.count()
demand_dedup = demand_raw.dropDuplicates(["period", "type"])
n_dupes = n_before - demand_dedup.count()
print(f"Demand duplicate (period,type) rows removed: {n_dupes:,}")
 
demand_wide = (
    demand_dedup
    .groupBy("period", "respondent")
    .pivot("type", ["D", "DF"])           # only the two types we pulled
    .agg(F.first("value", ignorenulls=True))
    .withColumnRenamed("D",  "demand_mw")
    .withColumnRenamed("DF", "eia_dayahead_forecast_mw")
    # parse the raw UTC string "yyyy-MM-ddTHH" → a real UTC timestamp
    .withColumn("ts_utc", F.to_timestamp("period", "yyyy-MM-dd'T'HH"))
    .select("ts_utc", "respondent", "demand_mw", "eia_dayahead_forecast_mw")
)

# COMMAND ----------

# ## Weather: 4 cities → one load-weighted series

# COMMAND ----------

# (bronze column, silver column)
WEATHER_MAP = [
    ("temperature_2m", "temp_c"),
    ("apparent_temperature", "apparent_temp_c"),
    ("dew_point_2m", "dew_point_c"),
    ("relative_humidity_2m", "rel_humidity"),
    ("wind_speed_10m", "wind_speed_ms"),
    ("cloud_cover", "cloud_cover"),
    ("shortwave_radiation", "shortwave_rad"),
    ("direct_normal_irradiance", "direct_normal_rad"),
]

def weighted_mean(src, tgt):
    """Load-weighted mean across cities. 
    Null-safe: a city only contributes its weight for hours where it actually 
    has a value (ERA5 is gap-free, so this is belt-and-suspenders, but it keeps 
    the weights honest if a city is ever added with partial history)."""
    w = F.when(F.col(src).isNotNull(), F.col("load_weight")).otherwise(F.lit(0.0))
    return (F.sum(F.col(src) * w) / F.sum(w)).alias(tgt)

# ============================================================================
# WEATHER SOURCE UNION: Historical (ERA5) + Operational (Forecast)
# ============================================================================
# Strategy: Use FORECAST for recent/future periods, HISTORICAL for the past.
# For overlapping periods, FORECAST takes priority (dedup by period+city).
# ============================================================================

print("Reading weather sources...")
historical = spark.table(WEATHER_HISTORICAL).withColumn("_priority", F.lit(1))  # Lower priority
forecast   = spark.table(WEATHER_FORECAST).withColumn("_priority", F.lit(0))    # Higher priority

# UNION both sources
weather_combined = historical.unionByName(forecast, allowMissingColumns=True)

# Dedup: For same (period, city), keep the one with LOWEST _priority (forecast wins)
from pyspark.sql.window import Window
weather_dedup = (
    weather_combined
    .withColumn(
        "_rank",
        F.row_number().over(
            Window.partitionBy("period", "city").orderBy("_priority", F.col("_ingested_at").desc())
        )
    )
    .filter(F.col("_rank") == 1)
    .drop("_priority", "_rank")
)

print(f"  Historical rows: {historical.count():,}")
print(f"  Forecast rows: {forecast.count():,}")
print(f"  Combined (after dedup): {weather_dedup.count():,}")

# Now aggregate as before
weather_agg = (
    weather_dedup
    # weather period carries minutes ("yyyy-MM-ddTHH:mm")
    .withColumn("ts_utc", F.to_timestamp("period", "yyyy-MM-dd'T'HH:mm"))
    .groupBy("ts_utc")
    .agg(*[weighted_mean(src, tgt) for src, tgt in WEATHER_MAP])
)

print(f"  Aggregated weather hours: {weather_agg.count():,}")

# COMMAND ----------

# ## Build a gap-free UTC spine over the overlap, then join

# COMMAND ----------

# Overlap window = where BOTH demand and weather exist. Outside it there's nothing
# to train on, so we don't manufacture spine hours there.
d_bounds = demand_wide.agg(F.min("ts_utc"), F.max("ts_utc")).first()
w_bounds = weather_agg.agg(F.min("ts_utc"), F.max("ts_utc")).first()

overlap_start = max(d_bounds[0], w_bounds[0])
overlap_end = min(d_bounds[1], w_bounds[1])
print(f"Overlap: {overlap_start} → {overlap_end}")

s = overlap_start.strftime("%Y-%m-%d %H:%M:%S")
e = overlap_end.strftime("%Y-%m-%d %H:%M:%S")

# A contiguous hourly spine is what makes gold's lag/rolling windows correct:
# without it, a window ordered by time would treat the row before a gap as "t-1".
spine = spark.sql(
    f"SELECT explode(sequence(to_timestamp('{s}'), to_timestamp('{e}'), interval 1 hour)) AS ts_utc"
)

silver = (
    spine
    .join(demand_wide, "ts_utc", "left")
    .join(weather_agg, "ts_utc", "left")
    # local wall-clock, derived from UTC — DST handled automatically by the tz db
    .withColumn("ts_local", F.from_utc_timestamp("ts_utc", LOCAL_TZ))
    # calendar attributes of the LOCAL clock (feature engineering proper is gold)
    .withColumn("hour_local", F.hour("ts_local"))
    .withColumn("dayofweek_local", F.dayofweek("ts_local"))  # 1=Sun … 7=Sat
    .withColumn("month", F.month("ts_local"))
    .withColumn("dayofyear", F.dayofyear("ts_local"))
    .withColumn("is_weekend", F.dayofweek("ts_local").isin(1, 7))
    # quality flags — never silently impute in silver
    .withColumn("demand_is_missing", F.col("demand_mw").isNull())
    .withColumn("weather_is_missing", F.col("temp_c").isNull()) 
    .withColumn("respondent", F.coalesce("respondent", F.lit(BA)))
    .withColumn("_silver_built_at", F.current_timestamp())
)

(silver.write
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable(SILVER_TABLE))

print(f"Wrote {silver.count():,} rows → {SILVER_TABLE}")

# COMMAND ----------

# ## Validation

# COMMAND ----------

sv = spark.table(SILVER_TABLE)
 
# 1) Spine integrity: distinct hours must equal expected, and no duplicate ts_utc
agg = sv.agg(
    F.min("ts_utc").alias("first"),
    F.max("ts_utc").alias("last"),
    F.count("*").alias("rows"),
    F.countDistinct("ts_utc").alias("distinct_hours"),
).first()
expected = int((agg["last"] - agg["first"]).total_seconds() // 3600) + 1
print(f"first={agg['first']}  last={agg['last']}")
print(f"rows={agg['rows']:,}  distinct_hours={agg['distinct_hours']:,}  expected={expected:,}")
print(f"contiguous & unique: {agg['rows'] == agg['distinct_hours'] == expected}")
 
# 2) How much is actually missing?
print("\nMissing counts:")
sv.select(
    F.sum(F.col("demand_is_missing").cast("int")).alias("demand_missing"),
    F.sum(F.col("weather_is_missing").cast("int")).alias("weather_missing"),
    F.sum(F.col("eia_dayahead_forecast_mw").isNull().cast("int")).alias("forecast_missing"),
).show()
 
# 3) DST proof — fall-back night (2019-11-03, clocks go back at 02:00 local).
#    Local 01:00 occurs TWICE; both survive as distinct UTC hours with distinct load.
print("Fall-back check (two distinct UTC hours both label as 01:00 local):")
sv.filter("ts_local >= '2019-11-03 01:00:00' AND ts_local < '2019-11-03 02:00:00'") \
  .select("ts_utc", "ts_local", "hour_local", "demand_mw") \
  .orderBy("ts_utc").show(truncate=False)
 
# COMMAND ----------
 
# 4) Eyeball the table + a sanity check that demand peaks late-afternoon local
display(sv.orderBy("ts_utc").limit(48))
 
display(
    sv.filter("NOT demand_is_missing")
      .groupBy("hour_local").agg(F.round(F.avg("demand_mw")).alias("avg_demand_mw"))
      .orderBy("hour_local")
)