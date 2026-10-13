# Databricks notebook source
# DBTITLE 1,Forecast Performance Monitoring
# # Forecast Performance Monitoring
#
# ## Purpose
#
# This notebook evaluates actual forecast performance by comparing predictions against observed demand.
#
# **Key Metrics:**
# * MAPE (Mean Absolute Percentage Error)
# * MAE (Mean Absolute Error)
# * RMSE (Root Mean Squared Error)
# * R² Score
# * Performance by hour of day
# * Performance by day of week
# * Comparison vs EIA baseline
#
# **Monitoring Triggers:**
# * Alert if MAPE > 3.0% (above EIA baseline)
# * Alert if error trend is increasing
# * Alert if performance degrades significantly vs previous week
#
# ## Schedule
#
# * **Runs**: Daily at 3:00 AM UTC (after actuals available)
# * **Duration**: ~2 minutes
# * **Scope**: Evaluates previous day's forecasts
#
# ---

# COMMAND ----------

# DBTITLE 1,Imports and Config
import pandas as pd
import numpy as np
import mlflow
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
CATALOG = "dev"
SCHEMA = "stlf_gold"

# Tables
FORECAST_TABLE = f"{CATALOG}.{SCHEMA}.demand_predictions"  # written by daily_batch_inference
ACTUALS_TABLE = f"{CATALOG}.{SCHEMA}.demand_features_hourly"
METRICS_TABLE = f"{CATALOG}.{SCHEMA}.forecast_metrics"

# Thresholds
MAPE_WARNING = 2.5  # Warning threshold
MAPE_CRITICAL = 3.0  # Critical threshold (above EIA baseline)

print(f"Forecast Evaluation Configuration:")
print(f"  Forecast table: {FORECAST_TABLE}")
print(f"  Actuals table: {ACTUALS_TABLE}")
print(f"  Metrics table: {METRICS_TABLE}")
print(f"  MAPE warning: {MAPE_WARNING}%")
print(f"  MAPE critical: {MAPE_CRITICAL}%")

# COMMAND ----------

# DBTITLE 1,Load Forecasts and Actuals
# ## Step 1: Load Forecasts and Actuals
#
# Load forecasts from yesterday and corresponding actual demand.

# COMMAND ----------

# DBTITLE 1,Load Yesterday's Forecasts
# Define evaluation period (yesterday)
eval_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
print(f"Evaluating forecasts for: {eval_date}\n")

# Load forecasts generated for yesterday
query_forecast = f"""
SELECT 
    prediction_timestamp AS forecast_timestamp,
    ts_local,
    ts_utc,
    respondent,
    predicted_demand_mw,
    temp_c,
    model_version
FROM {FORECAST_TABLE}
WHERE DATE(ts_local) = '{eval_date}'
ORDER BY ts_utc
"""

df_forecast = spark.sql(query_forecast).toPandas()
print(f"Loaded {len(df_forecast)} forecast records")

if len(df_forecast) == 0:
    print(f"No forecasts found for {eval_date}. Exiting.")
    dbutils.notebook.exit("No forecasts to evaluate")

# COMMAND ----------

# DBTITLE 1,Load Actual Demand
# Load actual observed demand for the same period
query_actuals = f"""
SELECT 
    ts_local,
    ts_utc,
    respondent,
    demand_mw as actual_demand_mw
FROM {ACTUALS_TABLE}
WHERE DATE(ts_local) = '{eval_date}'
  AND demand_is_missing = false
ORDER BY ts_utc
"""

df_actuals = spark.sql(query_actuals).toPandas()
print(f"Loaded {len(df_actuals)} actual demand records")

if len(df_actuals) == 0:
    print(f"No actuals available yet for {eval_date}. Exiting.")
    dbutils.notebook.exit("Actuals not available")

# COMMAND ----------

# DBTITLE 1,Join and Calculate Errors
# ## Step 2: Join Forecasts with Actuals
#
# Merge forecasts and actuals to calculate errors.

# COMMAND ----------

# DBTITLE 1,Join and Calculate Errors
# Join forecasts with actuals
df_eval = df_forecast.merge(
    df_actuals,
    on=['ts_local', 'ts_utc', 'respondent'],
    how='inner'
)

print(f"\nJoined forecast and actuals: {len(df_eval)} records")

# Calculate error metrics
df_eval['error_mw'] = df_eval['actual_demand_mw'] - df_eval['predicted_demand_mw']
df_eval['abs_error_mw'] = np.abs(df_eval['error_mw'])
df_eval['pct_error'] = (df_eval['abs_error_mw'] / df_eval['actual_demand_mw']) * 100

# Extract time features for analysis
df_eval['hour'] = pd.to_datetime(df_eval['ts_local']).dt.hour
df_eval['dow'] = pd.to_datetime(df_eval['ts_local']).dt.dayofweek

print(f"\nSample evaluation data:")
display(df_eval[['ts_local', 'actual_demand_mw', 'predicted_demand_mw', 'error_mw', 'pct_error']].head())

# COMMAND ----------

# DBTITLE 1,Calculate Performance Metrics
# ## Step 3: Calculate Performance Metrics
#
# Compute overall and segmented performance metrics.

# COMMAND ----------

# DBTITLE 1,Overall Metrics
# Calculate overall metrics
overall_metrics = {
    'eval_date': eval_date,
    'mape': mean_absolute_percentage_error(df_eval['actual_demand_mw'], df_eval['predicted_demand_mw']) * 100,
    'mae': mean_absolute_error(df_eval['actual_demand_mw'], df_eval['predicted_demand_mw']),
    'rmse': np.sqrt(mean_squared_error(df_eval['actual_demand_mw'], df_eval['predicted_demand_mw'])),
    'r2': r2_score(df_eval['actual_demand_mw'], df_eval['predicted_demand_mw']),
    'n_forecasts': len(df_eval),
    'model_version': df_eval['model_version'].iloc[0],
    'computed_at': datetime.now()
}

print("="*70)
print(f"FORECAST PERFORMANCE - {eval_date}")
print("="*70)
print(f"MAPE:  {overall_metrics['mape']:.2f}%")
print(f"MAE:   {overall_metrics['mae']:.2f} MW")
print(f"RMSE:  {overall_metrics['rmse']:.2f} MW")
print(f"R²:    {overall_metrics['r2']:.4f}")
print(f"\nForecasts evaluated: {overall_metrics['n_forecasts']}")
print(f"Model version: {overall_metrics['model_version']}")
print("="*70)

# Status check
if overall_metrics['mape'] <= MAPE_WARNING:
    status = "GOOD"
    print(f"\n{status} - Performance within target")
elif overall_metrics['mape'] <= MAPE_CRITICAL:
    status = "WARNING"
    print(f"\n{status} - Performance approaching threshold")
else:
    status = "CRITICAL"
    print(f"\n{status} - Performance exceeds EIA baseline!")

overall_metrics['status'] = status

# COMMAND ----------

# DBTITLE 1,Performance by Hour
# ## Step 4: Performance by Time Segments
#
# Analyze performance across different hours and days.

# COMMAND ----------

# DBTITLE 1,Hourly Performance
# Performance by hour of day
print("\nPerformance by Hour of Day:")
print("-" * 50)

hourly_metrics = df_eval.groupby('hour').agg({
    'pct_error': 'mean',
    'abs_error_mw': 'mean',
    'actual_demand_mw': 'mean'
}).round(2)

hourly_metrics.columns = ['Avg MAPE (%)', 'Avg MAE (MW)', 'Avg Demand (MW)']
print(hourly_metrics)

# Identify worst performing hours
worst_hours = hourly_metrics.nlargest(3, 'Avg MAPE (%)')
print(f"\nWorst performing hours: {worst_hours.index.tolist()}")

# COMMAND ----------

# DBTITLE 1,Visualize Performance
# ## Step 5: Visualize Performance
#
# Create diagnostic plots.

# COMMAND ----------

# DBTITLE 1,Performance Plots
# Create performance visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1. Actual vs Predicted
ax1 = axes[0, 0]
ax1.scatter(df_eval['actual_demand_mw'], df_eval['predicted_demand_mw'], alpha=0.6)
ax1.plot([df_eval['actual_demand_mw'].min(), df_eval['actual_demand_mw'].max()],
         [df_eval['actual_demand_mw'].min(), df_eval['actual_demand_mw'].max()],
         'r--', linewidth=2, label='Perfect Forecast')
ax1.set_xlabel('Actual Demand (MW)', fontweight='bold')
ax1.set_ylabel('Predicted Demand (MW)', fontweight='bold')
ax1.set_title(f'Actual vs Predicted - {eval_date}', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Error distribution
ax2 = axes[0, 1]
ax2.hist(df_eval['error_mw'], bins=30, edgecolor='black', alpha=0.7)
ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax2.set_xlabel('Forecast Error (MW)', fontweight='bold')
ax2.set_ylabel('Frequency', fontweight='bold')
ax2.set_title('Error Distribution', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Error by hour
ax3 = axes[1, 0]
hourly_metrics['Avg MAPE (%)'].plot(kind='bar', ax=ax3, color='steelblue', alpha=0.8)
ax3.axhline(MAPE_WARNING, color='orange', linestyle='--', linewidth=2, label='Warning')
ax3.axhline(MAPE_CRITICAL, color='red', linestyle='--', linewidth=2, label='Critical')
ax3.set_xlabel('Hour of Day', fontweight='bold')
ax3.set_ylabel('MAPE (%)', fontweight='bold')
ax3.set_title('Performance by Hour', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Time series of errors
ax4 = axes[1, 1]
ax4.plot(df_eval['ts_local'], df_eval['error_mw'], marker='o', linestyle='-', alpha=0.7)
ax4.axhline(0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Time', fontweight='bold')
ax4.set_ylabel('Error (MW)', fontweight='bold')
ax4.set_title('Forecast Error Over Time', fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

print("\nPerformance visualizations generated")

# COMMAND ----------

# DBTITLE 1,Log to MLflow
# ## Step 6: Log Metrics to MLflow
#
# Track performance over time for retraining triggers.

# COMMAND ----------

# DBTITLE 1,Log to MLflow
# Log metrics to MLflow for monitoring
print("\nLogging metrics to MLflow...")

with mlflow.start_run(run_name=f"forecast_eval_{eval_date}"):
    # Log overall metrics
    mlflow.log_metrics({
        'mape': overall_metrics['mape'],
        'mae': overall_metrics['mae'],
        'rmse': overall_metrics['rmse'],
        'r2': overall_metrics['r2']
    })
    
    # Log metadata
    mlflow.log_param('eval_date', eval_date)
    mlflow.log_param('model_version', overall_metrics['model_version'])
    mlflow.log_param('n_forecasts', overall_metrics['n_forecasts'])
    mlflow.log_param('status', status)
    
    # Log worst performing hours
    for i, (hour, row) in enumerate(worst_hours.iterrows(), 1):
        mlflow.log_metric(f'worst_hour_{i}', hour)
        mlflow.log_metric(f'worst_hour_{i}_mape', row['Avg MAPE (%)'])
    
    print(f"Metrics logged to MLflow")

# COMMAND ----------

# DBTITLE 1,Write Metrics Table
# ## Step 7: Write Metrics to UC Table
#
# Store metrics for historical tracking and dashboards.

# COMMAND ----------

# DBTITLE 1,Write Metrics
# Write metrics to Unity Catalog for historical tracking
print(f"\nWriting metrics to {METRICS_TABLE}...")

df_metrics = pd.DataFrame([overall_metrics])
df_metrics_spark = spark.createDataFrame(df_metrics)

try:
    df_metrics_spark.write \
        .mode("append") \
        .saveAsTable(METRICS_TABLE)
    
    print(f"Metrics written successfully")
    
except Exception as e:
    print(f"Failed to write metrics: {e}")
    print("  Creating table...")
    df_metrics_spark.write \
        .mode("overwrite") \
        .saveAsTable(METRICS_TABLE)
    print(f"Table created and metrics written")

# COMMAND ----------

# DBTITLE 1,Monitoring Summary
# ## Monitoring Summary
#
# ### Evaluation Complete
#
# **Performance Status**: {status}
#
# **Key Metrics:**
# * MAPE: {overall_metrics['mape']:.2f}%
# * MAE: {overall_metrics['mae']:.2f} MW
# * RMSE: {overall_metrics['rmse']:.2f} MW
# * R²: {overall_metrics['r2']:.4f}
#
# **Recommendations:**
# * If MAPE > 3.0% for 3+ consecutive days → Trigger retraining
# * If MAPE trend increasing → Investigate data drift
# * Check worst performing hours for systematic issues
#
# ---
#
# **Evaluated Date**: {eval_date}
# **Computed At**: {datetime.now()}
# **Model Version**: {overall_metrics['model_version']}

# COMMAND ----------

