# Production MLOps Guide - Electricity Demand Forecasting

## Overview

This guide documents the production-ready MLOps structure for daily electricity demand forecasting on Databricks.

### Key Principles

1. **Separation of Concerns**: Training, inference, and monitoring are separate pipelines
2. **Shared Feature Logic**: Reusable feature engineering module ensures consistency
3. **Model Registry**: Unity Catalog for governed model versioning
4. **Minimal Daily Execution**: Inference runs daily; training only when needed
5. **Monitoring & Alerting**: Automated performance tracking with retraining triggers

---

## Project Structure

```text
Electricity Demand Forecasting/
├── Notebooks/
│   ├── ingestion/
│   │   └── (existing ingestion notebooks)
│   ├── processing/
│   │   ├── 01_bronze_eia.ipynb
│   │   ├── 02_silver_conform.ipynb
│   │   └── 03_gold_features.ipynb
│   ├── modeling/
│   │   ├── Demand Forecasting Models Advanced.ipynb  (training pipeline)
│   │   └── model_evaluation.ipynb  (model comparison)
│   ├── inference/
│   │   └── daily_batch_inference.ipynb  *** NEW ***
│   └── monitoring/
│       └── forecast_evaluation.ipynb  *** NEW ***
│
├── src/
│   ├── feature_engineering.py  *** NEW - Shared module ***
│   ├── metrics.py  (optional - evaluation utilities)
│   └── utils.py  (optional - common helpers)
│
└── PRODUCTION_MLOPS_GUIDE.md  *** This file ***
```

---

## Pipeline Responsibilities

### 1. **Training Pipeline** (Ad-hoc / Periodic)

**Notebook**: `modeling/Demand Forecasting Models Advanced.ipynb`

**Purpose**: Train, validate, and register production models

**Runs**:
* Initial model development
* Periodic retraining (weekly/monthly)
* When monitoring detects performance degradation

**Steps**:
1. Load historical data from gold layer
2. Apply feature engineering (using shared module)
3. Train multiple models (Linear, RF, XGBoost, CatBoost, etc.)
4. Evaluate on validation set
5. Hyperparameter tuning (time-series CV)
6. Final training on train+val
7. Test set evaluation
8. Register best model to Unity Catalog with `@Champion` alias

**Key Changes from Original**:
* Now uses shared `feature_engineering.py` module
* Removed prediction generation (moved to inference pipeline)
* Focus on model development and registration only

**Duration**: ~20-30 minutes (with HPO)

---

### 2. **Daily Inference Pipeline** (Automated)

**Notebook**: `inference/daily_batch_inference.ipynb` *** NEW ***

**Purpose**: Generate next-day demand forecasts (24 hours)

**Runs**: Every day at 2:00 AM UTC

**Steps**:
1. Load last 30 days of demand (for lag features)
2. Load next-day weather forecast
3. Apply feature engineering (using shared module)
4. Load Champion model from Unity Catalog
5. Generate 24-hour predictions
6. Write to `dev.stlf_gold.daily_forecasts` table

**Critical**: This pipeline does **NOT**:
* Train or retrain models
* Run cross-validation or HPO
* Touch test data
* Update model registry

**Duration**: ~2-5 minutes (scoring only)

---

### 3. **Monitoring & Evaluation Pipeline** (Automated)

**Notebook**: `monitoring/forecast_evaluation.ipynb` *** NEW ***

**Purpose**: Evaluate forecast accuracy and detect degradation

**Runs**: Every day at 3:00 AM UTC (after actuals available)

**Steps**:
1. Load yesterday's forecasts
2. Load actual observed demand
3. Calculate performance metrics (MAPE, MAE, RMSE, R²)
4. Analyze by hour/day segments
5. Generate diagnostic plots
6. Log metrics to MLflow
7. Write to `dev.stlf_gold.forecast_metrics` table
8. Alert if performance degrades

**Monitoring Triggers**:
* ⚠ Warning: MAPE > 2.5%
* ✗ Critical: MAPE > 3.0% (above EIA baseline)
* Trend: 3+ consecutive days above threshold → Trigger retraining

**Duration**: ~2 minutes

---

## Shared Feature Engineering Module

**File**: `src/feature_engineering.py` *** NEW ***

**Purpose**: Single source of truth for all feature transformations

**Why Important**:
* Ensures training and inference use identical features
* Prevents train-serve skew
* Makes features maintainable and testable
* Reduces code duplication

**Functions**:

```python
# Main function - use this for both training and inference
engineer_all_features(df, target_col='demand_mw', group_col='respondent', timestamp_col='ts_local')

# Individual transformation functions (called by above)
add_lag_features(df)
add_rolling_features(df)
add_weather_interaction_features(df)
add_demand_pattern_features(df)
add_calendar_features(df)
add_weather_trend_features(df)

# Utilities
get_feature_columns(exclude_cols)  # Get columns to exclude from features
identify_redundant_features(df)  # Find highly correlated features
```

**Usage in Notebooks**:

```python
import sys
sys.path.append('/Workspace/Users/qgf22@txstate.edu/Electricity Demand Forecasting/src')
import feature_engineering as fe

# Apply all features
df_features = fe.engineer_all_features(df_demand)

# Get feature columns for modeling
excluded = fe.get_feature_columns()
feature_cols = [col for col in df_features.columns if col not in excluded]
```

---

## Unity Catalog Tables

### Input Tables

| Table | Purpose | Updated By |
|-------|---------|------------|
| `dev.stlf_gold.demand_features_hourly` | Model-ready features | Gold layer ETL |
| `dev.stlf_gold.weather_forecast_24h` | Next-day weather | Weather API job |

### Output Tables

| Table | Purpose | Updated By | Retention |
|-------|---------|------------|----------|
| `dev.stlf_gold.daily_forecasts` | Daily predictions | Inference pipeline | 90 days |
| `dev.stlf_gold.forecast_metrics` | Performance metrics | Monitoring pipeline | 1 year |
| `dev.stlf_gold.demand_predictions` | Test set predictions | Training pipeline | Permanent |

### Model Registry

| Model | Purpose | Updated By | Alias |
|-------|---------|------------|-------|
| `dev.stlf_gold.demand_forecast_catboost` | Production model | Training pipeline | `@Champion` |

---

## Databricks Jobs Configuration

### Job 1: Daily Inference

**Purpose**: Generate next-day forecasts

**Schedule**: Daily at 2:00 AM UTC (Cron: `0 2 * * *`)

**Configuration**:
```json
{
  "name": "Daily Demand Forecast Inference",
  "tasks": [
    {
      "task_key": "daily_inference",
      "notebook_task": {
        "notebook_path": "/Users/qgf22@txstate.edu/Electricity Demand Forecasting/Notebooks/inference/daily_batch_inference",
        "base_parameters": {}
      },
      "new_cluster": {
        "spark_version": "14.3.x-scala2.12",
        "node_type_id": "Standard_DS3_v2",
        "num_workers": 2
      },
      "timeout_seconds": 900,
      "max_retries": 2
    }
  ],
  "email_notifications": {
    "on_failure": ["your-email@example.com"]
  },
  "schedule": {
    "quartz_cron_expression": "0 2 * * *",
    "timezone_id": "UTC"
  }
}
```

---

### Job 2: Forecast Monitoring

**Purpose**: Evaluate forecast accuracy

**Schedule**: Daily at 3:00 AM UTC (Cron: `0 3 * * *`)

**Configuration**:
```json
{
  "name": "Forecast Performance Monitoring",
  "tasks": [
    {
      "task_key": "evaluate_forecasts",
      "notebook_task": {
        "notebook_path": "/Users/qgf22@txstate.edu/Electricity Demand Forecasting/Notebooks/monitoring/forecast_evaluation",
        "base_parameters": {}
      },
      "new_cluster": {
        "spark_version": "14.3.x-scala2.12",
        "node_type_id": "Standard_DS3_v2",
        "num_workers": 2
      },
      "timeout_seconds": 600,
      "max_retries": 1
    }
  ],
  "email_notifications": {
    "on_failure": ["your-email@example.com"],
    "on_success": []
  },
  "schedule": {
    "quartz_cron_expression": "0 3 * * *",
    "timezone_id": "UTC"
  }
}
```

---

### Job 3: Weekly Model Retraining (Optional)

**Purpose**: Periodic model refresh

**Schedule**: Weekly on Sunday at 4:00 AM UTC (Cron: `0 4 * * 0`)

**Configuration**:
```json
{
  "name": "Weekly Model Retraining",
  "tasks": [
    {
      "task_key": "retrain_model",
      "notebook_task": {
        "notebook_path": "/Users/qgf22@txstate.edu/Electricity Demand Forecasting/Notebooks/modeling/Demand Forecasting Models Advanced",
        "base_parameters": {}
      },
      "new_cluster": {
        "spark_version": "14.3.x-scala2.12",
        "node_type_id": "Standard_DS4_v2",
        "num_workers": 4
      },
      "timeout_seconds": 3600,
      "max_retries": 1
    }
  ],
  "email_notifications": {
    "on_failure": ["your-email@example.com"],
    "on_success": ["your-email@example.com"]
  },
  "schedule": {
    "quartz_cron_expression": "0 4 * * 0",
    "timezone_id": "UTC",
    "pause_status": "PAUSED"  # Enable only when needed
  }
}
```

---

## Production Workflow

### Daily Flow

```text
2:00 AM UTC - Daily Inference Job
     ↓
[Load historical demand (30 days)]
     ↓
[Load weather forecast (24h)]
     ↓
[Apply feature engineering]
     ↓
[Load @Champion model]
     ↓
[Generate predictions]
     ↓
[Write to daily_forecasts table]
     ↓
3:00 AM UTC - Monitoring Job
     ↓
[Load yesterday's forecasts]
     ↓
[Load actual demand]
     ↓
[Calculate metrics]
     ↓
[Log to MLflow + UC table]
     ↓
[Check thresholds]
     ↓
[Alert if degraded]
```

### Retraining Trigger

```text
Monitoring Job detects:
  MAPE > 3.0% for 3+ days
     ↓
[Send alert to data science team]
     ↓
[Manually trigger training job]
  OR
[Auto-trigger via job dependency]
     ↓
[Train new model]
     ↓
[Evaluate on test set]
     ↓
[If better: Register as @Champion]
     ↓
[Next inference run uses new model]
```

---

## Key Differences from Original

### Before (Original Notebook)

* Single monolithic notebook
* Mixed training and inference
* Would retrain daily (wasteful)
* Feature engineering duplicated
* Test set predictions only
* No production monitoring

### After (Refactored Structure)

* Separated pipelines by concern
* Inference runs daily (fast)
* Training only when needed
* Shared feature module
* Daily forecasts + monitoring
* Automated alerting

---

## Migration Checklist

### Prerequisites

- [ ] Review shared `feature_engineering.py` module
- [ ] Test daily inference notebook with recent data
- [ ] Configure weather forecast data source
- [ ] Set up model alias (`@Champion`) in UC
- [ ] Create output tables in Unity Catalog

### Deployment Steps

1. **Deploy Shared Module**
   ```bash
   # Module is already in: /Workspace/Users/.../src/feature_engineering.py
   # No deployment needed - notebooks import directly
   ```

2. **Test Inference Pipeline**
   - Run `daily_batch_inference` notebook manually
   - Verify 24 predictions generated
   - Check `daily_forecasts` table populated

3. **Test Monitoring Pipeline**
   - Run `forecast_evaluation` notebook manually
   - Verify metrics calculated correctly
   - Check `forecast_metrics` table populated

4. **Create Databricks Jobs**
   - Use configurations above
   - Start with daily inference job
   - Add monitoring job after 1 day
   - Enable retraining job as needed

5. **Configure Alerts**
   - Email notifications on job failure
   - Slack/PagerDuty for critical MAPE alerts
   - Weekly summary report

### Validation

- [ ] Daily inference completes in < 5 minutes
- [ ] Forecasts written to UC table daily
- [ ] Monitoring job evaluates performance
- [ ] Metrics logged to MLflow
- [ ] Alerts triggered correctly
- [ ] Power BI dashboard refreshes

---

## Troubleshooting

### Issue: Inference job fails with "Model not found"

**Solution**: Ensure model registered with `@Champion` alias

```python
import mlflow
client = mlflow.MlflowClient()

# Set Champion alias on latest version
model_name = "dev.stlf_gold.demand_forecast_catboost"
latest_version = client.get_latest_versions(model_name)[0].version
client.set_registered_model_alias(model_name, "Champion", latest_version)
```

### Issue: Feature mismatch between training and inference

**Solution**: Both pipelines must use shared `feature_engineering.py` module

```python
# Always import from shared module
import sys
sys.path.append('/Workspace/Users/.../src')
import feature_engineering as fe

# Use the main function
df_features = fe.engineer_all_features(df)
```

### Issue: Weather forecast table not populated

**Solution**: Create fallback logic or stub data

```sql
-- Create stub forecast table for testing
CREATE TABLE dev.stlf_gold.weather_forecast_24h AS
SELECT 
    ts_local + INTERVAL 1 DAY as ts_local,
    ts_utc + INTERVAL 1 DAY as ts_utc,
    respondent,
    temp_c,
    rel_humidity,
    wind_speed_ms,
    ...
FROM dev.stlf_gold.demand_features_hourly
WHERE ts_local >= current_date() - INTERVAL 1 DAY
  AND ts_local < current_date()
ORDER BY ts_utc;
```

### Issue: Monitoring shows high MAPE but forecasts look good

**Solution**: Check join keys and timestamp alignment

```python
# Verify timestamps match
df_eval = df_forecast.merge(df_actuals, on=['ts_local', 'ts_utc', 'respondent'])
print(f"Forecasts: {len(df_forecast)}")
print(f"Actuals: {len(df_actuals)}")
print(f"Matched: {len(df_eval)}")
```

---

## Next Steps

1. **Power BI Integration**
   - Connect to `daily_forecasts` table
   - Create real-time dashboard
   - Add forecast vs actual comparison

2. **Advanced Monitoring**
   - Drift detection (Evidently AI)
   - Feature importance tracking
   - Model performance by segment

3. **MLOps Enhancements**
   - A/B testing framework
   - Shadow mode deployment
   - Automated retraining pipeline
   - Model explainability

4. **Weather Integration**
   - Connect to weather API (OpenWeather, NOAA)
   - Automate forecast ingestion
   - Handle forecast updates

---

## Contact & Support

**Maintained By**: Parth Patel  
**Last Updated**: 2026-08-24  
**Version**: 1.0

For questions or issues, contact:
- **Email**: patelparth5709@gmail.com
- **Slack**: #data-science-mlops
- **Documentation**:

---

## Appendix: Code Examples

### Using Shared Feature Engineering

```python
# In any notebook
import sys
sys.path.append('/Workspace/Users/qgf22@txstate.edu/Electricity Demand Forecasting/src')
import feature_engineering as fe

# Load data
df = spark.table('dev.stlf_gold.demand_features_hourly').toPandas()

# Apply all features
df_features = fe.engineer_all_features(
    df,
    target_col='demand_mw',
    group_col='respondent',
    timestamp_col='ts_local'
)

# Get feature columns
excluded_cols = fe.get_feature_columns()
X = df_features.drop(excluded_cols, axis=1, errors='ignore')
y = df_features['demand_mw']
```

### Loading Production Model

```python
import mlflow

# Load by alias (recommended)
model_uri = "models:/dev.stlf_gold.demand_forecast_catboost@Champion"
model = mlflow.pyfunc.load_model(model_uri)

# Or load by version
model_uri = "models:/dev.stlf_gold.demand_forecast_catboost/3"
model = mlflow.pyfunc.load_model(model_uri)

# Generate predictions
y_pred = model.predict(X_test)
```

### Querying Forecast Performance

```sql
-- Latest forecast metrics
SELECT 
    eval_date,
    mape,
    mae,
    rmse,
    r2,
    status
FROM dev.stlf_gold.forecast_metrics
ORDER BY eval_date DESC
LIMIT 7;

-- Performance trend
SELECT 
    DATE_TRUNC('week', eval_date) as week,
    AVG(mape) as avg_mape,
    AVG(mae) as avg_mae
FROM dev.stlf_gold.forecast_metrics
WHERE eval_date >= CURRENT_DATE - INTERVAL 30 DAYS
GROUP BY 1
ORDER BY 1;
```

---

**End of Production MLOps Guide**