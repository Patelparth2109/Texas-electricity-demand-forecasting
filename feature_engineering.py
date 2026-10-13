"""Shared Feature Engineering for Electricity Demand Forecasting

This module provides reusable feature transformation functions
used by both training and inference pipelines.
"""

import pandas as pd
import numpy as np
from typing import List


def add_lag_features(df: pd.DataFrame, target_col: str = 'demand_mw', 
                     group_col: str = 'respondent') -> pd.DataFrame:
    """
    Add short-term lag features for demand forecasting.
    
    Args:
        df: DataFrame with demand data
        target_col: Name of the target column to create lags from
        group_col: Column to group by (e.g., utility respondent)
    
    Returns:
        DataFrame with lag features added
    """
    df = df.copy()
    
    # Short-term lags (critical for 24h forecasting)
    for lag in [1, 2, 3, 6, 12, 72]:
        df[f'lag_{lag}'] = df.groupby(group_col)[target_col].shift(lag)
    
    # Weekly cycle lags
    for lag in [167, 169]:
        df[f'lag_{lag}'] = df.groupby(group_col)[target_col].shift(lag)
    
    return df


def add_rolling_features(df: pd.DataFrame, target_col: str = 'demand_mw',
                        group_col: str = 'respondent') -> pd.DataFrame:
    """
    Add rolling statistics over multiple time windows.
    
    Args:
        df: DataFrame with demand data
        target_col: Name of the target column for rolling statistics
        group_col: Column to group by
    
    Returns:
        DataFrame with rolling features added
    """
    df = df.copy()
    
    # Multiple windows: 3h, 6h, 12h, 48h (2 days), 336h (2 weeks)
    for window in [3, 6, 12, 48, 336]:
        df[f'roll_mean_{window}'] = df.groupby(group_col)[target_col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f'roll_std_{window}'] = df.groupby(group_col)[target_col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
        )
        df[f'roll_min_{window}'] = df.groupby(group_col)[target_col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).min()
        )
        df[f'roll_max_{window}'] = df.groupby(group_col)[target_col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).max()
        )
    
    return df


def add_weather_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create weather-time interaction features.
    Temperature effects vary by time of day and day of week.
    
    Args:
        df: DataFrame with weather and cyclical time features
    
    Returns:
        DataFrame with interaction features added
    """
    df = df.copy()
    
    # Temperature-time interactions
    df['temp_x_sin_hour'] = df['temp_c'] * df['sin_hour']
    df['temp_x_cos_hour'] = df['temp_c'] * df['cos_hour']
    df['temp_x_sin_dow'] = df['temp_c'] * df['sin_dow']
    df['temp_x_cos_dow'] = df['temp_c'] * df['cos_dow']
    
    # Temperature anomaly interactions
    df['temp_anomaly'] = df['temp_c'] - df['temp_roll_mean_24']
    df['temp_anomaly_x_sin_hour'] = df['temp_anomaly'] * df['sin_hour']
    df['temp_anomaly_x_cos_hour'] = df['temp_anomaly'] * df['cos_hour']
    
    # Heating/cooling degree days with time
    df['hdd_x_sin_hour'] = df['hdd_c'] * df['sin_hour']
    df['hdd_x_cos_hour'] = df['hdd_c'] * df['cos_hour']
    df['cdd_x_sin_hour'] = df['cdd_c'] * df['sin_hour']
    df['cdd_x_cos_hour'] = df['cdd_c'] * df['cos_hour']
    
    return df


def add_demand_pattern_features(df: pd.DataFrame, target_col: str = 'demand_mw',
                               group_col: str = 'respondent') -> pd.DataFrame:
    """
    Add demand trend, momentum, and volatility features.
    
    Args:
        df: DataFrame with demand data and lag features
        target_col: Name of the target column
        group_col: Column to group by
    
    Returns:
        DataFrame with pattern features added
    """
    df = df.copy()
    
    # Trend and momentum
    df['demand_trend_weekly'] = (df['lag_24'] - df['lag_168']) / (df['lag_168'] + 1e-6)
    df['demand_acceleration'] = df['lag_24'] - 2 * df['lag_48'] + df['lag_72']
    df['demand_diff_24_48'] = df['lag_24'] - df['lag_48']
    df['demand_diff_168_336'] = df['lag_168'] - df.groupby(group_col)[target_col].shift(336)
    
    # Recent volatility
    df['demand_volatility_3h'] = df.groupby(group_col)[target_col].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).std()
    )
    
    return df


def add_calendar_features(df: pd.DataFrame, timestamp_col: str = 'ts_local') -> pd.DataFrame:
    """
    Add enhanced calendar features.
    
    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of the timestamp column
    
    Returns:
        DataFrame with calendar features added
    """
    df = df.copy()
    
    # Extract time components
    df['hour'] = pd.to_datetime(df[timestamp_col]).dt.hour
    df['dow'] = pd.to_datetime(df[timestamp_col]).dt.dayofweek
    df['day'] = pd.to_datetime(df[timestamp_col]).dt.day
    df['month'] = pd.to_datetime(df[timestamp_col]).dt.month
    
    # Binary calendar features
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['is_working_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & (df['dow'] < 5)).astype(int)
    df['is_peak_hour'] = (df['hour'].isin([7, 8, 9, 17, 18, 19, 20])).astype(int)
    df['is_month_start'] = (df['day'] <= 7).astype(int)
    df['is_month_end'] = (df['day'] >= 24).astype(int)
    df['week_of_month'] = (df['day'] - 1) // 7 + 1
    
    return df


def add_weather_trend_features(df: pd.DataFrame, group_col: str = 'respondent') -> pd.DataFrame:
    """
    Add weather change and trend features.
    
    Args:
        df: DataFrame with weather data
        group_col: Column to group by
    
    Returns:
        DataFrame with weather trend features added
    """
    df = df.copy()
    
    # Weather changes
    df['temp_diff_24'] = df['temp_c'] - df['temp_roll_mean_24']
    df['humidity_change_24'] = df['rel_humidity'] - df.groupby(group_col)['rel_humidity'].shift(24)
    df['temp_momentum'] = df['temp_c'] - df.groupby(group_col)['temp_c'].shift(3)
    df['wind_change_24'] = df['wind_speed_ms'] - df.groupby(group_col)['wind_speed_ms'].shift(24)
    
    return df


def engineer_all_features(df: pd.DataFrame, 
                         target_col: str = 'demand_mw',
                         group_col: str = 'respondent',
                         timestamp_col: str = 'ts_local') -> pd.DataFrame:
    """
    Apply all feature engineering transformations in the correct order.
    
    This is the main function to use for both training and inference.
    
    Args:
        df: DataFrame with base features from gold layer
        target_col: Name of the target column (for training) or None (for inference)
        group_col: Column to group by
        timestamp_col: Name of the timestamp column
    
    Returns:
        DataFrame with all engineered features
    """
    print(f"Starting feature engineering on {len(df):,} rows...")
    
    # Step 1: Lag features (depends on target_col)
    if target_col in df.columns:
        df = add_lag_features(df, target_col, group_col)
        print(f"  ✓ Added lag features")
    
    # Step 2: Rolling features (depends on lags)
    if target_col in df.columns:
        df = add_rolling_features(df, target_col, group_col)
        print(f"  ✓ Added rolling features")
    
    # Step 3: Weather interactions (depends on existing weather features)
    df = add_weather_interaction_features(df)
    print(f"  ✓ Added weather interactions")
    
    # Step 4: Demand patterns (depends on lags)
    if target_col in df.columns:
        df = add_demand_pattern_features(df, target_col, group_col)
        print(f"  ✓ Added demand patterns")
    
    # Step 5: Calendar features
    df = add_calendar_features(df, timestamp_col)
    print(f"  ✓ Added calendar features")
    
    # Step 6: Weather trends
    df = add_weather_trend_features(df, group_col)
    print(f"  ✓ Added weather trends")
    
    print(f"Feature engineering complete: {len(df.columns)} total columns")
    
    return df


def get_feature_columns(exclude_cols: List[str] = None) -> List[str]:
    """
    Get the list of columns to exclude from features.
    
    Args:
        exclude_cols: Additional columns to exclude
    
    Returns:
        List of column names to exclude
    """
    base_excluded = [
        'ts_utc', 'ts_local', 'respondent', 'split', 'demand_mw', 
        'eia_dayahead_forecast_mw', 'is_trainable', 'demand_is_missing', 
        'weather_is_missing', '_gold_built_at', 'hour', 'dow', 'day', 
        'month', 'hour_day_interaction'
    ]
    
    if exclude_cols:
        base_excluded.extend(exclude_cols)
    
    return base_excluded


def identify_redundant_features(df: pd.DataFrame, threshold: float = 0.95) -> List[str]:
    """
    Identify highly correlated (redundant) features.
    
    Args:
        df: DataFrame with all features
        threshold: Correlation threshold for redundancy
    
    Returns:
        List of redundant feature names to exclude
    """
    # Calculate correlation matrix
    corr_matrix = df.select_dtypes(include=[np.number]).corr().abs()
    
    # Find features with correlation > threshold
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    redundant = [column for column in upper_triangle.columns 
                if any(upper_triangle[column] > threshold)]
    
    print(f"Found {len(redundant)} redundant features (correlation > {threshold})")
    
    return redundant