# src/forecasting.py

import pandas as pd
import xgboost as xgb
import streamlit as st # <-- ADD THIS LINE

def create_time_features(df):
    """Creates time series features from a datetime index."""
    features_df = pd.DataFrame(index=df.index)
    features_df['dayofweek'] = features_df.index.dayofweek
    features_df['quarter'] = features_df.index.quarter
    features_df['month'] = features_df.index.month
    features_df['year'] = features_df.index.year
    features_df['dayofyear'] = features_df.index.dayofyear
    return features_df

@st.cache_resource # <-- ADD THIS DECORATOR
def train_forecasting_model(daily_sales):
    """Trains the XGBoost model on the entire historical data."""
    X = create_time_features(daily_sales)
    y = daily_sales

    reg = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        objective='reg:squarederror'
    )
    reg.fit(X, y, verbose=False)
    return reg

def generate_forecast(model, future_days, last_date):
    """Generates a sales forecast for a set number of future days."""
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='D')
    
    future_features = create_time_features(pd.DataFrame(index=future_dates))
    
    forecast_values = model.predict(future_features)
    
    forecast = pd.Series(forecast_values, index=future_dates)
    return forecast