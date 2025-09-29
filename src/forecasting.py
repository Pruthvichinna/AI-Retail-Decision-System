# src/forecasting.py

import pandas as pd
import xgboost as xgb

def create_time_features(df):
    features_df = pd.DataFrame(index=df.index)
    features_df['dayofweek'] = features_df.index.dayofweek
    features_df['quarter'] = features_df.index.quarter
    features_df['month'] = features_df.index.month
    features_df['year'] = features_df.index.year
    features_df['dayofyear'] = features_df.index.dayofyear
    return features_df

def train_forecasting_model(daily_sales):
    X = create_time_features(daily_sales)
    y = daily_sales

    # For the app, we'll just train on all the data for simplicity
    reg = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        objective='reg:squarederror'
    )
    reg.fit(X, y, verbose=False) # We set verbose to False for a cleaner app experience
    return reg