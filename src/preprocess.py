# src/preprocess.py

import pandas as pd
import os

def load_and_preprocess_data(uploaded_files_dict):
    """
    This function loads all the Olist datasets from uploaded files, merges them,
    cleans the data, and performs feature engineering.

    Args:
        uploaded_files_dict (dict): A dictionary mapping filenames to uploaded file objects.

    Returns:
        pandas.DataFrame: The final, cleaned, and feature-enriched DataFrame.
    """
    # --- 1. Load the Data ---
    # Read the data from the dictionary of uploaded files
    dfs = {name: pd.read_csv(uploaded_files_dict[name]) for name in uploaded_files_dict}

    # --- 2. Clean & Convert Data Types ---
    for col in ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date']:
        dfs['orders'][col] = pd.to_datetime(dfs['orders'][col], errors='coerce')
    
    # --- 3. Merge Tables ---
    merged_df = pd.merge(dfs['orders'], dfs['order_items'], on='order_id', how='inner')
    merged_df = pd.merge(merged_df, dfs['customers'], on='customer_id', how='inner')

    # --- 4. Feature Engineering ---
    delivery_time = merged_df['order_delivered_customer_date'] - merged_df['order_purchase_timestamp']
    merged_df['delivery_time_days'] = delivery_time.dt.total_seconds() / (24 * 60 * 60)

    delivery_diff = merged_df['order_estimated_delivery_date'] - merged_df['order_delivered_customer_date']
    merged_df['delivery_diff_days'] = delivery_diff.dt.total_seconds() / (24 * 60 * 60)

    merged_df['purchase_month'] = merged_df['order_purchase_timestamp'].dt.month_name()
    merged_df['purchase_day_of_week'] = merged_df['order_purchase_timestamp'].dt.day_name()
    
    return merged_df