import pandas as pd
import os

def load_and_preprocess_data(data_path):
    """
    This function loads all the Olist datasets from a specified directory,
    merges them, cleans the data, and performs feature engineering.

    Args:
        data_path (str): The path to the directory containing the Olist CSV files.

    Returns:
        pandas.DataFrame: The final, cleaned, and feature-enriched DataFrame.
    """
    # --- 1. Load the Data ---
    # A dictionary to hold all the dataframes for easy access
    datasets = {
        "customers": "olist_customers_dataset.csv",
        "geolocation": "olist_geolocation_dataset.csv",
        "order_items": "olist_order_items_dataset.csv",
        "payments": "olist_order_payments_dataset.csv",
        "reviews": "olist_order_reviews_dataset.csv",
        "orders": "olist_orders_dataset.csv",
        "products": "olist_products_dataset.csv",
        "sellers": "olist_sellers_dataset.csv",
        "category_translation": "product_category_name_translation.csv"
    }
    
    dfs = {}
    for name, filename in datasets.items():
        file_path = os.path.join(data_path, filename)
        dfs[name] = pd.read_csv(file_path)

    # --- 2. Clean & Convert Data Types ---
    # Convert all relevant date columns from 'object' (text) to 'datetime'
    for col in ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date']:
        dfs['orders'][col] = pd.to_datetime(dfs['orders'][col], errors='coerce')
    
    # --- 3. Merge Tables ---
    # We'll create one master table by merging the most important datasets
    # (orders -> order_items -> customers)
    merged_df = pd.merge(dfs['orders'], dfs['order_items'], on='order_id', how='inner')
    merged_df = pd.merge(merged_df, dfs['customers'], on='customer_id', how='inner')

    # --- 4. Feature Engineering ---
    # Calculate delivery time in days
    delivery_time = merged_df['order_delivered_customer_date'] - merged_df['order_purchase_timestamp']
    merged_df['delivery_time_days'] = delivery_time.dt.total_seconds() / (24 * 60 * 60)

    # Calculate difference between estimated and actual delivery
    delivery_diff = merged_df['order_estimated_delivery_date'] - merged_df['order_delivered_customer_date']
    merged_df['delivery_diff_days'] = delivery_diff.dt.total_seconds() / (24 * 60 * 60)

    # Extract purchase month and day of the week
    merged_df['purchase_month'] = merged_df['order_purchase_timestamp'].dt.month_name()
    merged_df['purchase_day_of_week'] = merged_df['order_purchase_timestamp'].dt.day_name()
    
    print("✅ Data loading and preprocessing complete.")
    return merged_df

# This special block allows you to test the script directly by running `python preprocess.py`
if __name__ == '__main__':
    # Define the relative path to the data folder
    # This path goes up one level from 'src' to the project root, then into 'data'
    path_to_data = '../data'
    
    # Run the main function
    master_df = load_and_preprocess_data(path_to_data)
    
    # Print a summary to verify it works
    print("\n--- Master DataFrame Info ---")
    master_df.info()
    
    print("\n--- Master DataFrame Head ---")
    print(master_df.head())