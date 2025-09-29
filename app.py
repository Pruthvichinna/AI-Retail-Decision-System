# app.py

import streamlit as st
import pandas as pd
from src.preprocess import load_and_preprocess_data
from src.optimization import run_promotion_optimization
from src.insights import generate_ai_summary
from src.plotting import plot_delivery_time_distribution, plot_delivery_estimate_diff, plot_orders_by_day_of_week

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Retail Decision Support System",
    page_icon="🤖",
    layout="wide"
)

# --- Title and Description ---
st.title("🤖 Autonomous AI-Driven Retail Decision Support System")
st.write("Upload your 9 Olist CSV files to begin the analysis.")
st.divider()

# --- Main Application ---
uploaded_files = st.file_uploader(
    "Upload your Olist CSV files",
    accept_multiple_files=True,
    type='csv'
)

if uploaded_files and len(uploaded_files) == 9:
    # --- Step 1: Data Preprocessing ---
    st.header("Step 1: Data Preprocessing & EDA")
    with st.spinner("Loading and preprocessing data..."):
        file_dict = {file.name.replace('olist_', '').replace('_dataset.csv', ''): file for file in uploaded_files}
        master_df = load_and_preprocess_data(file_dict)
    st.success("✅ Data preprocessing complete!")
    st.dataframe(master_df.head())

    # --- Display EDA Plots ---
    st.subheader("Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_delivery_time_distribution(master_df))
    with col2:
        st.pyplot(plot_delivery_estimate_diff(master_df))
    
    st.pyplot(plot_orders_by_day_of_week(master_df))
    st.divider()

    # --- Step 2: Run Optimization and Get Insights ---
    st.header("Step 2: Generate Promotion Plan")
    if st.button("▶️ Generate Weekly Promotion Plan"):
        with st.spinner("Running optimization and AI summary..."):
            # Prepare data for optimization
            min_date = master_df['order_purchase_timestamp'].min()
            max_date = master_df['order_purchase_timestamp'].max()
            number_of_weeks = (max_date - min_date).days / 7
            
            top_products = master_df.groupby('product_id')['price'].sum().nlargest(5).index
            top_products_df = master_df[master_df['product_id'].isin(top_products)]
            
            product_summary = top_products_df.groupby('product_id').agg(
                avg_price=('price', 'mean'),
                weekly_demand_forecast=('order_item_id', lambda x: x.count() / number_of_weeks)
            ).reset_index()
            
            product_summary['discount_price'] = product_summary['avg_price'] * 0.90
            product_summary['promo_demand_forecast'] = product_summary['weekly_demand_forecast'] * 1.2
            
            # Run Optimization
            optimization_results = run_promotion_optimization(product_summary)
            
            # Generate AI Insights
            ai_report = generate_ai_summary(optimization_results)

            # --- Final Output ---
            st.subheader("🎉 Your Recommended Weekly Promotion Plan")
            st.markdown(ai_report)
else:
    st.info("Please upload required CSV files to start the analysis.")