# app.py

import streamlit as st
import pandas as pd
from src.preprocess import load_and_preprocess_data
from src.optimization import run_promotion_optimization
from src.insights import generate_ai_summary

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Retail Decision Support System",
    page_icon="🤖",
    layout="wide"
)

# --- Title and Description ---
st.title("🤖 Autonomous AI-Driven Retail Decision Support System")
st.write("""
This system automates the process of promotion optimization and insight generation.
Upload your 9 Olist CSV files to begin the analysis.
""")
st.divider()

# --- Main Application ---
# Create the file uploader
uploaded_files = st.file_uploader(
    "Upload your Olist CSV files",
    accept_multiple_files=True,
    type='csv'
)

# Check if the correct number of files have been uploaded
if uploaded_files and len(uploaded_files) == 9:
    with st.spinner("Running full pipeline... This may take a few minutes..."):
        
        # Create a dictionary to pass to the processing function
        file_dict = {file.name.split('_dataset')[0].replace('olist_', ''): file for file in uploaded_files}
        
        # --- Step 1: Data Preprocessing ---
        st.write("🔄 Step 1: Loading and preprocessing data...")
        master_df = load_and_preprocess_data(file_dict)
        st.success("✅ Data preprocessing complete!")

        # --- Step 2, 3, 4... (same as before) ---
        st.write("📊 Step 2: Preparing data for optimization...")
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
        st.success("✅ Optimization data prepared!")

        st.write("🧠 Step 3: Solving for the optimal promotion plan...")
        optimization_results = run_promotion_optimization(product_summary)
        st.success("✅ Optimization complete!")

        st.write("✍️ Step 4: Generating AI-powered summary...")
        ai_report = generate_ai_summary(optimization_results)
        st.success("✅ AI summary generated!")

        st.divider()
        st.subheader("🎉 Your Recommended Weekly Promotion Plan")
        st.markdown(ai_report)
else:
    st.info("Please upload all 9 required CSV files to start the analysis.")