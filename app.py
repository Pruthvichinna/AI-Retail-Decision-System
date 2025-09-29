# app.py

import streamlit as st
import pandas as pd
from src.preprocess import load_and_preprocess_data
from src.forecasting import train_forecasting_model # Not used yet, but good to have
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
Welcome to the future of retail management. This system automates the complex process of demand forecasting,
promotion optimization, and insight generation. Click the button below to run the full analysis pipeline.
""")
st.divider()

# --- Main Application ---
if st.button("▶️ Generate Weekly Promotion Plan"):
    with st.spinner("Running full pipeline... This may take a few minutes..."):
        
        # --- Step 1: Data Preprocessing ---
        st.write("🔄 Step 1: Loading and preprocessing data...")
        master_df = load_and_preprocess_data('data/')
        st.success("✅ Data preprocessing complete!")

        # --- Step 2: Prepare data for optimization (same logic as notebook) ---
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

        # --- Step 3: Run Promotion Optimization ---
        st.write("🧠 Step 3: Solving for the optimal promotion plan...")
        optimization_results = run_promotion_optimization(product_summary)
        st.success("✅ Optimization complete!")

        # --- Step 4: Generate AI Insights ---
        st.write("✍️ Step 4: Generating AI-powered summary...")
        model_path = "models/flan-t5-small"
        ai_report = generate_ai_summary(optimization_results, model_path=model_path)
        st.success("✅ AI summary generated!")

        # --- Final Output ---
        st.divider()
        st.subheader("🎉 Your Recommended Weekly Promotion Plan")
        st.markdown(ai_report)