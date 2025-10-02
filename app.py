# app.py
import streamlit as st
import pandas as pd
from src.preprocess import load_and_preprocess_data
from src.optimization import run_promotion_optimization
from src.insights import generate_ai_summary
from src.plotting import plot_delivery_time_distribution, plot_delivery_estimate_diff, plot_orders_by_day_of_week, plot_forecast
from src.forecasting import train_forecasting_model, generate_forecast

st.set_page_config(page_title="AI Retail Decision Support System", page_icon="🤖", layout="wide")
st.title("🤖 Autonomous AI-Driven Retail Decision Support System")
st.write("Upload your 9 Olist CSV files to begin the analysis.")
st.divider()

uploaded_files = st.file_uploader("Upload your Olist CSV files", accept_multiple_files=True, type='csv')

if uploaded_files and len(uploaded_files) == 9:
    # --- Perform all data-heavy operations once and cache them ---
    @st.cache_data
    def run_initial_analysis(files):
        file_dict = {file.name.replace('olist_', '').replace('_dataset.csv', ''): file for file in files}
        master_df = load_and_preprocess_data(file_dict)
        daily_sales = master_df.set_index('order_purchase_timestamp')['price'].resample('D').sum().fillna(0)
        return master_df, daily_sales

    master_df, daily_sales = run_initial_analysis(uploaded_files)

    # --- Display EDA and Forecasting sections ---
    st.header("Step 1: Data Preprocessing & EDA")
    st.dataframe(master_df.head())
    st.subheader("Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_delivery_time_distribution(master_df))
    with col2:
        st.pyplot(plot_delivery_estimate_diff(master_df))
    st.pyplot(plot_orders_by_day_of_week(master_df))
    st.divider()

    st.header("Step 2: Demand Forecasting")
    model = train_forecasting_model(daily_sales)
    last_date = daily_sales.index.max()
    # The argument is now `_model`
    forecast = generate_forecast(_model=model, future_days=30, last_date=last_date)
    st.pyplot(plot_forecast(daily_sales.tail(180), forecast))
    st.divider()

    # --- Step 3: Interactive Promotion Plan ---
    st.header("Step 3: Generate Promotion Plan (What-If Analysis)")

    # Add the interactive slider for budget
    budget = st.slider("Select Your Weekly Discount Budget ($)", min_value=100, max_value=2000, value=500, step=50)

    if st.button("▶️ Generate Plan"):
        with st.spinner("Running optimization and AI summary..."):
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

            # Pass the selected budget to the function
            optimization_results = run_promotion_optimization(product_summary, budget)

            ai_report = generate_ai_summary(optimization_results)

            st.subheader("🎉 Your Recommended Weekly Promotion Plan")
            st.markdown(ai_report)
else:
    st.info("Please upload all required CSV files to start the analysis.")