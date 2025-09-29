# src/plotting.py

import seaborn as sns
import matplotlib.pyplot as plt

def plot_delivery_time_distribution(df):
    """Generates a histogram of delivery times."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='delivery_time_days', bins=50, kde=True, ax=ax)
    ax.set_title('Distribution of Delivery Time (in Days)', fontsize=16)
    ax.set_xlabel('Delivery Time (Days)')
    ax.set_ylabel('Number of Orders')
    ax.set_xlim(0, 60)
    return fig

def plot_delivery_estimate_diff(df):
    """Generates a histogram of delivery estimate differences."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='delivery_diff_days', bins=50, kde=True, ax=ax)
    ax.set_title('Distribution of Delivery Estimate vs. Actual', fontsize=16)
    ax.set_xlabel('Difference in Days (Positive = Early, Negative = Late)')
    ax.set_ylabel('Number of Orders')
    ax.set_xlim(-20, 80)
    return fig

def plot_orders_by_day_of_week(df):
    """Generates a count plot of orders by day of the week."""
    fig, ax = plt.subplots(figsize=(10, 6))
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    sns.countplot(data=df, x='purchase_day_of_week', order=day_order, palette='viridis', ax=ax, hue='purchase_day_of_week', legend=False)
    ax.set_title('Number of Orders by Day of the Week', fontsize=16)
    ax.set_xlabel('Day of the Week')
    ax.set_ylabel('Number of Orders')
    plt.setp(ax.get_xticklabels(), rotation=45)
    return fig