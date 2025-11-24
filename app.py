import logging
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)

import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from datetime import datetime

# Page Config
st.set_page_config(page_title="Nexus 300 Swing System", layout="wide", initial_sidebar_state="expanded")

# CSS for clean look
st.markdown("""
<style>
    .reportview-container { background: #ffffff; color: #000000; }
    .sidebar .sidebar-content { background: #f5f5f5; }
    h1, h2, h3 { font-family: 'Arial', sans-serif; color: #000000; }
    .stButton>button { background-color: #000000; color: #ffffff; border: none; }
    .stButton>button:hover { background-color: #333333; }
    div.stDataFrame { border: 1px solid #e0e0e0; }
</style>
""", unsafe_allow_html=True)

MODEL_FILE = "final_model.pkl"
DATA_FILE = "filtered_data.parquet"

# Check Files
if not os.path.exists(MODEL_FILE) or not os.path.exists(DATA_FILE):
    st.markdown("""
    <div style='text-align: center; color: red; margin-top: 50px;'>
        <h1>Model not found. Please run: python train_model.py first.</h1>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Load Resources
@st.cache_resource
def load_resources():
    artifact = joblib.load(MODEL_FILE)
    df = pd.read_parquet(DATA_FILE)
    # Fix Date column for filtering (convert to datetime if needed)
    if df['Date'].dtype == 'object':
        df['Date'] = pd.to_datetime(df['Date']).dt.date
    return artifact, df

artifact, df = load_resources()
model = artifact['model']
features = artifact['features']
last_updated = artifact.get('last_updated', 'Unknown')

# Sidebar
st.sidebar.header("Control Panel")
# Convert dates for clean calendar picker
df['Date_pd'] = pd.to_datetime(df['Date'])
max_date = df['Date_pd'].max().date()
selected_date = st.sidebar.date_input("Select Date", value=max_date, format='YYYY-MM-DD')
run_btn = st.sidebar.button("Generate Today's Signals")

# Main Screen
st.title("Nexus 300 Swing System")
st.caption(f"Universe: {df['Symbol'].nunique()} stocks ≤ ₹300 | Model date: {last_updated}")

if run_btn:
    target_date = pd.to_datetime(selected_date).date()
    
    # Find exact or nearest previous date (handles missing dates)
    df['Date_pd'] = pd.to_datetime(df['Date'])
    available_dates = df['Date_pd'].dt.date.unique()
    prev_dates = [d for d in available_dates if d <= target_date]
    if not prev_dates:
        st.error("No data available for selected or earlier dates.")
        st.stop()
    target_date = max(prev_dates)  # Nearest previous
    if target_date != pd.to_datetime(selected_date).date():
        st.info(f"Using nearest available date: {target_date.strftime('%Y-%m-%d')}")

    # Get data for target date
    daily_data = df[df['Date_pd'].dt.date == target_date].copy()
    
    if daily_data.empty:
        st.warning(f"No data found for {target_date}.")
    else:
        # Strict Filters: Price <= 300 + Liquid
        mask = (daily_data['Close'] <= 300) & (daily_data['Is_Liquid'] == True)
        candidates = daily_data[mask].copy()
        
        if candidates.empty:
            st.info("No qualifying stocks today (Price > ₹300 or illiquid).")
        else:
            # Predict Scores
            X = candidates[features].fillna(0)
            preds = model.predict(X)
            candidates['Score'] = preds
            
            # Calculate Confidence (Historical Percentile)
            start_lookback = target_date - pd.Timedelta(days=365)
            history = df[(df['Date_pd'].dt.date >= start_lookback) & 
                         (df['Date_pd'].dt.date < target_date) & 
                         (df['Symbol'].isin(candidates['Symbol']))].copy()
            
            if not history.empty and len(history) > 50:
                X_hist = history[features].fillna(0)
                history['Score'] = model.predict(X_hist)
                confidence_scores = []
                for _, row in candidates.iterrows():
                    sym = row['Symbol']
                    current_score = row['Score']
                    hist_scores = history[history['Symbol'] == sym]['Score']
                    if len(hist_scores) >= 50:
                        percentile = (hist_scores < current_score).mean() * 100
                    else:
                        percentile = 50
                    confidence_scores.append(percentile)
                candidates['Confidence'] = confidence_scores
            else:
                candidates['Confidence'] = 50  # Fallback
            
            # High-Conviction Filter (Confidence >= 75)
            final_candidates = candidates[candidates['Confidence'] >= 75].copy()
            
            if final_candidates.empty:
                st.info("No high-probability setups today (Confidence < 75%).")
            else:
                # Generate Trade Parameters
                final_candidates['Entry'] = (final_candidates['Close'] * 1.003).round(2)
                final_candidates['SL'] = (final_candidates['Entry'] * 0.945).round(2)  # ~5.5% risk
                final_candidates['Target'] = (final_candidates['Entry'] * 1.092).round(2)  # ~9.2% reward
                final_candidates['Risk%'] = 5.5
                final_candidates['Reward%'] = 9.2
                final_candidates['RR'] = 1.67
                final_candidates['Confidence'] = final_candidates['Confidence'].round(0).astype(int)
                
                # Prepare Display Table (All Signals)
                display_df = final_candidates[['Symbol', 'Close', 'Entry', 'SL', 'Target', 
                                               'Reward%', 'Risk%', 'RR', 'Confidence', 'Score']].copy()
                display_df['Symbol'] = display_df['Symbol'].str.replace('.NS', '')
                display_df = display_df.rename(columns={
                    'Symbol': 'Symbol',
                    'Close': 'Close ₹',
                    'Entry': 'Entry ₹',
                    'SL': 'Stop Loss ₹',
                    'Target': 'Target ₹',
                    'Score': 'Raw Score'
                }).round(2)
                
                st.success(f"**{len(display_df)} High-Conviction Signals Found** (All with Confidence ≥ 75%)")
                
                # Fully Sortable + Searchable Table
                st.dataframe(
                    display_df.sort_values("Confidence", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Confidence": st.column_config.ProgressColumn(
                            "Confidence %",
                            help="Historical percentile strength (Higher = Rarer/Better Edge)",
                            format="%d%%",
                            min_value=0,
                            max_value=100,
                        )
                    }
                )
                
                # Export All Signals
                csv = display_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Export All Signals (CSV)", 
                    csv, 
                    f"nexus300_signals_{target_date.strftime('%Y-%m-%d')}.csv", 
                    "text/csv"
                )

st.sidebar.caption(f"Model: {last_updated} | Universe: {df['Symbol'].nunique()} stocks")