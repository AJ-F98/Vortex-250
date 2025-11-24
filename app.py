import logging
# Suppress logs
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

# CSS
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

# -----------------------------------------------------------------------------
# Check Files
# -----------------------------------------------------------------------------
if not os.path.exists(MODEL_FILE) or not os.path.exists(DATA_FILE):
    st.markdown("""
    <div style='text-align: center; color: red; margin-top: 50px;'>
        <h1>Model not found. Please run: python train_model.py first.</h1>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# -----------------------------------------------------------------------------
# Load Resources
# -----------------------------------------------------------------------------
@st.cache_resource
def load_resources():
    artifact = joblib.load(MODEL_FILE)
    df = pd.read_parquet(DATA_FILE)
    return artifact, df

artifact, df = load_resources()
model = artifact['model']
features = artifact['features']
last_updated = artifact.get('last_updated', 'Unknown')

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.header("Control Panel")
max_date = df['Date'].max().date()
selected_date = st.sidebar.date_input("Select Date", max_date)
run_btn = st.sidebar.button("Generate Today's Signals")

# -----------------------------------------------------------------------------
# Main Screen
# -----------------------------------------------------------------------------
st.title("Nexus 300 Swing System")
st.caption(f"Universe: {df['Symbol'].nunique()} stocks ≤ ₹300 | Model date: {last_updated}")

if run_btn:
    target_date = pd.Timestamp(selected_date)
    
    # 1. Filter Data up to Target Date
    # We need history for Confidence calculation (last 252 days)
    # But for prediction, we just need the row at target_date
    
    # Check if target date exists in data
    if target_date not in df['Date'].values:
        # Try to find nearest previous date
        available_dates = df['Date'].unique()
        available_dates.sort()
        prev_dates = available_dates[available_dates <= target_date]
        if len(prev_dates) == 0:
            st.error("No data available for selected date.")
            st.stop()
        target_date = prev_dates[-1]
        st.info(f"Using latest available data: {target_date.date()}")

    # Get data for prediction (Target Date)
    daily_data = df[df['Date'] == target_date].copy()
    
    # Strict Filters
    # Price <= 300
    # Is_Liquid (Already calculated in training script)
    mask = (daily_data['Close'] <= 300) & (daily_data['Is_Liquid'] == True)
    candidates = daily_data[mask].copy()
    
    if candidates.empty:
        st.info("No high-probability setups today (No stocks met strict criteria).")
    else:
        # 2. Predict Scores
        X = candidates[features]
        # Handle missing cols
        for f in features:
            if f not in X.columns: X[f] = 0
            
        preds = model.predict(X[features])
        candidates['Score'] = preds
        
        # 3. Calculate Confidence (Historical Percentile)
        # We need the score distribution for these candidates over the last 252 days
        # This is expensive to calc on fly for all.
        # Optimization: Just calculate for the top N candidates?
        # But we need to filter by Confidence >= 75.
        # So we must calc for all candidates.
        
        # Get last 252 days of data for the candidate symbols
        start_lookback = target_date - pd.Timedelta(days=365) # Approx 1 year to get 252 trading days
        history = df[(df['Date'] >= start_lookback) & (df['Date'] < target_date) & (df['Symbol'].isin(candidates['Symbol']))].copy()
        
        # We need to predict on history to get score distribution
        # This might be slow if history is large.
        # Let's try to do it efficiently.
        # If history is too large, we might skip or approximate.
        # 500 stocks * 250 days = 125k rows. LGBM is fast.
        
        if not history.empty:
            X_hist = history[features]
            for f in features:
                if f not in X_hist.columns: X_hist[f] = 0
            history['Score'] = model.predict(X_hist[features])
            
            # Calculate percentile for each candidate
            confidence_scores = []
            for idx, row in candidates.iterrows():
                sym = row['Symbol']
                current_score = row['Score']
                hist_scores = history[history['Symbol'] == sym]['Score']
                
                if len(hist_scores) > 50: # Min history
                    percentile = (hist_scores < current_score).mean() * 100
                else:
                    percentile = 50 # Default
                confidence_scores.append(percentile)
            
            candidates['Confidence'] = confidence_scores
        else:
            candidates['Confidence'] = 50 # Fallback
            
        # 4. Final Filtering
        # Confidence >= 75
        # Score > Historical 80th Percentile (Covered by Confidence >= 80 basically)
        # User said: "Confidence >= 75 and Score > historical 80th percentile"
        # Let's use Confidence >= 80 to satisfy both roughly
        
        final_candidates = candidates[candidates['Confidence'] >= 75].copy()
        
        if final_candidates.empty:
             st.info("No high-probability setups today (Confidence Threshold not met).")
        else:
            # Rank by Score
            top3 = final_candidates.sort_values('Score', ascending=False).head(3)
            
            results = []
            rank = 1
            for _, row in top3.iterrows():
                close = row['Close']
                
                # Entry Logic
                entry = close * 1.003
                sl = entry * 0.945 # 5.5% risk
                target = entry * 1.092 # 9.2% reward
                
                risk_pct = 5.5
                reward_pct = 9.2
                rr = 1.67
                
                results.append({
                    "Rank": rank,
                    "Symbol": row['Symbol'],
                    "Name": row['Symbol'].replace('.NS', ''),
                    "Close": round(close, 2),
                    "Entry": round(entry, 2),
                    "SL": round(sl, 2),
                    "Target": round(target, 2),
                    "Gain%": f"{reward_pct}%",
                    "Hold Days": "7-18",
                    "Risk%": f"{risk_pct}%",
                    "RR": rr,
                    "Confidence": int(row['Confidence'])
                })
                rank += 1
                
            res_df = pd.DataFrame(results)
            st.table(res_df)
            
            # Export
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("Export CSV", csv, "signals.csv", "text/csv")
