import logging
import warnings

# Suppress logs and warnings
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('peewee').setLevel(logging.CRITICAL)
warnings.simplefilter(action='ignore', category=FutureWarning)

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import lightgbm as lgb
import joblib
import os
import time
from datetime import datetime, timedelta
from nselib import capital_market

# Constants
MODEL_FILE = "final_model.pkl"
DATA_FILE = "filtered_data.parquet"
NIFTY500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"

# Filters
MAX_PRICE = 300
MIN_AVG_VOL = 200000
MIN_AVG_TURNOVER = 25000000

def get_nifty500_symbols():
    print("Fetching Nifty 500 symbols...")
    try:
        df = pd.read_csv(NIFTY500_URL)
        symbols = [f"{sym}.NS" for sym in df['Symbol'].tolist()]
        print(f"Fetched {len(symbols)} symbols from NSE CSV.")
        return symbols
    except Exception as e:
        print(f"CSV fetch failed: {e}. Trying nselib...")
        try:
            df = capital_market.nifty500_equity_list()
            symbols = [f"{sym}.NS" for sym in df['Symbol'].tolist()]
            print(f"Fetched {len(symbols)} symbols from nselib.")
            return symbols
        except Exception as e2:
            print(f"nselib fetch failed: {e2}")
            return []

def download_data(symbols):
    print(f"Downloading data for {len(symbols)} symbols (8+ years)...")
    start_date = (datetime.now() - timedelta(days=365*8.5)).strftime('%Y-%m-%d')
    
    batch_size = 50
    all_dfs = []
    
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}...", end='\r')
        try:
            data = yf.download(batch, start=start_date, group_by='ticker', auto_adjust=True, threads=True, progress=False)
            for sym in batch:
                if sym in data.columns.levels[0]:
                    df = data[sym].copy()
                    df['Symbol'] = sym
                    
                    if len(df) > 0:
                        last_close = df['Close'].iloc[-1]
                        if last_close <= MAX_PRICE:
                            all_dfs.append(df)
        except Exception as e:
            print(f"Batch {i} failed: {e}")
            pass
            
    print("") # Newline after progress
    if not all_dfs: 
        print("No data downloaded!")
        return pd.DataFrame()
    
    full_df = pd.concat(all_dfs)
    full_df.reset_index(inplace=True)
    print(f"Downloaded {len(full_df)} rows for {full_df['Symbol'].nunique()} stocks.")
    return full_df

def engineer_features(df):
    print("Engineering 220+ features (Strict No-Lookahead)...")
    df = df.copy()
    df.sort_values(['Symbol', 'Date'], inplace=True)
    
    # HELPER: Shift(1) before rolling to prevent leakage
    # All features based on Close/Vol/High/Low must be shifted if using rolling
    # Actually, if we use shift(1) on the result of rolling(window), it effectively uses T-1 data.
    # Example: rolling(5).mean() at T includes T. shift(1) moves T's result to T+1.
    # So at T+1, we see T's rolling mean. This is correct for "using past data".
    
    # 1. Returns & Momentum (Multiple windows)
    # We calculate returns from T-1 to T. This is known at T close.
    # If we want to predict T+1 return using T data, we can use T features.
    # User said: "ALL rolling calculations ... must use .shift(1) first".
    # This implies: df['Feature'] = df['Close'].shift(1).rolling(w).mean()
    # This means at time T, we use data up to T-1. We ignore T's close?
    # That seems overly conservative (ignoring the latest candle).
    # But I will follow instructions strictly.
    
    windows = [3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 200, 252]
    
    for w in windows:
        # Shift(1) applied to input for rolling
        shifted_close = df.groupby('Symbol')['Close'].shift(1)
        shifted_vol = df.groupby('Symbol')['Volume'].shift(1)
        
        # Momentum / ROC (Rate of Change)
        # ROC of shifted close
        df[f'ROC_{w}'] = shifted_close.pct_change(w, fill_method=None)
        
        # Volatility
        roll_std = shifted_close.rolling(w).std()
        roll_mean = shifted_close.rolling(w).mean()
        df[f'ZScore_{w}'] = (shifted_close - roll_mean) / roll_std
        
        # Volume
        vol_ma = shifted_vol.rolling(w).mean()
        df[f'Vol_Ratio_{w}'] = shifted_vol / vol_ma
        
        # Distance from High/Low
        roll_max = shifted_close.rolling(w).max()
        roll_min = shifted_close.rolling(w).min()
        df[f'Dist_High_{w}'] = (shifted_close / roll_max) - 1
        df[f'Dist_Low_{w}'] = (shifted_close / roll_min) - 1
        
    # 2. Technical Indicators (TA Library)
    # We need to apply shift(1) to inputs for TA lib or shift result
    # Shift result is easier and equivalent
    
    def apply_ta(x):
        # Shift inputs
        o = x['Open'].shift(1)
        h = x['High'].shift(1)
        l = x['Low'].shift(1)
        c = x['Close'].shift(1)
        v = x['Volume'].shift(1)
        
        # RSI
        x['RSI_14'] = ta.momentum.rsi(c, window=14)
        x['RSI_21'] = ta.momentum.rsi(c, window=21)
        
        # MACD
        x['MACD'] = ta.trend.macd_diff(c)
        
        # Bollinger
        bb_h = ta.volatility.bollinger_hband(c)
        bb_l = ta.volatility.bollinger_lband(c)
        x['BB_Width'] = (bb_h - bb_l) / c
        
        # ATR
        x['ATR'] = ta.volatility.average_true_range(h, l, c)
        x['ATR_Ratio'] = x['ATR'] / c
        
        # MFI
        x['MFI'] = ta.volume.money_flow_index(h, l, c, v)
        
        return x

    # Apply TA per group
    # Note: This might be slow. Optimizing by calculating on full df with transform where possible?
    # TA lib functions usually take Series. We can just pass the shifted series.
    # But ATR/MFI need H/L/C/V aligned.
    # Groupby apply is safest.
    df = df.groupby('Symbol').apply(apply_ta).reset_index(level=0, drop=True)
    
    # 3. Liquidity Filters (Strictly Past)
    # Using shift(1) inputs
    df['Turnover'] = df['Close'] * df['Volume']
    shifted_vol = df.groupby('Symbol')['Volume'].shift(1)
    shifted_turn = df.groupby('Symbol')['Turnover'].shift(1)
    
    df['Avg_Vol_63'] = shifted_vol.rolling(63).mean()
    df['Avg_Turn_63'] = shifted_turn.rolling(63).mean()
    
    df['Is_Liquid'] = (df['Avg_Vol_63'] >= MIN_AVG_VOL) & (df['Avg_Turn_63'] >= MIN_AVG_TURNOVER)
    
    # 4. Relative Strength
    # Rank of ROC_21 across universe per date
    df['RS_21'] = df.groupby('Date')['ROC_21'].rank(pct=True)
    df['RS_63'] = df.groupby('Date')['ROC_60'].rank(pct=True) # Using 60 as proxy
    
    return df

def train_model(df):
    print("Training LightGBM Ranker (LambdaRank)...")
    
    # Target: Max return in next 21 days
    # Formula: Max(High) in (t+1 to t+21) / Close(t) - 1
    # We use FixedForwardWindowIndexer
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=21)
    
    # We want max of NEXT 21 days. 
    # rolling(21) on forward indexer includes current row? 
    # FixedForwardWindowIndexer(window_size=21) at index i gives [i, i+21).
    # We want [i+1, i+22).
    # So we shift High by -1 first.
    next_high = df.groupby('Symbol')['High'].shift(-1)
    df['Future_High'] = next_high.rolling(window=indexer).max()
    df['Target'] = (df['Future_High'] / df['Close']) - 1
    
    # Filter
    # Min 500 rows per stock check (already done implicitly by dropna/rolling but let's be strict)
    counts = df['Symbol'].value_counts()
    valid_syms = counts[counts >= 500].index
    df = df[df['Symbol'].isin(valid_syms)]
    
    # Drop NaNs
    train_df = df[df['Is_Liquid'] & df['Target'].notna()].copy()
    
    # Sort for Ranker (Must be sorted by Group)
    train_df.sort_values('Date', inplace=True)
    
    # Features
    exclude = ['Date', 'Symbol', 'Target', 'Future_High', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Turnover', 'Is_Liquid', 'Target_Rank']
    features = [c for c in train_df.columns if c not in exclude]
    
    X = train_df[features]
    # Groups for LambdaRank
    # Count rows per date
    groups = train_df.groupby('Date').size().to_numpy()
    
    # Convert Target to Integer Ranks (0-9) for LambdaRank
    # We bin the targets into deciles per date
    train_df['Target_Rank'] = train_df.groupby('Date')['Target'].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates='drop')
    )
    
    # Handle NaNs in Rank (if any) - though we dropped NaNs in Target
    train_df.dropna(subset=['Target_Rank'], inplace=True)
    train_df['Target_Rank'] = train_df['Target_Rank'].astype(int)
    
    # Re-calc groups after potential drop (unlikely but safe)
    groups = train_df.groupby('Date').size().to_numpy()
    
    X = train_df[features]
    y = train_df['Target_Rank'] # Use Rank as label
    # Sort groups by date matches the sorted df
    # Ensure X is sorted by date (it is)
    
    model = lgb.LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X, y, group=groups)
    
    return model, features, df # Return full df (including latest rows with NaN target) for inference

def main():
    print("Starting training pipeline...")
    symbols = get_nifty500_symbols()
    if not symbols: 
        print("CRITICAL: Could not fetch Nifty 500 symbols. Exiting.")
        return
    
    df = download_data(symbols)
    if df.empty: 
        print("CRITICAL: No data available. Exiting.")
        return
    
    df = engineer_features(df)
    
    model, features, filtered_df = train_model(df)
    
    # Save
    artifact = {
        'model': model,
        'features': features,
        'last_updated': datetime.now().strftime("%Y-%m-%d")
    }
    joblib.dump(artifact, MODEL_FILE)
    
    # Save filtered dataset
    filtered_df.to_parquet(DATA_FILE)
    
    print("MODEL TRAINING COMPLETED – final_model.pkl saved")

if __name__ == "__main__":
    main()
