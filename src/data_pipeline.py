"""
Smart Money Concepts (SMC) Data Pipeline for NIFTY 50
Fetches historical data, engineers SMC & price action indicators, and generates leak-free labels.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf


FEATURE_COLUMNS = [
    "liquidity_sweep_bull",
    "liquidity_sweep_bear",
    "bos_bullish",
    "fvg_size",
    "wick_to_body_ratio",
    "rsi_14",
    "atr_pct",
    "volume_spike",
]

SMC_FEATURE_COLUMNS = [
    "liquidity_sweep_bull",
    "liquidity_sweep_bear",
    "bos_bullish",
    "fvg_size",
]

TECHNICAL_FEATURE_COLUMNS = [
    "rsi_14",
    "atr_pct",
    "volume_spike",
]


def load_raw_data(
    ticker: str = "^NSEI",
    start_date: str = "2014-01-01",
    filepath: str = "nifty50.csv",
    raw_save_path: str = "data/nifty50_raw.csv",
) -> pd.DataFrame:
    """
    Check if nifty50.csv exists in root folder; if not, automatically download
    10+ years of daily OHLCV data for ^NSEI using yfinance and save to raw_save_path.
    """
    if os.path.exists(filepath):
        print(f"[Data Pipeline] Loading local data from '{filepath}'...")
        df = pd.read_csv(filepath)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
    else:
        if os.path.exists(raw_save_path):
            print(f"[Data Pipeline] Found cached raw data at '{raw_save_path}'...")
            df = pd.read_csv(raw_save_path)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df.set_index("Date", inplace=True)
        else:
            print(f"[Data Pipeline] '{filepath}' not found. Downloading 10+ years of {ticker} via yfinance...")
            os.makedirs(os.path.dirname(raw_save_path), exist_ok=True)
            df = yf.download(ticker, start=start_date, interval="1d", auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.dropna(how="all", inplace=True)
            df.to_csv(raw_save_path)
            print(f"[Data Pipeline] Saved {len(df)} rows to '{raw_save_path}'.")

    # Clean column names
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Standardize column naming
    rename_map = {c: c.capitalize() for c in df.columns}
    df.rename(columns=rename_map, inplace=True)

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required price column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=required_cols, inplace=True)
    df.sort_index(inplace=True)
    return df


def calculate_smc_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer Smart Money Concepts (SMC) & Price Action features using ONLY past/current candle data.
    Strictly zero future lookahead.
    """
    data = df.copy()

    # 1. liquidity_sweep_bull: 1 if current Low < rolling 10-day min Low (shifted by 1) AND current Close > prior 10-day min Low
    prior_low_10 = data["Low"].shift(1).rolling(window=10).min()
    data["liquidity_sweep_bull"] = (
        (data["Low"] < prior_low_10) & (data["Close"] > prior_low_10)
    ).astype(int)

    # 2. liquidity_sweep_bear: 1 if current High > rolling 10-day max High (shifted by 1) AND current Close < prior 10-day max High
    prior_high_10 = data["High"].shift(1).rolling(window=10).max()
    data["liquidity_sweep_bear"] = (
        (data["High"] > prior_high_10) & (data["Close"] < prior_high_10)
    ).astype(int)

    # 3. bos_bullish (Break of Structure): 1 if Close > rolling 20-day high (shifted by 1), else 0
    prior_high_20 = data["High"].shift(1).rolling(window=20).max()
    data["bos_bullish"] = (data["Close"] > prior_high_20).astype(int)

    # 4. fvg_size (Fair Value Gap normalized): (Low - High.shift(2)) / Close when positive, else 0
    raw_fvg = (data["Low"] - data["High"].shift(2)) / data["Close"]
    data["fvg_size"] = raw_fvg.clip(lower=0.0)

    # 5. wick_to_body_ratio: (High - Low) / (abs(Close - Open) + 1e-5)
    data["wick_to_body_ratio"] = (data["High"] - data["Low"]) / (
        np.abs(data["Close"] - data["Open"]) + 1e-5
    )

    # 6. rsi_14: 14-period Relative Strength Index
    delta = data["Close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    data["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # 7. atr_pct: 14-period Average True Range divided by Close
    prev_close = data["Close"].shift(1)
    tr1 = data["High"] - data["Low"]
    tr2 = (data["High"] - prev_close).abs()
    tr3 = (data["Low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14).mean()
    data["atr_14"] = atr_14
    data["atr_pct"] = atr_14 / data["Close"]

    # 8. volume_spike: Volume / rolling_20_mean_volume
    vol_ma20 = data["Volume"].rolling(window=20).mean()
    data["volume_spike"] = data["Volume"] / (vol_ma20 + 1e-5)

    return data


def calculate_trade_target(
    df: pd.DataFrame,
    lookahead: int = 5,
    rr_ratio: float = 1.5,
    sl_ratio: float = 1.0,
) -> pd.DataFrame:
    """
    Define take_trade = 1 if within next 5 trading days, price reaches +1.5 * ATR profit target
    before hitting -1.0 * ATR stop loss (1:1.5 Risk-Reward). Drop the last 5 rows.
    """
    data = df.copy()
    n = len(data)
    close = data["Close"].values
    high = data["High"].values
    low = data["Low"].values
    atr = data["atr_14"].values

    targets = np.zeros(n, dtype=int)

    for i in range(n - lookahead):
        entry_price = close[i]
        curr_atr = atr[i]
        
        # If ATR is invalid/NaN, skip
        if np.isnan(curr_atr) or curr_atr <= 0:
            targets[i] = 0
            continue

        tp = entry_price + rr_ratio * curr_atr
        sl = entry_price - sl_ratio * curr_atr

        outcome = 0
        for k in range(1, lookahead + 1):
            h_k = high[i + k]
            l_k = low[i + k]

            # Stop loss hit (or both hit on same candle: conservative assumption is stop triggered)
            if l_k <= sl:
                outcome = 0
                break
            # Take profit hit before stop loss
            if h_k >= tp:
                outcome = 1
                break

        targets[i] = outcome

    data["take_trade"] = targets
    # Drop the last lookahead rows where future target cannot be evaluated
    data = data.iloc[:-lookahead].copy()
    return data


def build_dataset(
    filepath: str = "nifty50.csv",
    raw_save_path: str = "data/nifty50_raw.csv",
    save_processed: bool = True,
    processed_path: str = "data/nifty50_smc_features.csv",
) -> pd.DataFrame:
    """
    Full pipeline: load raw, compute features, generate target, drop warm-up NaNs.
    """
    raw_df = load_raw_data(filepath=filepath, raw_save_path=raw_save_path)
    featured_df = calculate_smc_features(raw_df)
    labeled_df = calculate_trade_target(featured_df, lookahead=5, rr_ratio=1.5, sl_ratio=1.0)

    # Drop warm-up rows (first 25 rows have rolling NaNs)
    clean_df = labeled_df.dropna(subset=FEATURE_COLUMNS + ["take_trade"]).copy()

    if save_processed:
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        clean_df.to_csv(processed_path)
        print(f"[Data Pipeline] Processed dataset saved with {len(clean_df)} candles to '{processed_path}'.")

    return clean_df


if __name__ == "__main__":
    df = build_dataset()
    print(f"Dataset shape: {df.shape}")
    print("Class balance:")
    print(df["take_trade"].value_counts(normalize=True))
    print("\nSample features:")
    print(df[FEATURE_COLUMNS + ["take_trade"]].tail())
