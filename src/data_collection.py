"""
Loading and cleaning SPY minute-bar data from FirstRate Data.
The raw data is 1-minute OHLCV bars — we can still derive useful
microstructure signals from this granularity.
"""

import pandas as pd
import numpy as np
import os
import zipfile
from pathlib import Path


# project paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def extract_zip(zip_path=None):
    """Extract the FirstRate Data zip file into data/raw/."""
    if zip_path is None:
        zip_path = RAW_DATA_DIR / "SPY_FirstRateDatacom.zip"

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(RAW_DATA_DIR)
        print(f"Extracted {len(z.namelist())} files to {RAW_DATA_DIR}")
        return z.namelist()


def load_bars(filepath=None):
    """
    Load the FirstRate Data file.
    Format: datetime, open, high, low, close, volume (no header)
    """
    if filepath is None:
        filepath = RAW_DATA_DIR / "SPY_FirstRateDatacom.txt"

    df = pd.read_csv(
        filepath,
        header=None,
        names=['datetime', 'open', 'high', 'low', 'close', 'volume'],
        parse_dates=['datetime']
    )

    df.sort_values('datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"Loaded {len(df):,} bars")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    return df


def clean_bars(df):
    """
    Basic cleaning:
    - Drop nulls
    - Remove bars with zero/negative prices or volume
    - Keep only regular trading hours (9:30 AM - 4:00 PM ET)
    - Remove obvious bad prints (OHLC sanity checks)
    """
    original_len = len(df)

    df = df.dropna()

    # prices and volume must be positive
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        df = df[df[col] > 0]
    df = df[df['volume'] > 0]

    # OHLC sanity: high >= low, high >= open/close, low <= open/close
    df = df[
        (df['high'] >= df['low']) &
        (df['high'] >= df['open']) &
        (df['high'] >= df['close']) &
        (df['low'] <= df['open']) &
        (df['low'] <= df['close'])
    ]

    # regular trading hours only
    df = df[
        (df['datetime'].dt.time >= pd.Timestamp('09:30').time()) &
        (df['datetime'].dt.time < pd.Timestamp('16:00').time())
    ]

    df = df.reset_index(drop=True)

    removed = original_len - len(df)
    print(f"Cleaned: removed {removed:,} bars ({removed/original_len*100:.1f}%)")
    print(f"Remaining: {len(df):,} bars")

    return df


def add_derived_columns(df):
    """
    Add columns we'll use a lot in analysis.
    Keeps things simple — just the basics.
    """
    df = df.copy()
    df['date'] = df['datetime'].dt.date
    df['time'] = df['datetime'].dt.time
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['bar_range'] = df['high'] - df['low']
    df['bar_range_pct'] = df['bar_range'] / df['close'] * 100
    df['vwap_proxy'] = (df['high'] + df['low'] + df['close']) / 3  # typical price
    df['dollar_volume'] = df['vwap_proxy'] * df['volume']

    return df


def get_daily_summary(df):
    """Quick summary stats by day — useful for picking analysis days."""
    summary = df.groupby('date').agg(
        num_bars=('close', 'count'),
        total_volume=('volume', 'sum'),
        total_dollar_vol=('dollar_volume', 'sum'),
        open_price=('open', 'first'),
        close_price=('close', 'last'),
        high=('high', 'max'),
        low=('low', 'min'),
        avg_bar_volume=('volume', 'mean'),
        avg_range_pct=('bar_range_pct', 'mean'),
    ).reset_index()

    summary['day_range_pct'] = (summary['high'] - summary['low']) / summary['open_price'] * 100
    summary['return_pct'] = (summary['close_price'] - summary['open_price']) / summary['open_price'] * 100

    return summary


if __name__ == "__main__":
    print("Extracting zip file...")
    extract_zip()

    print("\nLoading bars...")
    bars = load_bars()

    print("\nCleaning...")
    bars = clean_bars(bars)

    print("\nAdding derived columns...")
    bars = add_derived_columns(bars)

    print("\nDaily summary:")
    summary = get_daily_summary(bars)
    print(summary.to_string(index=False))

    # save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    bars.to_csv(PROCESSED_DIR / "spy_bars_clean.csv", index=False)
    summary.to_csv(PROCESSED_DIR / "daily_summary.csv", index=False)
    print(f"\nSaved to {PROCESSED_DIR}")
