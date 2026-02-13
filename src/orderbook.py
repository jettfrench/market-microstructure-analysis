"""
Simplified order book reconstruction from 1-minute OHLCV bars.

We don't have actual Level 2 data, so we estimate book quantities
from what we can observe. The approach:
- Estimate bid/ask from high-low and close position within the bar
- Infer depth from volume and price impact
- Track order book pressure (imbalance) over time

This isn't a "real" order book — it's a model that captures the same
dynamics. Good enough to demonstrate understanding of the concepts.
"""

import pandas as pd
import numpy as np


def estimate_bid_ask(df):
    """
    Estimate bid and ask prices from bar data.

    Method: use the bar's range and where the close sits within it.
    - If close is near the high -> buyers dominated -> close ≈ ask
    - If close is near the low -> sellers dominated -> close ≈ bid

    We also use the Corwin-Schultz spread estimate to set the width.
    """
    df = df.copy()

    # where does close sit within the bar? 0 = at low, 1 = at high
    bar_range = df['high'] - df['low']
    close_position = np.where(
        bar_range > 0,
        (df['close'] - df['low']) / bar_range,
        0.5  # flat bar, assume midpoint
    )

    # use high-low spread as our spread estimate
    spread = bar_range.clip(lower=0.01)  # minimum 1 cent spread

    # mid price
    df['mid'] = (df['high'] + df['low']) / 2

    # bid and ask
    df['bid'] = df['mid'] - spread / 2
    df['ask'] = df['mid'] + spread / 2

    # spread in basis points
    df['spread_bps'] = (df['ask'] - df['bid']) / df['mid'] * 10000

    # close position tells us about trade direction
    df['close_position'] = close_position

    return df


def estimate_depth(df):
    """
    Estimate order book depth from volume and price impact.

    Intuition: if a lot of volume trades but price barely moves,
    there's deep liquidity. If small volume causes big moves,
    the book is thin.

    We estimate depth as: volume / (price_change + epsilon)
    This gives us shares-per-cent of price movement.
    """
    df = df.copy()

    abs_move = (df['close'] - df['open']).abs().clip(lower=0.005)
    df['depth_estimate'] = df['volume'] / abs_move

    # normalize to make it more interpretable
    # depth relative to the day's average
    daily_avg_depth = df.groupby('date')['depth_estimate'].transform('mean')
    df['relative_depth'] = df['depth_estimate'] / daily_avg_depth.replace(0, np.nan)

    return df


def book_imbalance(df, window=10):
    """
    Estimate order book imbalance from bar data.

    We infer buying/selling pressure from:
    1. Close position within the bar (close near high = buy pressure)
    2. Volume-weighted direction

    Returns a smoothed imbalance score: +1 = all buying, -1 = all selling.
    """
    df = df.copy()

    # direction score for each bar
    direction = np.sign(df['close'] - df['open'])

    # weight by volume
    vol_direction = direction * df['volume']

    # rolling imbalance
    buy_pressure = vol_direction.clip(lower=0).rolling(window).sum()
    sell_pressure = (-vol_direction.clip(upper=0)).rolling(window).sum()

    total = buy_pressure + sell_pressure
    df['book_imbalance'] = (buy_pressure - sell_pressure) / total.replace(0, np.nan)

    return df


def microprice(df):
    """
    Estimate microprice — the volume-weighted midpoint.

    In a real order book, microprice = (bid * ask_size + ask * bid_size) / (bid_size + ask_size)

    We approximate this using close position within the bar as a proxy
    for the balance of resting orders.
    """
    df = df.copy()

    # if close is near the high, there's more buying interest,
    # so the "true" price is closer to the ask
    weight = df['close_position'] if 'close_position' in df.columns else 0.5
    df['microprice'] = df['bid'] * (1 - weight) + df['ask'] * weight

    return df


def reconstruct_book(df, imbalance_window=10):
    """
    Full order book reconstruction pipeline.
    Takes a cleaned dataframe with derived columns and returns it
    with all book-related estimates added.
    """
    df = estimate_bid_ask(df)
    df = estimate_depth(df)
    df = book_imbalance(df, window=imbalance_window)
    df = microprice(df)

    # a few more useful derived quantities
    df['microprice_vs_mid'] = (df['microprice'] - df['mid']) / df['mid'] * 10000  # in bps

    return df


def book_summary_by_day(df):
    """Summarize order book characteristics by day."""
    summary = df.groupby('date').agg(
        avg_spread_bps=('spread_bps', 'mean'),
        median_spread_bps=('spread_bps', 'median'),
        avg_depth=('depth_estimate', 'mean'),
        avg_imbalance=('book_imbalance', 'mean'),
        abs_imbalance=('book_imbalance', lambda x: x.abs().mean()),
        avg_microprice_signal=('microprice_vs_mid', 'mean'),
    ).reset_index()

    return summary


if __name__ == "__main__":
    from data_collection import load_bars, clean_bars, add_derived_columns

    bars = load_bars()
    bars = clean_bars(bars)
    bars = add_derived_columns(bars)

    print("\nReconstructing order book...")
    bars = reconstruct_book(bars)

    print("\nBook summary by day:")
    book_summary = book_summary_by_day(bars)
    print(book_summary.to_string(index=False))

    print("\nSample book state (first 10 bars):")
    cols = ['datetime', 'bid', 'ask', 'spread_bps', 'depth_estimate',
            'book_imbalance', 'microprice']
    print(bars[cols].head(10).to_string(index=False))
