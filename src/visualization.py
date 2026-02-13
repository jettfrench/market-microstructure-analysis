"""
Plotting functions for the analysis.
Keeps the notebooks cleaner by pulling chart logic out here.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


# consistent style
sns.set_style('whitegrid')
COLORS = {
    'normal': 'steelblue',
    'crisis': 'indianred',
    'VWAP': '#2196F3',
    'TWAP': '#4CAF50',
    'Aggressive': '#F44336',
    'Passive': '#FF9800',
}


def plot_price_volume(day_bars, title=None):
    """Price and volume chart for a single day."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})

    if title is None:
        ret = (day_bars['close'].iloc[-1] / day_bars['open'].iloc[0] - 1) * 100
        title = f'SPY — {day_bars["date"].iloc[0]} ({ret:+.2f}%)'

    ax1.plot(day_bars['datetime'], day_bars['close'], color='steelblue', linewidth=1)
    ax1.fill_between(day_bars['datetime'], day_bars['low'], day_bars['high'],
                     alpha=0.15, color='steelblue')
    ax1.set_ylabel('Price ($)')
    ax1.set_title(title)

    ax2.bar(day_bars['datetime'], day_bars['volume'] / 1e3, width=0.0005,
            color='gray', alpha=0.6)
    ax2.set_ylabel('Volume (K)')
    ax2.set_xlabel('Time')

    plt.tight_layout()
    return fig


def plot_spread_comparison(book_summary, analysis_dates=None):
    """Bar chart comparing spreads across days."""
    data = book_summary.copy()
    if analysis_dates is not None:
        data = data[data['date'].isin(analysis_dates)]

    fig, ax = plt.subplots(figsize=(12, 5))

    colors = ['steelblue' if s < 10 else 'indianred'
              for s in data['avg_spread_bps']]

    ax.bar(range(len(data)), data['avg_spread_bps'], color=colors, alpha=0.8)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([str(d) for d in data['date']], rotation=45, ha='right')
    ax.set_ylabel('Average Spread (bps)')
    ax.set_title('Estimated Bid-Ask Spread by Day')
    ax.axhline(y=data['avg_spread_bps'].median(), color='gray',
               linestyle='--', alpha=0.5, label='Median')
    ax.legend()

    plt.tight_layout()
    return fig


def plot_book_dynamics(day_bars, title=None):
    """
    Multi-panel chart showing order book dynamics for a single day:
    price, spread, depth, and imbalance.
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    dt = day_bars['datetime']

    if title is None:
        title = f'Order Book Dynamics — {day_bars["date"].iloc[0]}'

    # price with bid-ask
    axes[0].fill_between(dt, day_bars['bid'], day_bars['ask'],
                         alpha=0.3, color='steelblue', label='Bid-Ask')
    axes[0].plot(dt, day_bars['close'], color='black', linewidth=0.8, label='Close')
    if 'microprice' in day_bars.columns:
        axes[0].plot(dt, day_bars['microprice'], color='orange',
                     linewidth=0.8, alpha=0.7, label='Microprice')
    axes[0].set_ylabel('Price ($)')
    axes[0].set_title(title)
    axes[0].legend(fontsize=9)

    # spread
    axes[1].plot(dt, day_bars['spread_bps'], color='indianred', linewidth=0.8)
    axes[1].set_ylabel('Spread (bps)')
    axes[1].axhline(y=day_bars['spread_bps'].median(), color='gray',
                    linestyle='--', alpha=0.5)

    # depth
    if 'relative_depth' in day_bars.columns:
        axes[2].plot(dt, day_bars['relative_depth'], color='steelblue', linewidth=0.8)
        axes[2].set_ylabel('Relative Depth')
        axes[2].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        axes[2].set_ylim(0, day_bars['relative_depth'].quantile(0.95) * 1.5)

    # imbalance
    if 'book_imbalance' in day_bars.columns:
        axes[3].fill_between(dt, 0, day_bars['book_imbalance'],
                            where=day_bars['book_imbalance'] > 0,
                            color='green', alpha=0.4, label='Buy pressure')
        axes[3].fill_between(dt, 0, day_bars['book_imbalance'],
                            where=day_bars['book_imbalance'] < 0,
                            color='red', alpha=0.4, label='Sell pressure')
        axes[3].set_ylabel('Book Imbalance')
        axes[3].set_ylim(-1, 1)
        axes[3].legend(fontsize=9)

    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    return fig


def plot_execution_comparison(results_df, date):
    """Bar chart comparing execution strategy performance."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    strategies = results_df.index.tolist()
    colors = [COLORS.get(s, 'gray') for s in strategies]

    # slippage vs VWAP
    axes[0].barh(strategies, results_df['slippage_vs_vwap_bps'], color=colors, alpha=0.8)
    axes[0].set_xlabel('Slippage vs VWAP (bps)')
    axes[0].set_title(f'Execution Performance — {date}')
    axes[0].axvline(x=0, color='black', linewidth=0.8)

    # slippage vs arrival
    axes[1].barh(strategies, results_df['slippage_vs_arrival_bps'], color=colors, alpha=0.8)
    axes[1].set_xlabel('Slippage vs Arrival Price (bps)')
    axes[1].set_title('vs Arrival Price')
    axes[1].axvline(x=0, color='black', linewidth=0.8)

    plt.tight_layout()
    return fig


def plot_execution_fills(fills_dict, day_bars, date):
    """Show cumulative execution across strategies overlaid on price."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [1, 1]})

    # price
    ax1.plot(day_bars['datetime'], day_bars['close'], color='black',
             linewidth=0.8, label='Price')
    ax1.set_ylabel('Price ($)')
    ax1.set_title(f'Execution Fill Patterns — {date}')
    ax1.legend()

    # cumulative shares filled
    for name, fills in fills_dict.items():
        color = COLORS.get(name, 'gray')
        cum_shares = fills['shares'].cumsum() / fills['shares'].sum() * 100
        ax2.plot(fills['datetime'], cum_shares, color=color,
                 linewidth=1.5, label=name, alpha=0.8)

    ax2.set_ylabel('% Order Filled')
    ax2.set_xlabel('Time')
    ax2.legend()
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    return fig


def plot_feature_regime_comparison(bars, feature, feature_label=None, window=60):
    """
    Compare a microstructure feature across normal and crisis periods.
    Shows rolling average for clarity.
    """
    if feature_label is None:
        feature_label = feature

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, period in enumerate(['normal', 'crisis']):
        subset = bars[bars['period'] == period] if 'period' in bars.columns else bars
        color = COLORS.get(period, 'steelblue')

        # rolling average to smooth
        values = subset[feature].rolling(window, min_periods=1).mean()

        axes[i].plot(range(len(values)), values, color=color, linewidth=0.8)
        axes[i].set_title(f'{period.capitalize()} Period')
        axes[i].set_ylabel(feature_label)
        axes[i].set_xlabel('Bar')

        # add stats
        raw = subset[feature].dropna()
        stats = f'Mean: {raw.mean():.6f}\nStd: {raw.std():.6f}'
        axes[i].text(0.95, 0.95, stats, transform=axes[i].transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle(f'{feature_label} — Normal vs Crisis', y=1.02)
    plt.tight_layout()
    return fig
