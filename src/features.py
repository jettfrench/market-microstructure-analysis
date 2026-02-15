"""
Microstructure feature calculations from 1-minute OHLCV bars.

We can't observe the order book directly from bar data, but we can estimate
a lot of the same quantities using well-known estimators from the literature.
"""

import warnings
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Spread estimation
# ---------------------------------------------------------------------------

def corwin_schultz_spread(high, low):
    """
    Estimate bid-ask spread from high-low prices (Corwin & Schultz, 2012).

    The intuition: the high-low range captures both volatility and the spread.
    By comparing single-bar and two-bar ranges, we can separate the two.

    Returns estimated spread as a fraction of price.
    """
    # beta = sum of squared log(H/L) over two consecutive bars
    log_hl = np.log(high / low)
    log_hl_sq = log_hl ** 2

    beta = log_hl_sq.rolling(2).sum()

    # gamma = log(max(H_t, H_t-1) / min(L_t, L_t-1))^2
    high_2 = high.rolling(2).max()
    low_2 = low.rolling(2).min()
    gamma = np.log(high_2 / low_2) ** 2

    # alpha
    denom = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)

    # spread = 2 * (e^alpha - 1) / (1 + e^alpha)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))

    # negative estimates are set to 0 (common in practice)
    spread = spread.clip(lower=0)

    return spread


def roll_spread(returns):
    """
    Roll (1984) implied spread estimator.

    Uses the autocovariance of returns — if bid-ask bounce causes negative
    autocorrelation, we can back out the spread from it.

    Returns estimated spread (in return units).
    """
    cov = returns.rolling(20).apply(
        lambda x: np.cov(x[:-1], x[1:])[0, 1] if len(x) > 1 else np.nan,
        raw=True
    )
    # spread = 2 * sqrt(-cov) when cov < 0
    spread = np.where(cov < 0, 2 * np.sqrt(-cov), 0)
    return pd.Series(spread, index=returns.index)


def effective_spread_proxy(high, low, close):
    """
    Simple spread proxy: (high - low) as a fraction of close.
    Not theoretically rigorous but useful as a quick measure.
    """
    return (high - low) / close


# ---------------------------------------------------------------------------
# Volatility measures
# ---------------------------------------------------------------------------

def parkinson_volatility(high, low, window=20):
    """
    Parkinson (1980) volatility estimator.

    More efficient than close-to-close vol because it uses the full
    intraday range. Assumes no drift and continuous trading.
    """
    log_hl = np.log(high / low)
    factor = 1 / (4 * np.log(2))
    return np.sqrt(factor * (log_hl ** 2).rolling(window).mean())


def realized_volatility(returns, window=20):
    """Simple realized volatility — rolling std of returns."""
    return returns.rolling(window).std()


def garman_klass_volatility(open_price, high, low, close, window=20):
    """
    Garman-Klass (1980) volatility estimator.

    Uses OHLC data for a more efficient estimate than Parkinson.
    """
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open_price) ** 2

    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return np.sqrt(gk.rolling(window).mean())


# ---------------------------------------------------------------------------
# Order flow and price impact
# ---------------------------------------------------------------------------

def order_flow_imbalance(open_price, close, volume):
    """
    Estimate order flow imbalance from bar data.

    Idea: if close > open, net buying pressure dominated that bar.
    We sign the volume accordingly and compute a rolling imbalance ratio.

    Returns values between -1 (all selling) and +1 (all buying).
    """
    # sign each bar's volume by price direction
    signed_volume = volume * np.sign(close - open_price)

    # handle flat bars (close == open) — assign 0
    signed_volume = signed_volume.fillna(0)

    # rolling imbalance over 20 bars (~20 minutes)
    buy_vol = signed_volume.clip(lower=0).rolling(20).sum()
    sell_vol = (-signed_volume.clip(upper=0)).rolling(20).sum()

    total = buy_vol + sell_vol
    ofi = (buy_vol - sell_vol) / total.replace(0, np.nan)

    return ofi


def kyle_lambda(returns, volume, window=60):
    """
    Kyle's lambda — price impact per unit of signed volume.

    Estimated as the slope of returns on signed volume flow.
    Higher lambda = less liquid market.

    Window of 60 bars = 1 hour.
    """
    # approximate signed volume: volume * sign(return)
    signed_vol = volume * np.sign(returns)

    def _regress(chunk):
        x = chunk['signed_vol'].values
        y = chunk['returns'].values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 10:
            return np.nan
        x, y = x[mask], y[mask]
        if np.std(x) == 0:
            return np.nan
        slope = np.cov(x, y)[0, 1] / np.var(x)
        return slope

    df_temp = pd.DataFrame({'returns': returns, 'signed_vol': signed_vol})
    lam = df_temp.rolling(window).apply(
        lambda x: np.nan, raw=True  # placeholder
    )

    # rolling regression manually (faster than apply with DataFrame)
    result = pd.Series(np.nan, index=returns.index)
    ret_vals = returns.values
    svol_vals = signed_vol.values

    for i in range(window, len(ret_vals)):
        y = ret_vals[i-window:i]
        x = svol_vals[i-window:i]
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 10:
            continue
        x_m, y_m = x[mask], y[mask]
        var_x = np.var(x_m)
        if var_x == 0:
            continue
        result.iloc[i] = np.cov(x_m, y_m)[0, 1] / var_x

    return result


def amihud_illiquidity(returns, dollar_volume, window=20):
    """
    Amihud (2002) illiquidity ratio.

    |return| / dollar_volume — how much does price move per dollar traded.
    Higher = less liquid.
    """
    illiq = returns.abs() / dollar_volume.replace(0, np.nan)
    return illiq.rolling(window).mean()


# ---------------------------------------------------------------------------
# VWAP and volume analysis
# ---------------------------------------------------------------------------

def compute_vwap(df):
    """
    Cumulative intraday VWAP using typical price.
    Resets each day.
    """
    df = df.copy()
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['cum_tp_vol'] = (typical_price * df['volume']).groupby(df['date']).cumsum()
    df['cum_vol'] = df['volume'].groupby(df['date']).cumsum()
    df['vwap'] = df['cum_tp_vol'] / df['cum_vol']
    return df['vwap']


def volume_participation_rate(volume, window=20):
    """
    Rolling share of total daily volume.
    Useful for understanding when liquidity concentrates.
    """
    daily_total = volume.groupby(volume.index // 390).transform('sum')
    return volume / daily_total.replace(0, np.nan)


def relative_volume(volume, window=20):
    """
    Volume relative to its rolling average.
    >1 means unusually high volume for that time.
    """
    return volume / volume.rolling(window).mean()


# ---------------------------------------------------------------------------
# Return dynamics
# ---------------------------------------------------------------------------

def return_autocorrelation(returns, window=20):
    """
    Rolling first-order autocorrelation of returns.

    Negative = mean reversion (common in liquid markets due to bid-ask bounce).
    Positive = momentum / trending.
    """
    return returns.rolling(window).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan,
        raw=False
    )


def trade_intensity(volume, window=5):
    """
    Rolling average volume — proxy for how actively the market is trading.
    Spikes often precede or accompany big moves.
    """
    return volume.rolling(window).mean()


# ---------------------------------------------------------------------------
# Putting it all together
# ---------------------------------------------------------------------------

def compute_all_features(df):
    """
    Compute all microstructure features and add them to the dataframe.
    Expects a cleaned dataframe with derived columns already added.
    """
    df = df.copy()

    # suppress sqrt warnings from volatile periods (produces NaN, which we handle)
    warnings.filterwarnings('ignore', 'invalid value encountered in sqrt', RuntimeWarning)

    # spread estimates
    df['cs_spread'] = corwin_schultz_spread(df['high'], df['low'])
    df['roll_spread'] = roll_spread(df['returns'])
    df['hl_spread'] = effective_spread_proxy(df['high'], df['low'], df['close'])

    # volatility
    df['parkinson_vol'] = parkinson_volatility(df['high'], df['low'])
    df['realized_vol'] = realized_volatility(df['returns'])
    df['gk_vol'] = garman_klass_volatility(df['open'], df['high'], df['low'], df['close'])

    # order flow
    df['ofi'] = order_flow_imbalance(df['open'], df['close'], df['volume'])

    # price impact
    df['amihud'] = amihud_illiquidity(df['returns'], df['dollar_volume'])

    # vwap
    df['vwap'] = compute_vwap(df)
    df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']

    # volume measures
    df['rel_volume'] = relative_volume(df['volume'])
    df['trade_intensity'] = trade_intensity(df['volume'])

    # return dynamics
    df['ret_autocorr'] = return_autocorrelation(df['returns'])

    print(f"Computed {sum(1 for c in df.columns if c not in ['datetime','open','high','low','close','volume','date','time','returns','log_returns','bar_range','bar_range_pct','vwap_proxy','dollar_volume'])} microstructure features")

    return df


if __name__ == "__main__":
    from data_collection import load_bars, clean_bars, add_derived_columns

    bars = load_bars()
    bars = clean_bars(bars)
    bars = add_derived_columns(bars)

    print("\nComputing features...")
    bars = compute_all_features(bars)

    print("\nFeature summary:")
    feature_cols = ['cs_spread', 'roll_spread', 'hl_spread', 'parkinson_vol',
                    'realized_vol', 'gk_vol', 'ofi', 'amihud', 'vwap_deviation',
                    'rel_volume', 'ret_autocorr']
    print(bars[feature_cols].describe().round(6).to_string())
