"""
Execution strategy simulation.

Simulates executing a large order across the day using different strategies,
then compares their performance. We model a hypothetical order to buy 100k
shares of SPY and measure how each strategy performs.

Strategies:
- VWAP: trade proportional to historical volume curve
- TWAP: trade evenly across time
- Aggressive: front-load execution (get it done fast)
- Passive: wait for favorable prices, back-load execution
"""

import pandas as pd
import numpy as np


# total shares we want to execute
DEFAULT_ORDER_SIZE = 100_000


def get_volume_schedule(day_bars):
    """
    Build the intraday volume profile — what fraction of total volume
    trades in each minute. Used by VWAP strategy.
    """
    total_vol = day_bars['volume'].sum()
    schedule = day_bars['volume'] / total_vol
    return schedule


def simulate_vwap(day_bars, order_size=DEFAULT_ORDER_SIZE):
    """
    VWAP strategy: trade proportional to volume.

    Each bar, we trade (bar_volume / total_volume) * order_size.
    This is the benchmark — most institutional orders target VWAP.
    """
    schedule = get_volume_schedule(day_bars)
    shares_per_bar = schedule * order_size

    # execution price is the bar's typical price (VWAP proxy)
    exec_prices = (day_bars['high'] + day_bars['low'] + day_bars['close']) / 3

    fills = pd.DataFrame({
        'datetime': day_bars['datetime'].values,
        'shares': shares_per_bar.values,
        'price': exec_prices.values,
        'cost': (shares_per_bar * exec_prices).values,
    })

    return fills


def simulate_twap(day_bars, order_size=DEFAULT_ORDER_SIZE):
    """
    TWAP strategy: trade equal amounts each bar.

    Simple and predictable. Works well in stable markets but doesn't
    adapt to volume patterns.
    """
    n_bars = len(day_bars)
    shares_per_bar = order_size / n_bars

    exec_prices = (day_bars['high'] + day_bars['low'] + day_bars['close']) / 3

    fills = pd.DataFrame({
        'datetime': day_bars['datetime'].values,
        'shares': shares_per_bar,
        'price': exec_prices.values,
        'cost': (shares_per_bar * exec_prices).values,
    })

    return fills


def simulate_aggressive(day_bars, order_size=DEFAULT_ORDER_SIZE):
    """
    Aggressive strategy: front-load execution.

    Execute 60% in the first hour, 30% in the next two hours,
    10% in the final hour. Minimizes timing risk but takes more
    market impact.
    """
    n_bars = len(day_bars)
    weights = np.zeros(n_bars)

    # figure out bar boundaries for each phase
    first_hour = min(60, n_bars)
    mid_section = min(180, n_bars)

    weights[:first_hour] = 0.60 / first_hour
    weights[first_hour:mid_section] = 0.30 / max(mid_section - first_hour, 1)
    weights[mid_section:] = 0.10 / max(n_bars - mid_section, 1)

    shares_per_bar = weights * order_size
    exec_prices = (day_bars['high'] + day_bars['low'] + day_bars['close']) / 3

    # aggressive execution gets worse fills — add impact cost
    # impact proportional to our share of bar volume
    participation = shares_per_bar / day_bars['volume'].values.clip(min=1)
    impact = participation * 0.001  # 10bps per 1% participation
    adjusted_prices = exec_prices.values * (1 + impact)

    fills = pd.DataFrame({
        'datetime': day_bars['datetime'].values,
        'shares': shares_per_bar,
        'price': adjusted_prices,
        'cost': (shares_per_bar * adjusted_prices),
    })

    return fills


def simulate_passive(day_bars, order_size=DEFAULT_ORDER_SIZE):
    """
    Passive strategy: wait for low prices, back-load to close.

    Only trade when price is below the running VWAP, and increase
    size toward the close. Patient but risks not completing the order.
    """
    n_bars = len(day_bars)
    exec_prices = (day_bars['high'] + day_bars['low'] + day_bars['close']) / 3

    # compute running VWAP
    cum_cost = (exec_prices * day_bars['volume'].values).cumsum()
    cum_vol = day_bars['volume'].values.cumsum()
    running_vwap = cum_cost / cum_vol

    # base schedule: linearly increasing toward close
    base_weight = np.linspace(0.5, 2.0, n_bars)
    base_weight = base_weight / base_weight.sum()

    # only trade when price <= running VWAP (favorable)
    favorable = exec_prices.values <= running_vwap.values
    weights = base_weight * favorable

    # if we haven't filled enough, dump remaining in last 30 mins
    total_scheduled = weights.sum()
    if total_scheduled < 0.9:  # less than 90% would be filled
        shortfall = 1.0 - total_scheduled
        last_30 = max(30, 1)
        weights[-last_30:] += shortfall / last_30

    # normalize
    weights = weights / weights.sum()
    shares_per_bar = weights * order_size

    fills = pd.DataFrame({
        'datetime': day_bars['datetime'].values,
        'shares': shares_per_bar,
        'price': exec_prices.values,
        'cost': (shares_per_bar * exec_prices.values),
    })

    return fills


def evaluate_execution(fills, day_bars):
    """
    Evaluate how well an execution strategy performed.

    Key metrics:
    - Average execution price
    - VWAP benchmark price
    - Slippage vs VWAP (bps)
    - Arrival price slippage (vs open)
    - Implementation shortfall
    """
    total_cost = fills['cost'].sum()
    total_shares = fills['shares'].sum()
    avg_price = total_cost / total_shares

    # benchmark VWAP
    typical_prices = (day_bars['high'] + day_bars['low'] + day_bars['close']) / 3
    benchmark_vwap = (typical_prices * day_bars['volume']).sum() / day_bars['volume'].sum()

    # arrival price (open)
    arrival_price = day_bars['open'].iloc[0]

    # close price
    close_price = day_bars['close'].iloc[-1]

    return {
        'avg_exec_price': avg_price,
        'benchmark_vwap': benchmark_vwap,
        'slippage_vs_vwap_bps': (avg_price - benchmark_vwap) / benchmark_vwap * 10000,
        'slippage_vs_arrival_bps': (avg_price - arrival_price) / arrival_price * 10000,
        'arrival_price': arrival_price,
        'close_price': close_price,
        'total_cost': total_cost,
        'total_shares': total_shares,
    }


def compare_strategies(day_bars, order_size=DEFAULT_ORDER_SIZE):
    """
    Run all four strategies on a single day and compare results.
    """
    strategies = {
        'VWAP': simulate_vwap,
        'TWAP': simulate_twap,
        'Aggressive': simulate_aggressive,
        'Passive': simulate_passive,
    }

    results = {}
    all_fills = {}

    for name, strategy_fn in strategies.items():
        fills = strategy_fn(day_bars, order_size)
        metrics = evaluate_execution(fills, day_bars)
        results[name] = metrics
        all_fills[name] = fills

    results_df = pd.DataFrame(results).T
    results_df.index.name = 'strategy'

    return results_df, all_fills


def compare_across_days(bars, dates, order_size=DEFAULT_ORDER_SIZE):
    """
    Run strategy comparison across multiple days.
    Returns a dict of {date: results_df}.
    """
    all_results = {}

    for date in dates:
        day = bars[bars['date'] == date].reset_index(drop=True)
        if len(day) == 0:
            print(f"  No data for {date}, skipping")
            continue

        results_df, _ = compare_strategies(day, order_size)
        all_results[date] = results_df
        print(f"  {date}: done ({len(day)} bars)")

    return all_results


if __name__ == "__main__":
    import datetime
    from data_collection import load_bars, clean_bars, add_derived_columns

    bars = load_bars()
    bars = clean_bars(bars)
    bars = add_derived_columns(bars)

    test_dates = [
        datetime.date(2020, 1, 16),   # quiet
        datetime.date(2020, 3, 12),   # peak chaos
    ]

    print("\nComparing execution strategies...")
    for date in test_dates:
        day = bars[bars['date'] == date].reset_index(drop=True)
        results, _ = compare_strategies(day)

        print(f"\n{'='*60}")
        print(f"  {date}")
        print(f"{'='*60}")
        print(results[['avg_exec_price', 'benchmark_vwap',
                       'slippage_vs_vwap_bps', 'slippage_vs_arrival_bps']].round(4).to_string())
