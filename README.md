# Market Microstructure Analysis

Analyzing SPY intraday data to study market microstructure and evaluate execution strategies.

## Overview

This project uses 1-minute OHLCV bar data to:
- Calculate microstructure features (volatility regimes, volume profiles, price impact)
- Compare execution strategies (VWAP, TWAP, aggressive, passive)
- Visualize intraday trading dynamics across different market conditions

The dataset covers Jan-Apr 2020, which includes both normal market conditions and the COVID-19 crash — useful for studying how microstructure changes under stress.

## Data

SPY 1-minute bars from [FirstRate Data](https://firstratedata.com/) — OHLCV with ~24k bars across 63 trading days.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Project Structure

```
src/
  data_collection.py   - data loading and cleaning
  orderbook.py         - simplified order book reconstruction
  features.py          - microstructure feature calculations
  execution.py         - execution strategy simulation
  visualization.py     - plotting and charts
notebooks/
  01_exploration.ipynb - initial data exploration
  02_analysis.ipynb    - main analysis
```
