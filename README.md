# TradePair: Pair Trading Analysis Class

## Overview
`TradePair` is a Python class designed for pair trading analysis between two stock instruments. It fetches historical price data using Yahoo Finance, calculates various trading metrics, and suggests potential trade setups based on statistical data like correlation, price ratio, and density curves.

## Features
- Fetches historical stock data from Yahoo Finance.
- Calculates metrics such as:
  - Daily returns
  - Price spread and differential
  - Price ratio
  - Correlation between stocks
  - Standard deviation and density curve (CDF)
- Suggests trading actions based on statistical thresholds.
- Handles exceptions like mismatched data sizes and missing tickers.

## Usage

### Example Usage
```python
from trade_pair import TradePair

# Create a TradePair object for two stock tickers
pair = TradePair(stockA="AAPL", stockB="MSFT")

# Fetch data and display trading statistics
pair.decision()
```