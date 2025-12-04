"""
Test script for Yahoo Finance API integration
Testing different methods to fetch stock price data
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

# Set up logging to see yfinance details
logging.basicConfig(level=logging.INFO)
yf_logger = logging.getLogger('yfinance')
yf_logger.setLevel(logging.INFO)

print("=" * 80)
print("YAHOO FINANCE API TESTING")
print("=" * 80)

# Test tickers
test_tickers = ['NVDA', 'AAPL', 'MSFT', 'META', 'GOOGL']

# Test date (from your signals)
test_end_date = '2024-12-12'
end_dt = datetime.strptime(test_end_date, '%Y-%m-%d')
start_dt = end_dt - timedelta(days=30)

print(f"\nTest Parameters:")
print(f"  Date Range: {start_dt.strftime('%Y-%m-%d')} to {test_end_date}")
print(f"  Tickers: {', '.join(test_tickers)}")
print("=" * 80)

# Method 1: yf.download() with date range
print("\n\n[METHOD 1] Testing yf.download() with date range...")
print("-" * 80)
for ticker in test_tickers:
    try:
        print(f"\n{ticker}:")
        hist = yf.download(
            ticker,
            start=start_dt,
            end=end_dt + timedelta(days=1),
            progress=False
        )
        if not hist.empty:
            print(f"  ✓ SUCCESS: {len(hist)} rows")
            print(f"    Date range: {hist.index.min()} to {hist.index.max()}")
            print(f"    Columns: {hist.columns.tolist()}")
            print(f"    Last close: ${hist['Close'].iloc[-1]:.2f}")
        else:
            print(f"  ✗ EMPTY result")
    except Exception as e:
        print(f"  ✗ ERROR: {e}")

# Method 2: Ticker().history() with date range
print("\n\n[METHOD 2] Testing Ticker().history() with date range...")
print("-" * 80)
for ticker in test_tickers:
    try:
        print(f"\n{ticker}:")
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_dt, end=end_dt + timedelta(days=1))
        if not hist.empty:
            print(f"  ✓ SUCCESS: {len(hist)} rows")
            print(f"    Date range: {hist.index.min()} to {hist.index.max()}")
            print(f"    Columns: {hist.columns.tolist()}")
            print(f"    Last close: ${hist['Close'].iloc[-1]:.2f}")
        else:
            print(f"  ✗ EMPTY result")
    except Exception as e:
        print(f"  ✗ ERROR: {e}")

# Method 3: Ticker().history() with period parameter
print("\n\n[METHOD 3] Testing Ticker().history() with period='1mo'...")
print("-" * 80)
for ticker in test_tickers:
    try:
        print(f"\n{ticker}:")
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        if not hist.empty:
            print(f"  ✓ SUCCESS: {len(hist)} rows")
            print(f"    Date range: {hist.index.min()} to {hist.index.max()}")
            print(f"    Columns: {hist.columns.tolist()}")
            print(f"    Last close: ${hist['Close'].iloc[-1]:.2f}")
        else:
            print(f"  ✗ EMPTY result")
    except Exception as e:
        print(f"  ✗ ERROR: {e}")

# Method 4: Ticker().history() with period='max' (get all available data)
print("\n\n[METHOD 4] Testing Ticker().history() with period='max' (recent data)...")
print("-" * 80)
for ticker in test_tickers:
    try:
        print(f"\n{ticker}:")
        stock = yf.Ticker(ticker)
        hist = stock.history(period="max")
        if not hist.empty:
            print(f"  ✓ SUCCESS: {len(hist)} rows")
            print(f"    Date range: {hist.index.min()} to {hist.index.max()}")
            print(f"    Columns: {hist.columns.tolist()}")

            # Filter to our desired date range
            hist_filtered = hist[(hist.index >= start_dt) & (hist.index <= end_dt + timedelta(days=1))]
            print(f"    Filtered to target dates: {len(hist_filtered)} rows")
            if len(hist_filtered) > 0:
                print(f"    Last close in range: ${hist_filtered['Close'].iloc[-1]:.2f}")
        else:
            print(f"  ✗ EMPTY result")
    except Exception as e:
        print(f"  ✗ ERROR: {e}")

# Method 5: Using specific interval and auto_adjust
print("\n\n[METHOD 5] Testing with interval='1d' and auto_adjust=True...")
print("-" * 80)
for ticker in test_tickers:
    try:
        print(f"\n{ticker}:")
        stock = yf.Ticker(ticker)
        hist = stock.history(
            start=start_dt,
            end=end_dt + timedelta(days=1),
            interval="1d",
            auto_adjust=True
        )
        if not hist.empty:
            print(f"  ✓ SUCCESS: {len(hist)} rows")
            print(f"    Date range: {hist.index.min()} to {hist.index.max()}")
            print(f"    Columns: {hist.columns.tolist()}")
            print(f"    Last close: ${hist['Close'].iloc[-1]:.2f}")
        else:
            print(f"  ✗ EMPTY result")
    except Exception as e:
        print(f"  ✗ ERROR: {e}")

# Test with current date (to see if recent data works)
print("\n\n[METHOD 6] Testing with RECENT dates (last 30 days from today)...")
print("-" * 80)
today = datetime.now()
recent_start = today - timedelta(days=30)
print(f"Date range: {recent_start.strftime('%Y-%m-%d')} to {today.strftime('%Y-%m-%d')}")

for ticker in test_tickers[:2]:  # Just test first 2 tickers
    try:
        print(f"\n{ticker}:")
        hist = yf.download(
            ticker,
            start=recent_start,
            end=today,
            progress=False
        )
        if not hist.empty:
            print(f"  ✓ SUCCESS: {len(hist)} rows")
            print(f"    Date range: {hist.index.min()} to {hist.index.max()}")
            print(f"    Last close: ${hist['Close'].iloc[-1]:.2f}")
        else:
            print(f"  ✗ EMPTY result")
    except Exception as e:
        print(f"  ✗ ERROR: {e}")

print("\n" + "=" * 80)
print("TESTING COMPLETE")
print("=" * 80)
