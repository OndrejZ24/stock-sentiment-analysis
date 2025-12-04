"""
Test Yahoo Finance API with different approaches
Including manual requests to diagnose connection issues
"""

import yfinance as yf
import requests
from datetime import datetime, timedelta

print("=" * 80)
print("YAHOO FINANCE CONNECTIVITY TEST")
print("=" * 80)

# Test 1: Check if we can reach Yahoo Finance at all
print("\n[TEST 1] Testing basic connectivity to Yahoo Finance...")
print("-" * 80)
try:
    response = requests.get('https://finance.yahoo.com', timeout=10)
    print(f"✓ Can reach finance.yahoo.com: Status {response.status_code}")
except Exception as e:
    print(f"✗ Cannot reach finance.yahoo.com: {e}")

# Test 2: Try with custom user agent
print("\n[TEST 2] Testing yfinance with custom headers...")
print("-" * 80)

# Update yfinance session with custom headers
import yfinance.scrapers.quote

try:
    # Try a simple info request first
    ticker = yf.Ticker("NVDA")
    print("Attempting to get ticker info...")
    info = ticker.info
    if info:
        print(f"✓ Got ticker info for NVDA")
        print(f"  Name: {info.get('longName', 'N/A')}")
        print(f"  Current Price: ${info.get('currentPrice', 'N/A')}")
    else:
        print("✗ Empty ticker info")
except Exception as e:
    print(f"✗ Error getting ticker info: {e}")

# Test 3: Try upgrading yfinance or using alternative
print("\n[TEST 3] Checking yfinance version...")
print("-" * 80)
print(f"Current yfinance version: {yf.__version__}")
print("Recommended: Update to latest version if below 0.2.40")

# Test 4: Alternative - use pandas_datareader
print("\n[TEST 4] Testing alternative: pandas_datareader...")
print("-" * 80)
try:
    import pandas_datareader as pdr
    print("✓ pandas_datareader is installed")

    # Try to fetch data
    start_date = datetime(2024, 11, 12)
    end_date = datetime(2024, 12, 12)

    print(f"Fetching NVDA data from {start_date.date()} to {end_date.date()}...")
    data = pdr.get_data_yahoo('NVDA', start=start_date, end=end_date)

    if not data.empty:
        print(f"✓ SUCCESS with pandas_datareader!")
        print(f"  Rows: {len(data)}")
        print(f"  Date range: {data.index.min()} to {data.index.max()}")
        print(f"  Last close: ${data['Close'].iloc[-1]:.2f}")
        print("\nFirst few rows:")
        print(data.head())
    else:
        print("✗ Empty result from pandas_datareader")

except ImportError:
    print("✗ pandas_datareader not installed")
    print("  Install with: pip install pandas-datareader")
except Exception as e:
    print(f"✗ Error with pandas_datareader: {e}")

# Test 5: Alternative - use yfinance with retry mechanism
print("\n[TEST 5] Testing yfinance with session and retry...")
print("-" * 80)
try:
    from requests import Session
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    # Create session with retry strategy
    session = Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))

    # Set user agent
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    })

    ticker = yf.Ticker("NVDA", session=session)
    hist = ticker.history(period="1mo")

    if not hist.empty:
        print(f"✓ SUCCESS with custom session!")
        print(f"  Rows: {len(hist)}")
        print(f"  Date range: {hist.index.min()} to {hist.index.max()}")
        print(f"  Last close: ${hist['Close'].iloc[-1]:.2f}")
    else:
        print("✗ Empty result even with custom session")

except Exception as e:
    print(f"✗ Error with custom session: {e}")

# Test 6: Mock data as fallback
print("\n[TEST 6] Creating mock data generator as fallback...")
print("-" * 80)
import pandas as pd
import numpy as np

def generate_mock_stock_data(ticker, start_date, end_date, base_price=150):
    """Generate realistic mock stock data for demonstration."""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    # Filter to weekdays only
    dates = dates[dates.dayofweek < 5]

    n = len(dates)
    # Generate realistic price movement
    returns = np.random.normal(0.001, 0.02, n)  # Small daily returns
    price = base_price * (1 + returns).cumprod()

    # Create OHLC data
    high = price * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = price * (1 - np.abs(np.random.normal(0, 0.01, n)))
    open_price = np.roll(price, 1)
    open_price[0] = base_price

    data = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': price,
        'Volume': np.random.randint(1000000, 10000000, n)
    }, index=dates)

    return data

# Generate sample mock data
start = datetime(2024, 11, 12)
end = datetime(2024, 12, 12)
mock_data = generate_mock_stock_data('NVDA', start, end)

print(f"✓ Generated {len(mock_data)} rows of mock data")
print(f"  Date range: {mock_data.index.min()} to {mock_data.index.max()}")
print("\nSample data:")
print(mock_data.head())

print("\n" + "=" * 80)
print("RECOMMENDATIONS:")
print("=" * 80)
print("""
1. If TEST 4 (pandas_datareader) works:
   → Use pandas_datareader instead of yfinance
   → Install: pip install pandas-datareader

2. If TEST 5 (custom session) works:
   → Implement custom session in app.py

3. If all tests fail (network/firewall blocking Yahoo):
   → Option A: Use mock data generator for demonstration
   → Option B: Use alternative API (Alpha Vantage, IEX Cloud, etc.)
   → Option C: Pre-fetch and cache data in database

4. Check yfinance version and upgrade:
   → Current: """ + yf.__version__ + """
   → Run: pip install --upgrade yfinance
""")
print("=" * 80)
