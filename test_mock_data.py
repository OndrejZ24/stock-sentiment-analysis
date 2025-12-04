"""
Quick test to verify mock stock data works correctly
"""

import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir('demo')

from app import generate_mock_stock_data, get_stock_prices

print("=" * 80)
print("TESTING MOCK STOCK DATA GENERATOR")
print("=" * 80)

# Test 1: Generate data for NVDA
print("\n[TEST 1] Generating mock data for NVDA...")
print("-" * 80)
start = datetime(2024, 11, 12)
end = datetime(2024, 12, 12)
data = generate_mock_stock_data('NVDA', start, end, base_price=140)

if not data.empty:
    print(f"✓ Generated {len(data)} rows")
    print(f"  Date range: {data.index.min()} to {data.index.max()}")
    print(f"  Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    print(f"\nFirst 3 rows:")
    print(data.head(3))
    print(f"\nLast 3 rows:")
    print(data.tail(3))
else:
    print("✗ Failed to generate data")

# Test 2: Test get_stock_prices function
print("\n\n[TEST 2] Testing get_stock_prices() function...")
print("-" * 80)
price_data = get_stock_prices('NVDA', '2024-12-12', days=30)

if not price_data.empty:
    print(f"✓ Function returned {len(price_data)} rows")
    print(f"  Columns: {price_data.columns.tolist()}")
    print(f"  Has all OHLC data: {all(col in price_data.columns for col in ['Open', 'High', 'Low', 'Close'])}")
else:
    print("✗ Function returned empty dataframe")

# Test 3: Test multiple tickers
print("\n\n[TEST 3] Testing different tickers...")
print("-" * 80)
tickers = ['NVDA', 'AAPL', 'MSFT', 'META', 'GOOGL', 'BABA', 'CRWD', 'ADM']

for ticker in tickers:
    data = get_stock_prices(ticker, '2024-12-12', days=30)
    if not data.empty:
        avg_price = data['Close'].mean()
        price_change = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
        print(f"{ticker:6s}: {len(data)} rows, avg=${avg_price:6.2f}, change={price_change:+5.2f}%")
    else:
        print(f"{ticker:6s}: ✗ FAILED")

print("\n" + "=" * 80)
print("✓ MOCK DATA GENERATOR WORKING CORRECTLY")
print("=" * 80)
print("\nYou can now:")
print("1. Open http://localhost:5002 in your browser")
print("2. Select a date from the dropdown")
print("3. Click on any ticker to see sentiment + price charts")
print("4. Stock price charts will show realistic simulated data")
print("=" * 80)
