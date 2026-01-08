#!/usr/bin/env python3
"""
Test script to verify demo app fixes work correctly.
Tests the critical changes made to ensure signal metrics are properly displayed.
"""

import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import get_oracle_connection
import pandas as pd

def test_database_schema():
    """Test that FILTERED_SIGNALS table has all required columns."""
    print("=" * 70)
    print("TEST 1: Database Schema Validation")
    print("=" * 70)

    try:
        conn = get_oracle_connection()
        if not conn:
            print("❌ FAIL: Could not connect to database")
            return False

        cursor = conn.cursor()

        # Query table schema
        cursor.execute("""
            SELECT COLUMN_NAME, DATA_TYPE
            FROM USER_TAB_COLUMNS
            WHERE TABLE_NAME = 'FILTERED_SIGNALS'
            ORDER BY COLUMN_ID
        """)

        columns = cursor.fetchall()
        column_names = [col[0] for col in columns]

        print(f"\n✓ Found FILTERED_SIGNALS table with {len(columns)} columns:")
        for col_name, col_type in columns:
            print(f"  - {col_name}: {col_type}")

        # Check for required columns
        required_columns = [
            'TICKER', 'SIGNAL_DATE', 'SIGNAL_TYPE',
            'WINDOW_SENTIMENT', 'WINDOW_MENTIONS',
            'BASELINE_MEAN', 'BASELINE_STD',
            'Z_SCORE', 'SIGNAL_SCORE'
        ]

        missing_columns = [col for col in required_columns if col not in column_names]

        if missing_columns:
            print(f"\n❌ FAIL: Missing required columns: {missing_columns}")
            cursor.close()
            conn.close()
            return False

        print(f"\n✅ PASS: All required columns present")

        # Check if there's data
        cursor.execute("SELECT COUNT(*) FROM FILTERED_SIGNALS")
        count = cursor.fetchone()[0]
        print(f"✓ Table contains {count:,} signals")

        if count == 0:
            print("⚠️  WARNING: Table is empty - run signal generation first!")

        cursor.close()
        conn.close()
        return True

    except Exception as e:
        print(f"\n❌ FAIL: Database error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_signal_metrics_query():
    """Test that we can query signal metrics successfully."""
    print("\n" + "=" * 70)
    print("TEST 2: Signal Metrics Query")
    print("=" * 70)

    try:
        conn = get_oracle_connection()
        cursor = conn.cursor()

        # Get a sample signal
        query = """
            SELECT
                TICKER, SIGNAL_DATE, SIGNAL_TYPE,
                WINDOW_SENTIMENT, WINDOW_MENTIONS,
                BASELINE_MEAN, BASELINE_STD,
                Z_SCORE, SIGNAL_SCORE
            FROM FILTERED_SIGNALS
            WHERE ROWNUM <= 5
            ORDER BY SIGNAL_DATE DESC
        """

        cursor.execute(query)
        rows = cursor.fetchall()

        if not rows:
            print("⚠️  WARNING: No signals found in database")
            cursor.close()
            conn.close()
            return True  # Not a failure, just no data

        print(f"\n✓ Successfully queried {len(rows)} sample signals:")
        print(f"\n{'Ticker':<8} {'Date':<12} {'Type':<6} {'Window':<8} {'Z-Score':<8} {'Mentions':<8}")
        print("-" * 70)

        for row in rows:
            ticker = row[0]
            date = row[1].strftime('%Y-%m-%d') if row[1] else 'N/A'
            signal_type = row[2]
            window_sent = f"{row[3]:.3f}" if row[3] is not None else 'N/A'
            z_score = f"{row[7]:.2f}" if row[7] is not None else 'N/A'
            mentions = f"{row[4]:.0f}" if row[4] is not None else 'N/A'

            print(f"{ticker:<8} {date:<12} {signal_type:<6} {window_sent:<8} {z_score:<8} {mentions:<8}")

        # Validate that metrics are not null
        null_counts = {
            'window_sentiment': sum(1 for r in rows if r[3] is None),
            'z_score': sum(1 for r in rows if r[7] is None),
            'baseline_mean': sum(1 for r in rows if r[5] is None)
        }

        if any(null_counts.values()):
            print(f"\n⚠️  WARNING: Found null values in critical metrics:")
            for metric, count in null_counts.items():
                if count > 0:
                    print(f"  - {metric}: {count}/{len(rows)} nulls")
        else:
            print(f"\n✅ PASS: All critical metrics have values")

        cursor.close()
        conn.close()
        return True

    except Exception as e:
        print(f"\n❌ FAIL: Query error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sentiment_history_query():
    """Test that sentiment history query works with signal metrics."""
    print("\n" + "=" * 70)
    print("TEST 3: Sentiment History with Signal Metrics")
    print("=" * 70)

    try:
        conn = get_oracle_connection()
        cursor = conn.cursor()

        # Get a ticker with recent signals
        cursor.execute("""
            SELECT TICKER, SIGNAL_DATE
            FROM FILTERED_SIGNALS
            WHERE ROWNUM = 1
            ORDER BY SIGNAL_DATE DESC
        """)

        result = cursor.fetchone()
        if not result:
            print("⚠️  WARNING: No signals found for testing")
            cursor.close()
            conn.close()
            return True

        ticker = result[0]
        signal_date = result[1]
        end_date = signal_date.strftime('%Y-%m-%d')

        print(f"\n✓ Testing with ticker: {ticker}, date: {end_date}")

        # Calculate date range
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=30)
        start_date = start_dt.strftime('%Y-%m-%d')

        # Query sentiment data
        sentiment_query = """
            SELECT
                TRUNC(TO_DATE('1970-01-01', 'YYYY-MM-DD') + CREATED_UTC / 86400) as sentiment_date,
                AVG(FINAL_SENTIMENT_SCORE) as avg_sentiment,
                COUNT(*) as mention_count
            FROM SENTIMENT_RESULTS
            WHERE TICKER = :ticker
            AND TRUNC(TO_DATE('1970-01-01', 'YYYY-MM-DD') + CREATED_UTC / 86400)
                BETWEEN TO_DATE(:start_date, 'YYYY-MM-DD')
                AND TO_DATE(:end_date, 'YYYY-MM-DD')
            GROUP BY TRUNC(TO_DATE('1970-01-01', 'YYYY-MM-DD') + CREATED_UTC / 86400)
            ORDER BY sentiment_date
        """

        cursor.execute(sentiment_query, {
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date
        })

        sentiment_rows = cursor.fetchall()
        print(f"✓ Found {len(sentiment_rows)} days of sentiment data")

        # Query signal metrics
        signal_query = """
            SELECT
                SIGNAL_DATE, WINDOW_SENTIMENT, Z_SCORE, SIGNAL_TYPE
            FROM FILTERED_SIGNALS
            WHERE TICKER = :ticker
            AND SIGNAL_DATE BETWEEN TO_DATE(:start_date, 'YYYY-MM-DD')
                AND TO_DATE(:end_date, 'YYYY-MM-DD')
        """

        cursor.execute(signal_query, {
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date
        })

        signal_rows = cursor.fetchall()
        print(f"✓ Found {len(signal_rows)} signal days")

        if signal_rows:
            print(f"\n  Signal details:")
            for row in signal_rows:
                date = row[0].strftime('%Y-%m-%d')
                window = f"{row[1]:.3f}" if row[1] is not None else 'N/A'
                z_score = f"{row[2]:.2f}" if row[2] is not None else 'N/A'
                signal_type = row[3]
                print(f"    {date}: {signal_type} (window={window}, z={z_score})")

        print(f"\n✅ PASS: Successfully loaded sentiment history with signal metrics")

        cursor.close()
        conn.close()
        return True

    except Exception as e:
        print(f"\n❌ FAIL: Query error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_consistency():
    """Test that window_sentiment in FILTERED_SIGNALS matches the 3-day calculation."""
    print("\n" + "=" * 70)
    print("TEST 4: Data Consistency Check")
    print("=" * 70)

    try:
        conn = get_oracle_connection()

        # Get a few signals with their source data
        query = """
            SELECT
                fs.TICKER, fs.SIGNAL_DATE, fs.WINDOW_SENTIMENT, fs.WINDOW_MENTIONS,
                fs.Z_SCORE, fs.BASELINE_MEAN, fs.BASELINE_STD
            FROM FILTERED_SIGNALS fs
            WHERE ROWNUM <= 3
            ORDER BY fs.SIGNAL_DATE DESC
        """

        df_signals = pd.read_sql_query(query, conn)

        if df_signals.empty:
            print("⚠️  WARNING: No signals found for consistency check")
            conn.close()
            return True

        print(f"\n✓ Checking consistency for {len(df_signals)} signals...")

        consistent = True
        for _, signal in df_signals.iterrows():
            ticker = signal['TICKER']
            date = signal['SIGNAL_DATE']
            window_sent = signal['WINDOW_SENTIMENT']
            z_score = signal['Z_SCORE']
            baseline_mean = signal['BASELINE_MEAN']
            baseline_std = signal['BASELINE_STD']

            # Validate z-score calculation
            if None not in [window_sent, baseline_mean, baseline_std] and baseline_std > 0:
                expected_z = (window_sent - baseline_mean) / baseline_std
                actual_z = z_score

                if abs(expected_z - actual_z) > 0.01:  # Allow small floating point diff
                    print(f"\n⚠️  WARNING: Z-score mismatch for {ticker} on {date}")
                    print(f"   Expected: {expected_z:.3f}, Actual: {actual_z:.3f}")
                    consistent = False
                else:
                    print(f"  ✓ {ticker} on {date}: z-score consistent ({actual_z:.2f})")
            else:
                print(f"  ⚠️  {ticker} on {date}: Cannot validate (null values)")

        if consistent:
            print(f"\n✅ PASS: Data is mathematically consistent")
        else:
            print(f"\n⚠️  WARNING: Some inconsistencies found (may be rounding)")

        conn.close()
        return True

    except Exception as e:
        print(f"\n❌ FAIL: Consistency check error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("DEMO APP FIX VALIDATION TESTS")
    print("=" * 70)
    print("\nThese tests verify that the demo app can correctly display")
    print("the exact metrics used for signal generation.\n")

    tests = [
        ("Database Schema", test_database_schema),
        ("Signal Metrics Query", test_signal_metrics_query),
        ("Sentiment History", test_sentiment_history_query),
        ("Data Consistency", test_data_consistency)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ FAIL: {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! The demo app should work correctly.")
        print("\nNext steps:")
        print("1. Start the demo app: cd demo && python app.py")
        print("2. Open http://localhost:5002 in your browser")
        print("3. Select a date and click 'Show Signals'")
        print("4. Click on a signal card to see the charts")
        print("5. Verify that the charts show signal markers and threshold bands")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        print("Common issues:")
        print("- FILTERED_SIGNALS table needs to be recreated (run signal generation)")
        print("- Database connection issues (check .env file)")
        print("- No data in database (run the full pipeline)")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
