from flask import Flask, render_template, jsonify
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import time
import yfinance as yf
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities from parent directory
from utils import get_oracle_connection

app = Flask(__name__)

# Create a persistent session with retry logic for yfinance
def get_yfinance_session():
    """Create a session with retry strategy and proper headers."""
    session = Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))

    # Set user agent to avoid blocking
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    return session

# Global session for all yfinance requests
yf_session = get_yfinance_session()

def get_available_dates():
    """Get list of unique dates available in FILTERED_SIGNALS table."""
    try:
        conn = get_oracle_connection()
        cursor = conn.cursor()

        query = """
            SELECT DISTINCT TRUNC(SIGNAL_DATE) as signal_date
            FROM FILTERED_SIGNALS
            ORDER BY TRUNC(SIGNAL_DATE) DESC
        """

        cursor.execute(query)
        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        # Convert to list of date strings
        dates = [row[0].strftime('%Y-%m-%d') for row in rows]
        return dates

    except Exception as e:
        print(f"Error fetching available dates: {e}")
        return []

def get_signals_for_date(selected_date):
    """Query FILTERED_SIGNALS table for a specific date - only TICKER, SIGNAL_DATE, SIGNAL_TYPE."""
    try:
        conn = get_oracle_connection()
        cursor = conn.cursor()

        # Query only the 3 essential columns
        query = """
            SELECT TICKER, SIGNAL_DATE, SIGNAL_TYPE
            FROM FILTERED_SIGNALS
            WHERE TRUNC(SIGNAL_DATE) = TO_DATE(:date_param, 'YYYY-MM-DD')
            ORDER BY TICKER
        """

        cursor.execute(query, {'date_param': selected_date})

        # Fetch results
        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        # Convert to list of dictionaries
        signals = []
        for row in rows:
            signal_date = row[1].strftime('%Y-%m-%d') if row[1] else selected_date

            signal = {
                'ticker': row[0],
                'signal_date': signal_date,
                'signal_type': row[2]  # This is BUY or SELL
            }
            signals.append(signal)

        print(f"Fetched {len(signals)} signals for date {selected_date}")
        return signals

    except Exception as e:
        print(f"Error querying database: {e}")
        import traceback
        traceback.print_exc()
        return []

def get_sentiment_history(ticker, end_date, days=30):
    """Get sentiment history for a ticker leading up to the signal date."""
    try:
        conn = get_oracle_connection()
        cursor = conn.cursor()

        # Calculate start date
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=days)

        query = """
            SELECT SIGNAL_DATE, SENTIMENT_MEAN, WINDOW_MENTIONS,
                   WINDOW_SENTIMENT, Z_SCORE
            FROM FILTERED_SIGNALS
            WHERE TICKER = :ticker
            AND SIGNAL_DATE BETWEEN TO_DATE(:start_date, 'YYYY-MM-DD')
                                AND TO_DATE(:end_date, 'YYYY-MM-DD')
            ORDER BY SIGNAL_DATE
        """

        cursor.execute(query, {
            'ticker': ticker,
            'start_date': start_dt.strftime('%Y-%m-%d'),
            'end_date': end_date
        })

        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        # Convert to dataframe
        if rows:
            df = pd.DataFrame(rows, columns=['date', 'sentiment_mean', 'mentions', 'window_sentiment', 'z_score'])
            df['date'] = pd.to_datetime(df['date'])
            print(f"Fetched {len(df)} sentiment records for {ticker}")
            return df
        else:
            print(f"No sentiment data found for {ticker} between {start_dt.strftime('%Y-%m-%d')} and {end_date}")
            return pd.DataFrame(columns=['date', 'sentiment_mean', 'mentions', 'window_sentiment', 'z_score'])

    except Exception as e:
        print(f"Error getting sentiment history for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=['date', 'sentiment_mean', 'mentions', 'window_sentiment', 'z_score'])

def generate_mock_stock_data(ticker, start_date, end_date, base_price=150):
    """
    Generate realistic mock stock data for demonstration purposes.
    Used when Yahoo Finance API is unavailable due to rate limiting.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    # Filter to weekdays only (trading days)
    dates = dates[dates.dayofweek < 5]

    n = len(dates)
    if n == 0:
        return pd.DataFrame()

    # Set seed based on ticker for consistent data per ticker
    np.random.seed(sum(ord(c) for c in ticker))

    # Generate realistic price movement with slight upward bias
    returns = np.random.normal(0.002, 0.02, n)  # 0.2% daily drift, 2% volatility
    price = base_price * (1 + returns).cumprod()

    # Create OHLC data with realistic intraday variation
    high = price * (1 + np.abs(np.random.normal(0, 0.015, n)))
    low = price * (1 - np.abs(np.random.normal(0, 0.015, n)))
    open_price = np.roll(price, 1)
    open_price[0] = base_price

    # Generate volume with some variation
    base_volume = 5000000
    volume = base_volume + np.random.randint(-2000000, 3000000, n)
    volume = np.maximum(volume, 1000000)  # Ensure positive volume

    data = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': price,
        'Volume': volume
    }, index=dates)

    return data

def get_stock_prices(ticker, end_date, days=30):
    """
    Get stock price history from Yahoo Finance with fallback to mock data.
    Tries multiple methods to fetch real data before falling back to mock.
    """
    try:
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=days)
        end_dt_fetch = end_dt + timedelta(days=1)

        print(f"Fetching REAL stock data for {ticker} from {start_dt.strftime('%Y-%m-%d')} to {end_date}")

        # Method 1: Try yfinance with custom session
        try:
            print(f"  Attempting yfinance with custom session...")
            time.sleep(0.5)  # Rate limiting delay

            stock = yf.Ticker(ticker, session=yf_session)
            hist = stock.history(
                start=start_dt,
                end=end_dt_fetch,
                interval="1d",
                auto_adjust=True,
                timeout=10
            )

            if not hist.empty:
                print(f"  ✓ SUCCESS: Fetched {len(hist)} REAL price records from Yahoo Finance")
                print(f"    Price range: ${hist['Close'].min():.2f} - ${hist['Close'].max():.2f}")
                return hist
            else:
                print(f"  ✗ yfinance returned empty data")
        except Exception as e:
            print(f"  ✗ yfinance failed: {e}")

        # Method 2: Try pandas_datareader (if available)
        try:
            import pandas_datareader as pdr
            print(f"  Attempting pandas_datareader...")
            time.sleep(0.5)

            hist = pdr.get_data_yahoo(ticker, start=start_dt, end=end_dt_fetch)

            if not hist.empty:
                print(f"  ✓ SUCCESS: Fetched {len(hist)} REAL price records from pandas_datareader")
                return hist
            else:
                print(f"  ✗ pandas_datareader returned empty data")
        except ImportError:
            print(f"  ⓘ pandas_datareader not installed, skipping...")
        except Exception as e:
            print(f"  ✗ pandas_datareader failed: {e}")

        # Method 3: Fallback to mock data
        print(f"  ⚠ All real data methods failed, using MOCK data for demonstration")

        base_prices = {
            'NVDA': 140, 'AAPL': 180, 'MSFT': 380, 'GOOGL': 140, 'META': 500,
            'TSLA': 250, 'AMZN': 170, 'AMD': 140, 'NFLX': 600, 'BABA': 90,
            'CRWD': 300, 'ADM': 55, 'SPY': 450
        }
        base_price = base_prices.get(ticker, 150)

        hist = generate_mock_stock_data(ticker, start_dt, end_dt, base_price)

        if hist.empty:
            print(f"  ✗ WARNING: Mock data generation also failed for {ticker}")
            return pd.DataFrame()

        print(f"  Generated {len(hist)} mock price records")
        print(f"    Price range: ${hist['Close'].min():.2f} - ${hist['Close'].max():.2f}")
        return hist

    except Exception as e:
        print(f"ERROR in get_stock_prices for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def create_sentiment_chart(ticker, end_date):
    """Create a Plotly sentiment chart with sentiment metrics only."""
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    import plotly

    df = get_sentiment_history(ticker, end_date)

    if df.empty:
        return None

    # Create figure with single chart
    fig = go.Figure()

    # Main chart - Sentiment Score with gradient fill
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['sentiment_mean'],
            mode='lines+markers',
            name='Sentiment Mean',
            line=dict(color='#667eea', width=3, shape='spline'),
            marker=dict(size=8, color='#667eea', line=dict(color='white', width=2)),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.1)',
            legendgroup='sentiment',
            showlegend=True,
            yaxis='y1'
        )
    )

    # Add window sentiment line
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['window_sentiment'],
            mode='lines',
            name='Window Sentiment',
            line=dict(color='#9f7aea', width=2, dash='dot'),
            legendgroup='sentiment',
            showlegend=True,
            yaxis='y1'
        )
    )

    # Add mentions as bar chart on secondary y-axis
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['mentions'],
            name='Reddit Mentions',
            marker=dict(color='rgba(255, 140, 0, 0.4)'),
            legendgroup='mentions',
            showlegend=True,
            yaxis='y2'
        )
    )

    # Update layout with dual y-axes
    fig.update_layout(
        title=dict(
            text=f'<b>{ticker}</b> - Sentiment Analysis (30 Days)',
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='#333')
        ),
        xaxis=dict(
            title='Date',
            showgrid=True,
            gridcolor='#e0e0e0',
            linecolor='#ccc'
        ),
        yaxis=dict(
            title='Sentiment Score',
            titlefont=dict(color='#667eea', size=12),
            tickfont=dict(color='#667eea'),
            showgrid=True,
            gridcolor='#e0e0e0',
            zeroline=True,
            zerolinecolor='#999'
        ),
        yaxis2=dict(
            title='Reddit Mentions',
            titlefont=dict(color='#ff8c00', size=12),
            tickfont=dict(color='#ff8c00'),
            overlaying='y',
            side='right',
            showgrid=False
        ),
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#333'),
        height=500,
        margin=dict(l=60, r=60, t=80, b=60),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0
        )
    )

    return plotly.io.to_json(fig)


@app.route('/')
def index():
    """Render the main page."""
    available_dates = get_available_dates()
    default_date = available_dates[0] if available_dates else None
    return render_template('index.html', available_dates=available_dates, default_date=default_date)

@app.route('/api/dates')
def api_dates():
    """API endpoint to get available dates."""
    try:
        dates = get_available_dates()
        return jsonify({
            'success': True,
            'dates': dates,
            'count': len(dates)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/signals/<date>')
def get_signals(date):
    """API endpoint to get signals for a specific date."""
    try:
        signals = get_signals_for_date(date)

        # Separate BUY and SELL signals based on signal_type
        buy_signals = [s for s in signals if s.get('signal_type', '').upper() == 'BUY']
        sell_signals = [s for s in signals if s.get('signal_type', '').upper() == 'SELL']

        print(f"Date: {date}, Total signals: {len(signals)}, BUY: {len(buy_signals)}, SELL: {len(sell_signals)}")

        return jsonify({
            'success': True,
            'date': date,
            'count': len(signals),
            'total_buy': len(buy_signals),
            'total_sell': len(sell_signals),
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'signals': signals
        })
    except Exception as e:
        print(f"Error in get_signals: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/chart/sentiment/<ticker>/<date>')
def sentiment_chart(ticker, date):
    """API endpoint to get sentiment chart data with summary statistics."""
    try:
        df = get_sentiment_history(ticker, date)

        if df.empty:
            return jsonify({
                'success': False,
                'error': 'No sentiment data available'
            }), 404

        chart_json = create_sentiment_chart(ticker, date)

        # Calculate summary statistics (sentiment only, no price data)
        summary = {
            'avg_sentiment': round(df['sentiment_mean'].mean(), 3),
            'total_mentions': int(df['mentions'].sum()),
            'window_sentiment': round(df['window_sentiment'].mean(), 3),
            'z_score': round(df['z_score'].mean(), 3) if 'z_score' in df.columns else 0.0
        }

        return jsonify({
            'success': True,
            'chart': chart_json,
            'summary': summary
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("STOCK SENTIMENT ANALYSIS - WEB APPLICATION")
    print("=" * 60)
    print("\nNLP Model Powered Buy/Sell Recommendations")
    print("Based on Reddit Sentiment Analysis")
    print("\n" + "=" * 60)
    print("\nStarting web server...")
    print("Access the application at: http://localhost:5002")
    print("\nPress Ctrl+C to stop the server\n")
    print("=" * 60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5002)
