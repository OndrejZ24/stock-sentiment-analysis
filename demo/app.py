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
    """
    Get daily aggregated sentiment history from SENTIMENT_RESULTS table.
    Aggregates FINAL_SENTIMENT_SCORE by day for the specified ticker.
    """
    try:
        conn = get_oracle_connection()
        cursor = conn.cursor()

        # Calculate start date
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=days)

        # Convert to Unix timestamps (seconds)
        start_timestamp = int(start_dt.timestamp())
        end_timestamp = int(end_dt.timestamp()) + 86400  # Add one day

        query = """
            SELECT
                TRUNC(TO_DATE('1970-01-01', 'YYYY-MM-DD') + CREATED_UTC / 86400) as sentiment_date,
                AVG(FINAL_SENTIMENT_SCORE) as avg_sentiment,
                COUNT(*) as mention_count,
                STDDEV(FINAL_SENTIMENT_SCORE) as sentiment_stddev
            FROM SENTIMENT_RESULTS
            WHERE TICKER = :ticker
            AND CREATED_UTC BETWEEN :start_ts AND :end_ts
            GROUP BY TRUNC(TO_DATE('1970-01-01', 'YYYY-MM-DD') + CREATED_UTC / 86400)
            ORDER BY sentiment_date
        """

        cursor.execute(query, {
            'ticker': ticker,
            'start_ts': start_timestamp,
            'end_ts': end_timestamp
        })

        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        # Convert to dataframe
        if rows:
            df = pd.DataFrame(rows, columns=['date', 'sentiment_mean', 'mentions', 'sentiment_stddev'])
            df['date'] = pd.to_datetime(df['date'])

            # Calculate rolling window sentiment (7-day moving average)
            df['window_sentiment'] = df['sentiment_mean'].rolling(window=min(7, len(df)), min_periods=1).mean()

            print(f"Fetched {len(df)} days of sentiment data for {ticker} from SENTIMENT_RESULTS")
            print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"  Total mentions: {df['mentions'].sum()}")
            return df
        else:
            print(f"No sentiment data found in SENTIMENT_RESULTS for {ticker} between {start_dt.strftime('%Y-%m-%d')} and {end_date}")
            return pd.DataFrame(columns=['date', 'sentiment_mean', 'mentions', 'window_sentiment'])

    except Exception as e:
        print(f"Error getting sentiment history for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=['date', 'sentiment_mean', 'mentions', 'window_sentiment', 'z_score'])

def get_stock_prices(ticker, end_date, days=30):
    """
    Get stock price history from Yahoo Finance.
    ROBUST FIX: No mock data fallback - only real data is used.
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

        # ROBUST FIX: No mock data fallback - return empty DataFrame if no real data available
        print(f"  ✗ WARNING: All real data methods failed for {ticker}")
        print(f"    Returning empty DataFrame - chart will show 'No data available'")
        return pd.DataFrame()

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
            'window_sentiment': round(df['window_sentiment'].mean(), 3) if not df['window_sentiment'].isna().all() else 0.0,
            'sentiment_volatility': round(df['sentiment_mean'].std(), 3) if len(df) > 1 else 0.0
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
