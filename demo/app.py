from flask import Flask, render_template, jsonify
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import time
import requests

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities from parent directory
from utils import get_oracle_connection

app = Flask(__name__)

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
    Get stock price history from Yahoo Finance using direct API calls.
    This bypasses yfinance library to avoid rate limiting issues.
    """
    import requests
    import json

    try:
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=days)

        # Convert to Unix timestamps
        period1 = int(start_dt.timestamp())
        period2 = int(end_dt.timestamp())

        print(f"Fetching stock data for {ticker} from {start_dt.strftime('%Y-%m-%d')} to {end_date}")

        # Add a small delay to avoid rate limiting
        time.sleep(1)

        # Yahoo Finance v8 API endpoint - try query2 as alternative
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {
            'period1': period1,
            'period2': period2,
            'interval': '1d',
            'includePrePost': 'false',
            'events': 'div,splits'
        }

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

        # Direct API call with timeout
        print(f"  Making direct API call to Yahoo Finance...")
        response = requests.get(url, params=params, headers=headers, timeout=15)

        if response.status_code == 200:
            data = response.json()

            # Extract price data from API response
            result = data.get('chart', {}).get('result', [])
            if result:
                quote = result[0]
                timestamps = quote.get('timestamp', [])
                indicators = quote.get('indicators', {}).get('quote', [{}])[0]

                # Build DataFrame
                df = pd.DataFrame({
                    'Date': pd.to_datetime(timestamps, unit='s'),
                    'Open': indicators.get('open', []),
                    'High': indicators.get('high', []),
                    'Low': indicators.get('low', []),
                    'Close': indicators.get('close', []),
                    'Volume': indicators.get('volume', [])
                })

                # Remove rows with missing data
                df = df.dropna(subset=['Close'])
                df = df.set_index('Date')

                if not df.empty:
                    print(f"  ✓ SUCCESS: Fetched {len(df)} price records")
                    print(f"    Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
                    return df
                else:
                    print(f"  ✗ No valid price data found")
            else:
                print(f"  ✗ Empty result from API")
        elif response.status_code == 429:
            print(f"  ✗ Rate limited (429) - too many requests. Try again in a few minutes.")
        else:
            print(f"  ✗ API returned status code {response.status_code}")
            print(f"    Response: {response.text[:200]}")

        print(f"  ✗ WARNING: Could not fetch data for {ticker}")
        return pd.DataFrame()

    except Exception as e:
        print(f"ERROR in get_stock_prices for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def create_sentiment_chart(ticker, end_date):
    """
    Create sentiment chart with mentions bars and sentiment lines (dual y-axis).
    Matches the original design.
    """
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    import plotly

    df = get_sentiment_history(ticker, end_date)

    if df.empty:
        return None

    # Sort by date
    df = df.sort_values('date')

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. Reddit Mentions bars (background, on secondary y-axis)
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['mentions'],
            name='Reddit Mentions',
            marker=dict(color='rgba(255, 167, 38, 0.5)'),
            hovertemplate='<b>%{x|%b %d}</b><br>Mentions: %{y}<extra></extra>',
            yaxis='y2'
        ),
        secondary_y=True
    )

    # 2. Sentiment Mean line (solid blue)
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['sentiment_mean'],
            mode='lines',
            name='Sentiment Mean',
            line=dict(color='#2196F3', width=2),
            hovertemplate='<b>%{x|%b %d}</b><br>Sentiment: %{y:.3f}<extra></extra>'
        ),
        secondary_y=False
    )

    # 3. Window Sentiment line (dotted, where available)
    signal_days = df[df['window_sentiment'].notna()].copy()
    if not signal_days.empty:
        fig.add_trace(
            go.Scatter(
                x=signal_days['date'],
                y=signal_days['window_sentiment'],
                mode='lines',
                name='Window Sentiment',
                line=dict(color='#90CAF9', width=2, dash='dot'),
                hovertemplate='<b>%{x|%b %d}</b><br>Window: %{y:.3f}<extra></extra>'
            ),
            secondary_y=False
        )

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, secondary_y=False)

    # Update layout
    fig.update_layout(
        title=f'{ticker} - Sentiment Analysis (30 Days)',
        height=350,
        hovermode='x unified',
        plot_bgcolor='white',
        legend=dict(
            orientation='h',
            yanchor='top',
            y=1.15,
            xanchor='left',
            x=0
        ),
        margin=dict(t=60, b=30, l=60, r=60)
    )

    # Update x-axis
    fig.update_xaxes(
        title_text='Date',
        showgrid=True,
        gridcolor='rgba(200,200,200,0.3)'
    )

    # Update y-axes
    fig.update_yaxes(
        title_text='Sentiment Score',
        showgrid=True,
        gridcolor='rgba(200,200,200,0.3)',
        zeroline=True,
        secondary_y=False
    )
    fig.update_yaxes(
        title_text='Reddit Mentions',
        showgrid=False,
        secondary_y=True
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
            'window_sentiment': round(df['window_sentiment'].mean(), 3) if not df['window_sentiment'].isna().all() else None,
            'sentiment_volatility': round(df['sentiment_mean'].std(), 3) if len(df) > 1 else 0.0,
            'total_days': len(df),
            'signal_days': int(df['z_score'].notna().sum()) if 'z_score' in df.columns else 0,
            'buy_signals': int(((df.get('z_score', 0) > 0) & df.get('z_score', pd.Series()).notna()).sum()),
            'sell_signals': int(((df.get('z_score', 0) < 0) & df.get('z_score', pd.Series()).notna()).sum()),
            'avg_z_score': round(df['z_score'].mean(), 2) if 'z_score' in df.columns and not df['z_score'].isna().all() else None
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

def create_price_chart(ticker, end_date):
    """Create stock price chart with candlestick-style visualization."""
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    import plotly

    df = get_stock_prices(ticker, end_date, days=30)

    if df.empty:
        return None

    df = df.reset_index()
    date_col = 'Date' if 'Date' in df.columns else 'index'
    df[date_col] = pd.to_datetime(df[date_col])

    close_col = 'Close'
    volume_col = 'Volume'

    df['pct_change'] = df[close_col].pct_change() * 100
    df['ma_3'] = df[close_col].rolling(window=3).mean()

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('', '')
    )

    fig.add_trace(
        go.Scatter(
            x=df[date_col],
            y=df[close_col],
            mode='lines',
            name='Close Price',
            line=dict(color='#2196F3', width=2),
            hovertemplate='<b>%{x|%b %d}</b><br>Close: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df[date_col],
            y=df['ma_3'],
            mode='lines',
            name='3-Day Avg',
            line=dict(color='#FF6B6B', width=1.5, dash='dot'),
            hovertemplate='<b>%{x|%b %d}</b><br>3D Avg: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    colors = ['#26a69a' if val >= 0 else '#ef5350' for val in df['pct_change'].fillna(0)]
    fig.add_trace(
        go.Bar(
            x=df[date_col],
            y=df[volume_col],
            name='Volume',
            marker=dict(color=colors, opacity=0.3),
            hovertemplate='<b>%{x|%b %d}</b><br>Volume: %{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )

    first_price = float(df[close_col].iloc[0])
    last_price = float(df[close_col].iloc[-1])
    price_change = ((last_price - first_price) / first_price) * 100
    change_str = f"+{price_change:.2f}%" if price_change >= 0 else f"{price_change:.2f}%"
    change_color = '#26a69a' if price_change >= 0 else '#ef5350'
    title_text = f'<b>{ticker}</b> - Stock Price (30 Days) <span style="color:{change_color}">{change_str}</span>'

    price_min = float(df[close_col].min())
    price_max = float(df[close_col].max())
    price_padding = (price_max - price_min) * 0.08
    y_min = price_min - price_padding
    y_max = price_max + price_padding

    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            xanchor='center',
            font=dict(size=18, color='#333'),
            font_family='Arial'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#333'),
        height=350,
        margin=dict(l=60, r=60, t=60, b=30),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0,
            font=dict(size=10)
        ),
        hovermode='x unified',
        bargap=0.1
    )

    fig.update_xaxes(showgrid=False, showticklabels=False, row=1, col=1)
    fig.update_xaxes(
        title_text='Date',
        showgrid=False,
        tickformat='%b %d',
        row=2, col=1
    )
    fig.update_yaxes(
        title='Price ($)',
        showgrid=True,
        gridcolor='rgba(200, 200, 200, 0.3)',
        tickprefix='$',
        tickformat='.2f',
        range=[y_min, y_max],
        row=1, col=1
    )
    fig.update_yaxes(
        title='',
        showticklabels=False,
        showgrid=False,
        row=2, col=1
    )

    return plotly.io.to_json(fig)

@app.route('/chart/price/<ticker>/<date>')
def price_chart(ticker, date):
    """API endpoint to get stock price chart data from Yahoo Finance."""
    try:
        df = get_stock_prices(ticker, date, days=30)

        if df.empty:
            return jsonify({
                'success': False,
                'error': 'No price data available from Yahoo Finance'
            }), 404

        chart_json = create_price_chart(ticker, date)

        if not chart_json:
            return jsonify({
                'success': False,
                'error': 'Failed to create price chart'
            }), 500

        df = df.reset_index()
        date_col = 'Date' if 'Date' in df.columns else 'index'
        close_col = 'Close'

        first_price = float(df[close_col].iloc[0])
        last_price = float(df[close_col].iloc[-1])
        price_change_pct = ((last_price - first_price) / first_price) * 100

        summary = {
            'current_price': last_price,
            'price_change_pct': price_change_pct
        }

        return jsonify({
            'success': True,
            'chart': chart_json,
            'summary': summary
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
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
