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
    """Query FILTERED_SIGNALS table for a specific date - with ALL signal metrics."""
    try:
        conn = get_oracle_connection()
        cursor = conn.cursor()

        # Query ALL signal metrics for transparency
        query = """
            SELECT
                TICKER, SIGNAL_DATE, SIGNAL_TYPE,
                WINDOW_SENTIMENT, WINDOW_MENTIONS,
                BASELINE_MEAN, BASELINE_STD,
                Z_SCORE, SIGNAL_SCORE,
                SENTIMENT_MEAN, TOTAL_UPVOTES
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
                'signal_type': row[2],  # BUY or SELL
                'window_sentiment': float(row[3]) if row[3] is not None else None,
                'window_mentions': float(row[4]) if row[4] is not None else None,
                'baseline_mean': float(row[5]) if row[5] is not None else None,
                'baseline_std': float(row[6]) if row[6] is not None else None,
                'z_score': float(row[7]) if row[7] is not None else None,
                'signal_score': float(row[8]) if row[8] is not None else None,
                'sentiment_mean': float(row[9]) if row[9] is not None else None,
                'total_upvotes': float(row[10]) if row[10] is not None else None
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
    Also loads actual signal metrics from FILTERED_SIGNALS for transparency.
    """
    try:
        conn = get_oracle_connection()
        cursor = conn.cursor()

        # Calculate start date
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=days)

        # Use date strings for comparison (fixes timezone bug)
        start_date_str = start_dt.strftime('%Y-%m-%d')
        end_date_str = end_dt.strftime('%Y-%m-%d')

        # Query sentiment data with proper date comparison
        query = """
            SELECT
                TRUNC(TO_DATE('1970-01-01', 'YYYY-MM-DD') + CREATED_UTC / 86400) as sentiment_date,
                AVG(FINAL_SENTIMENT_SCORE) as avg_sentiment,
                COUNT(*) as mention_count,
                SUM(NORMALIZED_UPVOTES) as total_upvotes,
                STDDEV(FINAL_SENTIMENT_SCORE) as sentiment_stddev
            FROM SENTIMENT_RESULTS
            WHERE TICKER = :ticker
            AND TRUNC(TO_DATE('1970-01-01', 'YYYY-MM-DD') + CREATED_UTC / 86400)
                BETWEEN TO_DATE(:start_date, 'YYYY-MM-DD')
                AND TO_DATE(:end_date, 'YYYY-MM-DD')
            GROUP BY TRUNC(TO_DATE('1970-01-01', 'YYYY-MM-DD') + CREATED_UTC / 86400)
            ORDER BY sentiment_date
        """

        cursor.execute(query, {
            'ticker': ticker,
            'start_date': start_date_str,
            'end_date': end_date_str
        })

        rows = cursor.fetchall()

        # Convert to dataframe
        if rows:
            df = pd.DataFrame(rows, columns=['date', 'sentiment_mean', 'mentions', 'total_upvotes', 'sentiment_stddev'])
            df['date'] = pd.to_datetime(df['date'])

            # Compute 3-day mention-weighted window sentiment for ALL days (matches signal generation logic)
            # This creates the blue line that spans the entire chart
            df['weighted_sentiment'] = df['sentiment_mean'] * df['mentions']

            # 3-day rolling window with mention-based weighting
            roll_mentions = df['mentions'].rolling(window=3, min_periods=1).sum()
            roll_sent_sum = df['weighted_sentiment'].rolling(window=3, min_periods=1).sum()
            df['window_sentiment'] = roll_sent_sum / roll_mentions.replace(0, np.nan)

            # Load signal markers and baseline metrics from FILTERED_SIGNALS
            signal_query = """
                SELECT
                    SIGNAL_DATE, BASELINE_MEAN, BASELINE_STD, Z_SCORE, SIGNAL_TYPE
                FROM FILTERED_SIGNALS
                WHERE TICKER = :ticker
                AND SIGNAL_DATE BETWEEN TO_DATE(:start_date, 'YYYY-MM-DD')
                    AND TO_DATE(:end_date, 'YYYY-MM-DD')
            """

            cursor.execute(signal_query, {
                'ticker': ticker,
                'start_date': start_date_str,
                'end_date': end_date_str
            })

            signal_rows = cursor.fetchall()

            if signal_rows:
                signal_df = pd.DataFrame(signal_rows,
                    columns=['date', 'baseline_mean', 'baseline_std', 'z_score', 'signal_type'])
                signal_df['date'] = pd.to_datetime(signal_df['date'])

                # Merge signal metrics (baseline and z_score for markers only)
                df = df.merge(signal_df, on='date', how='left')
            else:
                # No signals for this ticker in this period
                df['baseline_mean'] = None
                df['baseline_std'] = None
                df['z_score'] = None
                df['signal_type'] = None

            cursor.close()
            conn.close()

            print(f"Fetched {len(df)} days of sentiment data for {ticker}")
            print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"  Total mentions: {df['mentions'].sum()}")
            return df
        else:
            cursor.close()
            conn.close()
            print(f"No sentiment data found for {ticker} between {start_date_str} and {end_date_str}")
            return pd.DataFrame(columns=['date', 'sentiment_mean', 'mentions', 'total_upvotes', 'window_sentiment'])

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
    Create clean two-panel sentiment chart with 3-day window as primary focus.
    Top: Sentiment (3d window dominant, with faded daily points, baseline, threshold bands)
    Bottom: Reddit mentions volume
    """
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    import plotly

    df = get_sentiment_history(ticker, end_date)

    if df.empty:
        return None

    # Sort by date
    df = df.sort_values('date')

    # Compute rolling baseline (30-day) for the entire period
    df['baseline_mean_computed'] = df['window_sentiment'].rolling(window=30, min_periods=15).mean()
    df['baseline_std_computed'] = df['window_sentiment'].rolling(window=30, min_periods=15).std()

    # Create two-panel subplot (70% sentiment, 30% mentions)
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.70, 0.30],
        vertical_spacing=0.08,
        subplot_titles=(f'{ticker} - Sentiment Analysis', 'Reddit Discussion Volume'),
        shared_xaxes=True
    )

    # =============================================================================
    # TOP PANEL: SENTIMENT
    # =============================================================================

    # 1. Threshold bands (subtle background - show signal zones)
    threshold_days = df[(df['baseline_mean_computed'].notna()) & (df['baseline_std_computed'].notna())].copy()
    if not threshold_days.empty:
        # Calculate dynamic threshold
        threshold_days['mention_factor'] = threshold_days['mentions'].rolling(window=3, min_periods=1).sum()
        threshold_days['z_threshold'] = 2.0 / (1 + 0.2 * np.log1p(threshold_days['mention_factor'].clip(lower=1.0)))
        threshold_days['upper_band'] = threshold_days['baseline_mean_computed'] + (threshold_days['z_threshold'] * threshold_days['baseline_std_computed'])
        threshold_days['lower_band'] = threshold_days['baseline_mean_computed'] - (threshold_days['z_threshold'] * threshold_days['baseline_std_computed'])

        # Upper threshold band
        fig.add_trace(
            go.Scatter(
                x=threshold_days['date'],
                y=threshold_days['upper_band'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        # Lower threshold band
        fig.add_trace(
            go.Scatter(
                x=threshold_days['date'],
                y=threshold_days['lower_band'],
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(255, 200, 200, 0.12)',  # Very subtle red tint
                line=dict(width=0),
                name='Signal Zone',
                hovertemplate='<b>%{x|%b %d}</b><br>Threshold Band<extra></extra>'
            ),
            row=1, col=1
        )

    # 2. Baseline (only if meaningful - check if it's not close to zero)
    baseline_days = df[df['baseline_mean_computed'].notna()].copy()
    if not baseline_days.empty:
        baseline_mean = baseline_days['baseline_mean_computed'].mean()
        # Only show baseline if it's significantly different from zero
        if abs(baseline_mean) > 0.05:
            fig.add_trace(
                go.Scatter(
                    x=baseline_days['date'],
                    y=baseline_days['baseline_mean_computed'],
                    mode='lines',
                    name='30d Baseline',
                    line=dict(color='rgba(150, 150, 150, 0.4)', width=1, dash='dash'),
                    hovertemplate='<b>%{x|%b %d}</b><br>Baseline: %{y:.3f}<extra></extra>'
                ),
                row=1, col=1
            )

    # 3. Daily sentiment line (matches stock price "Close Price" style)
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['sentiment_mean'],
            mode='lines',
            name='Daily Sentiment',
            line=dict(color='#2196F3', width=2),  # Blue, solid - matches Close Price
            hovertemplate='<b>%{x|%b %d}</b><br>Daily: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )

    # 4. 3-Day Window Sentiment (matches stock price "3-Day Avg" style)
    window_days = df[df['window_sentiment'].notna()].copy()
    if not window_days.empty:
        fig.add_trace(
            go.Scatter(
                x=window_days['date'],
                y=window_days['window_sentiment'],
                mode='lines',
                name='3-Day Window',
                line=dict(color='#FF6B6B', width=1.5, dash='dot'),  # Coral/red, dotted - matches 3-Day Avg
                hovertemplate='<b>%{x|%b %d}</b><br>3d Window: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )

    # 5. Signal markers (BUY/SELL triangles on the 3-day line)
    buy_days = df[(df['z_score'].notna()) & (df['signal_type'] == 'BUY')].copy()
    sell_days = df[(df['z_score'].notna()) & (df['signal_type'] == 'SELL')].copy()

    if not buy_days.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_days['date'],
                y=buy_days['window_sentiment'],
                mode='markers',
                name='BUY',
                marker=dict(size=16, color='#28a745', symbol='triangle-up',
                           line=dict(width=2, color='white')),
                text=buy_days['z_score'],
                hovertemplate='<b>%{x|%b %d}</b><br>BUY<br>Sentiment: %{y:.3f}<br>Z-Score: %{text:.2f}<extra></extra>'
            ),
            row=1, col=1
        )

    if not sell_days.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_days['date'],
                y=sell_days['window_sentiment'],
                mode='markers',
                name='SELL',
                marker=dict(size=16, color='#dc3545', symbol='triangle-down',
                           line=dict(width=2, color='white')),
                text=sell_days['z_score'],
                hovertemplate='<b>%{x|%b %d}</b><br>SELL<br>Sentiment: %{y:.3f}<br>Z-Score: %{text:.2f}<extra></extra>'
            ),
            row=1, col=1
        )

    # Zero reference line
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(100, 100, 100, 0.3)", line_width=1, row=1, col=1)

    # =============================================================================
    # BOTTOM PANEL: REDDIT MENTIONS
    # =============================================================================

    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['mentions'],
            name='Mentions',
            marker=dict(color='rgba(100, 150, 200, 0.6)'),
            hovertemplate='<b>%{x|%b %d}</b><br>Mentions: %{y}<extra></extra>'
        ),
        row=2, col=1
    )

    # =============================================================================
    # LAYOUT & STYLING
    # =============================================================================

    fig.update_layout(
        height=550,  # Taller for two panels
        hovermode='x unified',
        plot_bgcolor='white',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0,
            bgcolor='rgba(255, 255, 255, 0.9)'
        ),
        margin=dict(t=80, b=40, l=60, r=40)
    )

    # Update x-axes
    fig.update_xaxes(
        showgrid=True,
        gridcolor='rgba(220, 220, 220, 0.4)',
        row=1, col=1
    )
    fig.update_xaxes(
        title_text='Date',
        showgrid=True,
        gridcolor='rgba(220, 220, 220, 0.4)',
        row=2, col=1
    )

    # Update y-axes
    fig.update_yaxes(
        title_text='Sentiment',
        showgrid=True,
        gridcolor='rgba(220, 220, 220, 0.3)',
        zeroline=True,
        zerolinecolor='rgba(100, 100, 100, 0.3)',
        zerolinewidth=1,
        row=1, col=1
    )
    fig.update_yaxes(
        title_text='Mentions',
        showgrid=True,
        gridcolor='rgba(220, 220, 220, 0.3)',
        row=2, col=1
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
