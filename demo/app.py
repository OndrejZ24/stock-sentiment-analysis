from flask import Flask, render_template, jsonify
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

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

def create_sentiment_chart(ticker, end_date):
    """Create a Plotly sentiment chart with multiple metrics."""
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    import plotly

    df = get_sentiment_history(ticker, end_date)

    if df.empty:
        return None

    # Create figure with subplots - main chart and additional metrics below
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'<b>{ticker}</b> - Sentiment Trend (30 Days)', 'Z-Score & Baseline'),
        vertical_spacing=0.12,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )

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
            showlegend=True
        ),
        row=1, col=1, secondary_y=False
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
            showlegend=True
        ),
        row=1, col=1, secondary_y=False
    )

    # Add mentions as bar chart on secondary y-axis
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['mentions'],
            name='Reddit Mentions',
            marker=dict(color='rgba(255, 140, 0, 0.4)'),
            legendgroup='mentions',
            showlegend=True
        ),
        row=1, col=1, secondary_y=True
    )

    # Bottom chart - Z-Score
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['z_score'],
            mode='lines+markers',
            name='Z-Score',
            line=dict(color='#e53e3e', width=2),
            marker=dict(size=6, color='#e53e3e'),
            fill='tozeroy',
            fillcolor='rgba(229, 62, 62, 0.1)',
            legendgroup='zscore',
            showlegend=True
        ),
        row=2, col=1
    )

    # Add zero line for z-score
    fig.add_hline(y=0, line=dict(color='#999', width=1, dash='dot'), row=2, col=1)

    # Update layout
    fig.update_xaxes(
        title_text='Date',
        showgrid=True,
        gridcolor='#e0e0e0',
        linecolor='#ccc',
        row=2, col=1
    )

    fig.update_yaxes(
        title_text='Sentiment Score',
        titlefont=dict(color='#667eea', size=12),
        tickfont=dict(color='#667eea'),
        showgrid=True,
        gridcolor='#e0e0e0',
        zeroline=True,
        zerolinecolor='#999',
        row=1, col=1, secondary_y=False
    )

    fig.update_yaxes(
        title_text='Mentions',
        titlefont=dict(color='#ff8c00', size=12),
        tickfont=dict(color='#ff8c00'),
        showgrid=False,
        row=1, col=1, secondary_y=True
    )

    fig.update_yaxes(
        title_text='Z-Score',
        titlefont=dict(color='#e53e3e', size=12),
        tickfont=dict(color='#e53e3e'),
        showgrid=True,
        gridcolor='#e0e0e0',
        zeroline=True,
        zerolinecolor='#999',
        row=2, col=1
    )

    fig.update_layout(
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#333'),
        height=650,
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

        # Calculate summary statistics
        summary = {
            'avg_sentiment': round(df['sentiment_mean'].mean(), 3),
            'total_mentions': int(df['mentions'].sum()),
            'max_z_score': round(df['z_score'].abs().max(), 2),
            'window_sentiment': round(df['window_sentiment'].mean(), 3)
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
