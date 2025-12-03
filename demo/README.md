# Stock Sentiment Analysis - Web Application

NLP model powered buy/sell recommendations based on Reddit sentiment.

## Quick Start

1. **Prerequisites**
   - Oracle Database with **FILTERED_SIGNALS** table
   - `.env` file in parent directory with database credentials

2. **Install & Run**
   ```bash
   pip install -r requirements_webapp.txt
   python app.py
   ```

3. **Access**: `http://localhost:5002`

## Database Table

**FILTERED_SIGNALS** columns used:
- `TICKER`, `SIGNAL_DATE`, `SIGNAL_TYPE`
- `SENTIMENT_MEAN`, `WINDOW_SENTIMENT`, `WINDOW_MENTIONS`, `Z_SCORE`

## Troubleshooting

- **Port conflict**: Change port in `app.py` (default: 5002)
- **No data**: Check FILTERED_SIGNALS table has records
- **DB error**: Verify `.env` credentials
