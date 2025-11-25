"""
SENTIMENT SHOCK & SIGNAL DETECTION
==========================================

Goal:
    - Detect statistically significant sentiment shocks for each ticker
    - Use sliding window aggregation over several days
    - Make thresholds dynamic based on how often the ticker is mentioned
    - Classify shocks into BUY / SELL signals
    - Provide visualization and methodology in code comments

Input:
    - sentiment_hybrid_twitter_llm.csv
      (output from the hybrid pipeline RoBERTa + Qwen)

Key ideas:
    1) Work at PER-TICKER, PER-DAY level (daily aggregated sentiment)
    2) Use SLIDING WINDOW sentiment (e.g. 3-day window)
    3) Compare window sentiment to a HISTORICAL BASELINE (rolling mean & std)
    4) Compute Z-SCORE = (window_sentiment - baseline_mean) / baseline_std
    5) Make the significance threshold LOWER for tickers that are heavily mentioned
       -> more mentions = we require smaller z-score to trigger a signal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import os

# =============================================================================
# CONFIG
# =============================================================================

INPUT_FILE  = "sentiment_hybrid_twitter_llm.csv"
OUTPUT_FILE = "sentiment_signals_advanced.csv"

# Sliding window for "current" sentiment shock
WINDOW_DAYS = 3              # how many days we aggregate over for the shock

# Baseline window for historical average/volatility
BASELINE_DAYS = 30           # rolling window to estimate "normal" sentiment
MIN_BASELINE_DAYS = 15       # minimum history to start evaluating shocks

# Minimum volume of discussion needed in the window
MIN_MENTIONS_IN_WINDOW = 5   # too few mentions = ignore (low information)

# Base z-score threshold (if mentions are "normal")
BASE_Z_THRESHOLD = 1.8       # ~1.8–2.0 is a decent trade-off

# How strongly to adjust threshold by mentions (0 = no adjustment)
MENTION_SENSITIVITY = 0.6    # higher => frequent tickers need much smaller shock

# Optional: require also some absolute change in sentiment, not just z-score
MIN_ABS_SENT_CHANGE = 0.05   # to avoid tiny moves with big z-score on tiny variance

# For visualization: which tickers to plot
PLOT_TICKERS = ["TSLA", "AAPL", "NVDA"]   # change as needed

print("=" * 90)
print("ADVANCED SENTIMENT SHOCK & SIGNAL DETECTION")
print("=" * 90)


# =============================================================================
# 1) LOAD SENTIMENT DATA
# =============================================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Input file {INPUT_FILE} not found.")

df = pd.read_csv(INPUT_FILE)

# Expect columns: ['ticker', 'final_sentiment_score', 'created_utc', ...]
required_cols = {"ticker", "final_sentiment_score", "created_utc"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in input: {missing}")

df["created_utc"] = pd.to_datetime(df["created_utc"])
df = df.sort_values(["ticker", "created_utc"])

print(f"Loaded {len(df):,} per-ticker rows from {INPUT_FILE}")


# =============================================================================
# 2) PREPARE DAILY AGGREGATED SENTIMENT
# =============================================================================
"""
We go from (many rows per ticker & day) -> (one row per ticker & day)
Metrics per day:
    - sentiment_mean: mean of final_sentiment_score
    - sentiment_median: median (robust to outliers)
    - mentions: how many mentions that day
Optionally: we could weight by normalized_upvotes, but here we keep it simple.
"""

df["date"] = df["created_utc"].dt.date

daily = (
    df.groupby(["ticker", "date"])
      .agg(
          sentiment_mean=("final_sentiment_score", "mean"),
          sentiment_median=("final_sentiment_score", "median"),
          mentions=("final_sentiment_score", "count")
      )
      .reset_index()
)

daily["date"] = pd.to_datetime(daily["date"])
daily = daily.sort_values(["ticker", "date"])

print(f"Built daily sentiment dataframe with {len(daily):,} rows")


# =============================================================================
# 3) CORE FUNCTION: COMPUTE SHOCKS & SIGNALS PER TICKER
# =============================================================================

def compute_shocks_for_ticker(df_ticker: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sentiment shocks and trading signals for a single ticker.

    Steps per ticker:
        1) Compute sliding window sentiment over WINDOW_DAYS (window_mean)
        2) Compute rolling baseline mean/std over BASELINE_DAYS (excluding early days)
        3) Compute z-score of window_mean vs. baseline
        4) Adjust z-threshold based on total mentions in the window
        5) Generate BUY / SELL signals if:
               - |z_score| >= dynamic_threshold
               - |sentiment change| >= MIN_ABS_SENT_CHANGE
               - mentions_window >= MIN_MENTIONS_IN_WINDOW
    """

    df_t = df_ticker.copy().sort_values("date")

    # Sliding window over DAYS (not rows) – easier with rolling on daily index
    df_t = df_t.set_index("date")

    # Ensure regular daily index (fill missing days with NaNs for mentions/sentiment)
    full_idx = pd.date_range(df_t.index.min(), df_t.index.max(), freq="D")
    df_t = df_t.reindex(full_idx)
    df_t.index.name = "date"

    # Forward-fill mention = 0 for missing days, sentiment = NaN (no posts)
    df_t["ticker"] = df_t["ticker"].ffill().bfill()  # ticker constant
    df_t["mentions"] = df_t["mentions"].fillna(0)
    # If no mentions, we don't define sentiment_mean that day
    # We'll handle it in the rolling window

    # Sliding window sentiment = weighted average over last WINDOW_DAYS using mentions
    # If there are no mentions in window, window_sentiment = NaN
    roll_mentions = df_t["mentions"].rolling(WINDOW_DAYS, min_periods=1).sum()

    # For sentiment, we use a mention-weighted average in the window
    # To achieve that, we compute sum(score * mentions) / sum(mentions)
    df_t["sent_weighted"] = df_t["sentiment_mean"] * df_t["mentions"]
    roll_sent_sum = df_t["sent_weighted"].rolling(WINDOW_DAYS, min_periods=1).sum()
    roll_sent_mean = roll_sent_sum / roll_mentions.replace(0, np.nan)

    df_t["window_sentiment"] = roll_sent_mean
    df_t["window_mentions"] = roll_mentions

    # Baseline: rolling mean & std of window_sentiment over long horizon
    baseline_mean = df_t["window_sentiment"].rolling(
        BASELINE_DAYS, min_periods=MIN_BASELINE_DAYS
    ).mean()
    baseline_std = df_t["window_sentiment"].rolling(
        BASELINE_DAYS, min_periods=MIN_BASELINE_DAYS
    ).std()

    df_t["baseline_mean"] = baseline_mean
    df_t["baseline_std"] = baseline_std

    # Z-score of current window sentiment vs historical baseline
    df_t["z_score"] = (df_t["window_sentiment"] - df_t["baseline_mean"]) / df_t["baseline_std"]

    # Dynamic z-threshold based on mentions in window
    # Idea: more mentions => smaller threshold
    # Example formula: z_thr = BASE_Z_THRESHOLD / (1 + alpha * log(1 + mentions_window))
    mention_factor = np.log1p(df_t["window_mentions"].clip(lower=1.0))
    df_t["z_threshold"] = BASE_Z_THRESHOLD / (1 + MENTION_SENSITIVITY * mention_factor)

    # Also compute absolute change vs baseline_mean
    df_t["abs_sent_change"] = (df_t["window_sentiment"] - df_t["baseline_mean"]).abs()

    # Initialize signals
    df_t["signal"] = "NONE"
    df_t["signal_direction"] = np.nan  # +1 = BUY, -1 = SELL
    df_t["signal_score"] = np.nan      # strength proxy (z-score)

    # Conditions for signals:
    cond_valid_baseline = df_t["baseline_std"].notna()
    cond_enough_mentions = df_t["window_mentions"] >= MIN_MENTIONS_IN_WINDOW
    cond_abs_move = df_t["abs_sent_change"] >= MIN_ABS_SENT_CHANGE

    # BUY: sentiment significantly above baseline, positive direction
    cond_buy = (
        cond_valid_baseline &
        cond_enough_mentions &
        cond_abs_move &
        (df_t["z_score"] >= df_t["z_threshold"]) &
        (df_t["window_sentiment"] > 0)
    )

    # SELL: sentiment significantly below baseline, negative direction
    cond_sell = (
        cond_valid_baseline &
        cond_enough_mentions &
        cond_abs_move &
        (df_t["z_score"] <= -df_t["z_threshold"]) &
        (df_t["window_sentiment"] < 0)
    )

    df_t.loc[cond_buy, "signal"] = "BUY"
    df_t.loc[cond_buy, "signal_direction"] = 1
    df_t.loc[cond_buy, "signal_score"] = df_t.loc[cond_buy, "z_score"]

    df_t.loc[cond_sell, "signal"] = "SELL"
    df_t.loc[cond_sell, "signal_direction"] = -1
    df_t.loc[cond_sell, "signal_score"] = df_t.loc[cond_sell, "z_score"]

    # Reset index back to columns
    df_t = df_t.reset_index()

    return df_t


# =============================================================================
# 4) APPLY TO ALL TICKERS
# =============================================================================

all_tickers = daily["ticker"].unique()
print(f"Tickers in dataset: {len(all_tickers)}")

results = []

for t in all_tickers:
    df_t = daily[daily["ticker"] == t].copy()
    shocks = compute_shocks_for_ticker(df_t)
    shocks["ticker"] = t
    results.append(shocks)

signals_df = pd.concat(results, ignore_index=True)

# Keep only rows where there is an actual signal
signals_only = signals_df[signals_df["signal"] != "NONE"].copy()

print(f"\nDetected {len(signals_only):,} BUY/SELL signals across all tickers.")
print(signals_only[["ticker", "date", "signal", "signal_score", "window_sentiment", "window_mentions"]].head())


# =============================================================================
# 5) SAVE SIGNALS
# =============================================================================

signals_only.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved advanced signals to: {OUTPUT_FILE}")


# =============================================================================
# 6) VISUALIZATION – SENTIMENT SHOCKS FOR SELECTED TICKERS
# =============================================================================

def plot_ticker_sentiment_shocks(signals_df: pd.DataFrame, ticker: str):
    """
    Plot:
       - window_sentiment over time
       - baseline_mean
       - bands baseline_mean ± z_threshold * baseline_std
       - markers where BUY / SELL signals fired
    """

    df_t = signals_df[signals_df["ticker"] == ticker].copy()
    if df_t.empty:
        print(f"No data for ticker {ticker}")
        return

    df_t = df_t.sort_values("date")

    plt.figure(figsize=(12, 6))
    plt.title(f"Sentiment shocks and signals for {ticker}")
    plt.plot(df_t["date"], df_t["window_sentiment"], label="Window sentiment", linewidth=2)
    plt.plot(df_t["date"], df_t["baseline_mean"], label="Baseline mean", linestyle="--", linewidth=1)

    # Upper and lower dynamic bands
    upper_band = df_t["baseline_mean"] + df_t["z_threshold"] * df_t["baseline_std"]
    lower_band = df_t["baseline_mean"] - df_t["z_threshold"] * df_t["baseline_std"]
    plt.fill_between(df_t["date"], upper_band, lower_band, color="gray", alpha=0.15,
                     label="Dynamic significance band")

    # BUY/SELL markers
    buys = df_t[df_t["signal"] == "BUY"]
    sells = df_t[df_t["signal"] == "SELL"]

    plt.scatter(buys["date"], buys["window_sentiment"], marker="^", s=80,
                label="BUY signal", edgecolor="black")
    plt.scatter(sells["date"], sells["window_sentiment"], marker="v", s=80,
                label="SELL signal", edgecolor="black")

    plt.axhline(0, color="black", linestyle=":", linewidth=1, alpha=0.7)
    plt.xlabel("Date")
    plt.ylabel("Window sentiment")
    plt.legend()
    plt.tight_layout()
    plt.show()


print("\nGenerating example plots for selected tickers...")
for t in PLOT_TICKERS:
    plot_ticker_sentiment_shocks(signals_df, t)

print("\nDone.")
