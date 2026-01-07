#!/usr/bin/env python3
"""
Stock Sentiment Analysis Utilities
Utility module for stock sentiment analysis - cleaned version with only used functions
"""

import os
import time
import re
import urllib.request
from io import StringIO
import logging
from typing import List, Set, Any, TYPE_CHECKING, TypeAlias
import yfinance as yf
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core imports with fallbacks
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    logger.warning("numpy not available. Some functions may not work.")
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    logger.warning("pandas not available. DataFrame functions will not work.")
    PANDAS_AVAILABLE = False
    pd = None

# alias that doesn't break at runtime when pandas is missing
if TYPE_CHECKING:
    import pandas as _pd
    DataFrameType: TypeAlias = _pd.DataFrame
else:
    DataFrameType: TypeAlias = Any

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    logger.warning("tqdm not available. Progress bars will not be shown.")
    TQDM_AVAILABLE = False
    def tqdm(iterable, desc=None, **kwargs):
        return iterable

# Optional imports with fallback
try:
    import oracledb
    ORACLE_AVAILABLE = True
except ImportError:
    logger.warning("oracledb not available. Database functions will not work.")
    ORACLE_AVAILABLE = False
    oracledb = None

try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    logger.warning("python-dotenv not available. Environment variables must be set manually.")
    DOTENV_AVAILABLE = False

# Configuration constants
RETRY_DELAY = 10
MIN_TEXT_LENGTH = 10
TICKER_REGEX = r'(?<!\w)(?:\$[A-Za-z]{1,5}|[A-Za-z]{2,5})(?:[.-][A-Za-z]{1,2})?(?!\w)'

# spaCy setup (optional) - Load model once at module level for performance
_SPACY_NLP = None  # Global spaCy model instance
SPACY_AVAILABLE = False

try:
    import spacy
    try:
        _SPACY_NLP = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
        logger.info("spaCy model loaded successfully")
    except OSError:
        logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
        _SPACY_NLP = None
        SPACY_AVAILABLE = False
except ImportError:
    logger.warning("spaCy not installed")
    _SPACY_NLP = None
    SPACY_AVAILABLE = False

# Setup progress_apply if both pandas and tqdm are available
if PANDAS_AVAILABLE and TQDM_AVAILABLE:
    tqdm.pandas()

def get_oracle_connection():
    """Get a new Oracle DB connection from environment variables."""
    try:
        cs = os.getenv("db-dsn")
        connection = oracledb.connect(
            user=os.getenv("db-username"),
            password=os.getenv("db-password"),
            dsn=cs
        )
        print("Oracle connection successful!")
        return connection
    except oracledb.DatabaseError as e:
        print(f"Error connecting to Oracle DB: {e}")
        return None


def safe_execute(conn, cursor, sql, params):
    """Execute a SQL statement safely with automatic reconnects."""
    while True:
        try:
            cursor.execute(sql, params)
            conn.commit()
            break
        except oracledb.DatabaseError as e:
            error_obj, = e.args
            print(f"DB error: {error_obj.message}")
            print(f"Retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)
            new_conn = get_oracle_connection()
            if new_conn:
                try:
                    conn.close()
                except:
                    pass
                conn = new_conn
                cursor = conn.cursor()
            else:
                print("Reconnect failed, retrying again...")
    return conn, cursor

# Data Processing Functions
def harmonize_schema(df_posts: DataFrameType, df_comments: DataFrameType) -> DataFrameType:
    """
    Rename columns and concatenate posts and comments to a unified dataframe.
    Expected input columns (based on your sample):
      posts: CREATED_UTC, ID, IS_ORIGINAL_CONTENT, SCORE, BODY, SUBREDDIT, UPVOTE_RATIO, URL
      comments: AUTHOR, CREATED_UTC, ID, PARENT_POST_ID, SCORE, BODY, SUBREDDIT
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas not available")

    # Make copies to avoid side effects
    posts = df_posts.copy()
    comments = df_comments.copy()

    # Normalize column names to lowercase for convenience
    posts.columns = [c.lower() for c in posts.columns]
    comments.columns = [c.lower() for c in comments.columns]

    # Posts rename map
    posts_rename = {
        "body": "text",
        "created_utc": "created_utc",
        "score": "score",
        "subreddit": "subreddit",
        "url": "url",
        "id": "id",
        "is_original_content": "is_original_content",
        "upvote_ratio": "upvote_ratio"
    }
    # Comments rename map
    comments_rename = {
        "body": "text",
        "created_utc": "created_utc",
        "score": "score",
        "subreddit": "subreddit",
        "id": "id",
        "parent_post_id": "parent_post_id",
        "author": "author"
    }

    posts = posts.rename(columns={k: v for k, v in posts_rename.items() if k in posts.columns})
    comments = comments.rename(columns={k: v for k, v in comments_rename.items() if k in comments.columns})

    # Ensure missing columns exist
    for col in ["author", "url", "parent_post_id", "is_original_content", "upvote_ratio"]:
        if col not in posts.columns:
            posts[col] = None
        if col not in comments.columns:
            comments[col] = None

    posts["type"] = "post"
    comments["type"] = "comment"

    # comments have no url, posts have no parent_post_id typically
    # These columns should already exist from the "Ensure missing columns exist" section above
    # No need to overwrite them here

    # unify column ordering (will be flexible)
    unified = pd.concat([posts, comments], ignore_index=True, sort=False)
    logger.info("Harmonized schema. Unified dataframe shape: %s", unified.shape)
    return unified


def drop_invalid_texts(df: DataFrameType, min_len: int = MIN_TEXT_LENGTH) -> DataFrameType:
    """
    Remove rows with missing, removed, deleted, or too-short text.
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas not available")

    df = df.copy()
    # remove NaN / empty
    df = df[df["text"].notna()]
    # strip whitespace for length checks
    df["text_stripped"] = df["text"].astype(str).str.strip()
    # remove deleted or removed content
    df = df[~df["text_stripped"].isin(["[deleted]", "[removed]", ""])]
    # remove short texts
    df = df[df["text_stripped"].str.len() > min_len]
    df = df.drop(columns=["text_stripped"])
    return df


def deduplicate_and_normalize_types(df: DataFrameType) -> DataFrameType:
    """
    Drop duplicates by 'id' and normalize types and numeric fields.
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas not available")

    df = df.copy()
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"])

    # created_utc -> datetime (handles seconds epoch)
    if "created_utc" in df.columns:
        # handle both numeric epochs and SQL datetime strings
        df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
        # If conversion failed (NaT), try parsing as string
        df.loc[df["created_utc"].isna(), "created_utc"] = pd.to_datetime(
            df.loc[df["created_utc"].isna(), "created_utc"], errors="coerce"
        )
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
    return df


def add_temporal_features(df: DataFrameType) -> DataFrameType:
    """Add temporal features based on created_utc column."""
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas not available")

    df = df.copy()
    df["date"] = df["created_utc"].dt.date
    df["hour"] = df["created_utc"].dt.hour
    df["day_of_week"] = df["created_utc"].dt.day_name()
    df["month"] = df["created_utc"].dt.month
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
    return df


def add_engagement_features(df: DataFrameType) -> DataFrameType:
    """Add engagement-related features."""
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas not available")
    if not NUMPY_AVAILABLE:
        raise ImportError("numpy not available")

    df = df.copy()
    df["text_length"] = df["text"].astype(str).str.len()
    df["word_count"] = df["text"].astype(str).apply(lambda x: len(x.split()))

    # upvote_ratio may be missing -> numeric
    if "upvote_ratio" in df.columns:
        try:
            df["upvote_ratio"] = pd.to_numeric(df["upvote_ratio"], errors="coerce")
        except Exception:
            df["upvote_ratio"] = None
    return df


def fetch_nasdaq_listed() -> DataFrameType:
    """
    Fetch NASDAQ-listed companies from nasdaqlisted.txt feed.
    Returns a DataFrame with at least columns: Symbol, Security Name, Test Issue (when available).
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas not available")

    primary = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    ftp_url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"

    def _download_text(url: str) -> str:
        try:
            if url.startswith("http"):
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (compatible; sentiment/1.0)'})
                with urllib.request.urlopen(req) as resp:
                    return resp.read().decode('utf-8', errors='ignore')
            else:
                with urllib.request.urlopen(url) as resp:
                    return resp.read().decode('utf-8', errors='ignore')
        except Exception as e:
            raise e

    content = None
    try:
        content = _download_text(primary)
    except Exception:
        try:
            content = _download_text(ftp_url)
        except Exception as e:
            raise RuntimeError(f"Failed to download NASDAQ list from both HTTP and FTP: {e}")

    df = pd.read_csv(StringIO(content), sep='|', comment='#', dtype=str)
    # Drop footer rows like "File Creation Time:..."
    if 'Symbol' in df.columns:
        df = df[~df['Symbol'].fillna('').str.contains('File Creation', na=False, case=False)]
    # Drop test issues when column present
    if 'Test Issue' in df.columns:
        df = df[df['Test Issue'] != 'Y']
    return df.reset_index(drop=True)


def fetch_nyse_listed() -> DataFrameType:
    """
    Fetch NYSE-listed companies from a public CSV (DataHub mirror or similar).
    Returns DataFrame with columns including ACT Symbol and Company Name when available.
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas not available")

    # Try a couple of mirrors
    urls = [
        "https://datahub.io/core/nyse-other-listings/r/nyse-listed.csv",
        "https://pkgstore.datahub.io/core/nyse-other-listings/nyse-listed_csv/data/nyse-listed_csv.csv"
    ]
    last_err = None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (compatible; sentiment/1.0)'})
            with urllib.request.urlopen(req) as resp:
                content = resp.read().decode('utf-8', errors='ignore')
            df = pd.read_csv(StringIO(content), dtype=str)
            return df
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to download NYSE list: {last_err}")


def get_all_us_tickers() -> DataFrameType:
    """
    Returns merged DataFrame with tickers and names from NASDAQ & NYSE,
    with a column indicating the exchange. Columns: ticker, company_name, exchange
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas not available")

    nas = fetch_nasdaq_listed()
    ny = fetch_nyse_listed()

    # Normalize NASDAQ
    nas_cols = {c: c.strip() for c in nas.columns}
    nas = nas.rename(columns=nas_cols)
    if 'Symbol' not in nas.columns or 'Security Name' not in nas.columns:
        raise ValueError("Unexpected NASDAQ columns; expected 'Symbol' and 'Security Name'")
    nas2 = nas.rename(columns={
        'Symbol': 'ticker',
        'Security Name': 'company_name'
    }).copy()
    nas2['exchange'] = 'NASDAQ'

    # Normalize NYSE
    ny_cols = {c: c.strip() for c in ny.columns}
    ny = ny.rename(columns=ny_cols)
    # Try several possible header variants
    ticker_col = None
    for c in ['ACT Symbol', 'Symbol', 'Act Symbol', 'act_symbol', 'symbol']:
        if c in ny.columns:
            ticker_col = c
            break
    name_col = None
    for c in ['Company Name', 'Security Name', 'company_name']:
        if c in ny.columns:
            name_col = c
            break
    if ticker_col is None or name_col is None:
        raise ValueError("Unexpected NYSE columns; cannot find ticker or company name")

    ny2 = ny.rename(columns={
        ticker_col: 'ticker',
        name_col: 'company_name'
    }).copy()
    ny2['exchange'] = 'NYSE'

    combined = pd.concat([
        nas2[['ticker', 'company_name', 'exchange']],
        ny2[['ticker', 'company_name', 'exchange']]
    ], ignore_index=True)
    combined['ticker'] = combined['ticker'].astype(str).str.upper().str.strip()
    combined['company_name'] = combined['company_name'].astype(str)
    combined = combined.dropna(subset=['ticker']).drop_duplicates(subset=['ticker']).reset_index(drop=True)
    logger.info("Built combined US tickers: %d", len(combined))
    return combined


def detect_tickers_in_text(text: str, ticker_set: Set[str]) -> List[str]:
    """
    Extract ticker symbols from text with improved precision to avoid false positives.
    Only matches:
    1. $cashtags (e.g., $AAPL, $TSLA) - these are almost always intentional
    2. ALL-CAPS words that are known tickers (reduces false positives significantly)
    3. Ticker symbols with class suffixes (e.g., BRK.A, BRK-B)

    Excludes common English words even if they're valid ticker symbols.
    """
    if not isinstance(text, str) or text.strip() == "" or not ticker_set:
        return []

    found_tickers = set()

    # Common English words to exclude even if they're valid tickers
    english_words = {
        'A', 'AM', 'AN', 'AND', 'ARE', 'AS', 'AT', 'BE', 'BY', 'FOR', 'FROM',
        'HAS', 'HE', 'IN', 'IS', 'IT', 'ITS', 'OF', 'ON', 'THAT', 'THE',
        'TO', 'WAS', 'WILL', 'WITH', 'YOU', 'YOUR', 'ALL', 'BUT', 'CAN',
        'HAD', 'HER', 'HIM', 'HOW', 'MAN', 'NEW', 'NOW', 'OLD', 'SEE',
        'TWO', 'WHO', 'BOY', 'DID', 'GET', 'LET', 'MAY', 'PUT', 'SAY',
        'SHE', 'TOO', 'USE', 'WAY', 'WE', 'WELL', 'WERE', 'WHAT', 'WHEN',
        'WHERE', 'WHICH', 'WHO', 'WHY', 'WOULD', 'YEAR', 'YES', 'YET',
        'GOOD', 'GREAT', 'BEST', 'BETTER', 'MUCH', 'MORE', 'MOST', 'VERY',
        'AFTER', 'AGAIN', 'AGAINST', 'ALSO', 'ALWAYS', 'ANOTHER', 'ANY',
        'AROUND', 'BACK', 'BEFORE', 'BEING', 'BETWEEN', 'BOTH', 'CAME',
        'COME', 'COULD', 'DOWN', 'EACH', 'EVEN', 'EVERY', 'FIND', 'FIRST',
        'GIVE', 'GO', 'GOING', 'GOOD', 'GOT', 'HAVE', 'HERE', 'HOME',
        'INTO', 'JUST', 'KEEP', 'KNOW', 'LAST', 'LEFT', 'LIFE', 'LIKE',
        'LITTLE', 'LONG', 'LOOK', 'MADE', 'MAKE', 'MANY', 'MIGHT', 'MOVE',
        'MUCH', 'MUST', 'NEED', 'NEVER', 'NEXT', 'NIGHT', 'NO', 'NOT',
        'ONLY', 'OTHER', 'OUR', 'OUT', 'OVER', 'OWN', 'PLACE', 'RIGHT',
        'SAID', 'SAME', 'SHOULD', 'SINCE', 'SO', 'SOME', 'STILL', 'SUCH',
        'TAKE', 'THAN', 'THEM', 'THERE', 'THESE', 'THEY', 'THING', 'THINK',
        'THIS', 'THOSE', 'THROUGH', 'TIME', 'UNDER', 'UP', 'WANT', 'WATER',
        'WELL', 'WENT', 'WHAT', 'WHEN', 'WHERE', 'WHILE', 'WORK', 'WORLD',
        'WRITE', 'YEAR', 'YOUNG', 'YOUR', 'TAX', 'BEAT', 'EA', 'LE'
    }

    # 1. Find $cashtags (always include these - they're intentional ticker mentions)
    cashtag_pattern = r'\$([A-Za-z]{1,5}(?:[.-][A-Za-z]{1,2})?)\b'
    for match in re.finditer(cashtag_pattern, text):
        ticker = match.group(1).upper()
        # Normalize variants (BRK.B vs BRK-B)
        variants = [ticker]
        if "." in ticker:
            variants.append(ticker.replace(".", "-"))
        if "-" in ticker:
            variants.append(ticker.replace("-", "."))

        for variant in variants:
            if variant in ticker_set:
                found_tickers.add(variant)
                break

    # 2. Find ALL-CAPS words that are known tickers (more restrictive)
    caps_pattern = r'\b([A-Z]{2,5}(?:[.-][A-Z]{1,2})?)\b'
    for match in re.finditer(caps_pattern, text):
        ticker = match.group(1)

        # Skip common English words
        if ticker in english_words:
            continue

        # Normalize variants
        variants = [ticker]
        if "." in ticker:
            variants.append(ticker.replace(".", "-"))
        if "-" in ticker:
            variants.append(ticker.replace("-", "."))

        for variant in variants:
            if variant in ticker_set:
                found_tickers.add(variant)
                break

    return sorted(list(found_tickers))


def apply_ticker_detection(df: DataFrameType, tickers_df: DataFrameType) -> DataFrameType:
    """Apply ticker detection to the text column and add exchange information."""
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas not available")

    df = df.copy()
    ticker_set = set(tickers_df["ticker"].tolist())

    if TQDM_AVAILABLE:
        df["mentioned_tickers"] = df["text"].progress_apply(
            lambda x: detect_tickers_in_text(x, ticker_set)
        )
    else:
        df["mentioned_tickers"] = df["text"].apply(
            lambda x: detect_tickers_in_text(x, ticker_set)
        )

    df["n_tickers"] = df["mentioned_tickers"].apply(lambda lst: len(lst))

    return df


def normalize_text_for_sentiment(text: str, keep_tickers: bool = True) -> str:
    """
    Light normalization targeted for transformer models:
    - removes URLs
    - removes markdown links
    - strips excessive punctuation / whitespace
    - lowercases text
    - keeps $TICKER or TICKER optionally (preserves original case)
    """
    if not isinstance(text, str):
        return ""

    s = text

    # Step 1: Extract tickers BEFORE any transformation (if keeping them)
    ticker_positions = []
    if keep_tickers:
        # Find all ticker matches with their positions
        for match in re.finditer(TICKER_REGEX, s):
            ticker_positions.append((match.start(), match.end(), match.group()))

    # Step 2: Remove URLs
    s = re.sub(r'http\S+', ' ', s)
    s = re.sub(r'\[([^\]]+)\]\((?:http\S+)\)', r'\1', s)  # markdown links -> keep text

    # Step 3: Replace tickers with placeholders to protect them
    if keep_tickers and ticker_positions:
        # Sort by position in reverse to replace from end to start (avoid offset issues)
        ticker_map = {}
        offset = 0
        new_text = s
        for i, (start, end, ticker) in enumerate(ticker_positions):
            placeholder = f"__TICKER{i}__"
            ticker_map[placeholder] = ticker.upper()  # Store uppercase ticker
            # Adjust position for previous replacements
            adj_start = start + offset
            adj_end = end + offset
            new_text = new_text[:adj_start] + placeholder + new_text[adj_end:]
            offset += len(placeholder) - (end - start)
        s = new_text

    # Step 4: Remove punctuation and lowercase
    s = re.sub(r'[^A-Za-z0-9_\s]', ' ', s)  # Keep underscores for placeholders
    s = s.lower()

    # Step 5: Normalize whitespace
    s = re.sub(r'\s+', ' ', s).strip()

    # Step 6: Restore tickers in uppercase
    if keep_tickers and ticker_positions:
        for placeholder, ticker in ticker_map.items():
            s = s.replace(placeholder.lower(), ticker)

    return s


def apply_text_normalization(df: DataFrameType, keep_tickers: bool = True) -> DataFrameType:
    """Apply text normalization to prepare for sentiment analysis."""
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas not available")

    df = df.copy()

    if TQDM_AVAILABLE:
        df["sentiment_ready_text"] = df["text"].progress_apply(
            lambda x: normalize_text_for_sentiment(x, keep_tickers)
        )
    else:
        df["sentiment_ready_text"] = df["text"].apply(
            lambda x: normalize_text_for_sentiment(x, keep_tickers)
        )

    return df


# Simple standalone functions that don't require pandas
def check_dependencies():
    """Check which dependencies are available."""
    deps = {
        "pandas": PANDAS_AVAILABLE,
        "numpy": NUMPY_AVAILABLE,
        "tqdm": TQDM_AVAILABLE,
        "oracledb": ORACLE_AVAILABLE,
        "dotenv": DOTENV_AVAILABLE,
        "spacy": SPACY_AVAILABLE
    }

    print("Dependency Status:")
    print("=" * 40)
    for dep, available in deps.items():
        status = "✅ Available" if available else "❌ Not Available"
        print(f"{dep:12} : {status}")

    return deps


def remove_financial_stopwords(text: str, preserve_tickers: bool = True) -> str:
    """
    Remove stopwords using NLTK but preserve financial terms and ticker symbols.
    """
    if not isinstance(text, str):
        return ""

    # Try to use NLTK stopwords, fall back to basic list if not available
    try:
        import nltk
        from nltk.corpus import stopwords

        # Download stopwords if not already present (with error handling)
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            try:
                nltk.download('stopwords', quiet=True)
            except Exception:
                logger.warning("Could not download NLTK stopwords, using basic list")
                raise ImportError("NLTK stopwords not available")

        english_stopwords = set(stopwords.words('english'))
    except (ImportError, Exception):
        # Fallback to basic stopwords list
        english_stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'my', 'myself', 'our', 'ours', 'ourselves', 'your', 'yours',
            'yourself', 'yourselves', 'him', 'his', 'himself', 'her', 'hers',
            'herself', 'its', 'itself', 'they', 'them', 'their', 'theirs',
            'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
            'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a',
            'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during',
            'before', 'after', 'above', 'below', 'up', 'down', 'in', 'out',
            'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
        }

    # Extract tickers if preserving them
    tickers = set()
    if preserve_tickers:
        ticker_matches = re.findall(TICKER_REGEX, text)
        tickers = {match.upper() for match in ticker_matches}

    # Split and filter words
    words = text.split()
    filtered_words = []

    for word in words:
        # Skip empty words
        if not word or not word.strip():
            continue

        # Clean word for comparison (remove punctuation)
        word_clean = re.sub(r'[^\w]', '', word).lower()

        # Skip if nothing left after cleaning
        if not word_clean:
            continue

        word_upper = word.upper()

        # Keep if:
        # 1. It's a ticker symbol (preserve tickers if requested)
        # 2. It's NOT a stopword
        is_ticker = preserve_tickers and word_upper in tickers
        is_stopword = word_clean in english_stopwords

        # Keep word if it's a ticker OR if it's not a stopword
        if is_ticker or not is_stopword:
            filtered_words.append(word)

    return ' '.join(filtered_words)


def remove_stopwords_spacy(text: str, preserve_tickers: bool = True) -> str:
    """
    Alternative stopword removal using spaCy (more advanced).
    Uses pre-loaded global spaCy model for better performance.
    """
    if not isinstance(text, str):
        return ""

    # Use pre-loaded global spaCy model
    if not SPACY_AVAILABLE or _SPACY_NLP is None:
        # Fall back to NLTK method if spaCy not available
        return remove_financial_stopwords(text, preserve_tickers)

    try:
        # Extract tickers if preserving them
        tickers = set()
        if preserve_tickers:
            ticker_matches = re.findall(TICKER_REGEX, text)
            tickers = {match.upper() for match in ticker_matches}

        # Process with pre-loaded spaCy model (NO LOADING HERE - HUGE PERFORMANCE GAIN)
        doc = _SPACY_NLP(text)
        filtered_tokens = []

        for token in doc:
            token_text = token.text
            token_upper = token.text.upper()

            # Skip empty or whitespace-only tokens
            if not token_text or not token_text.strip():
                continue

            # Keep if:
            # 1. It's a ticker symbol (preserve tickers if requested)
            # 2. It's NOT a stopword AND not punctuation
            is_ticker = preserve_tickers and token_upper in tickers
            is_content_word = not token.is_stop and not token.is_punct

            # Keep word if it's a ticker OR if it's content word
            if is_ticker or is_content_word:
                filtered_tokens.append(token_text)

        return ' '.join(filtered_tokens)

    except Exception as e:
        logger.warning(f"spaCy processing error: {e}, falling back to NLTK method")
        return remove_financial_stopwords(text, preserve_tickers)
        return remove_financial_stopwords(text, preserve_tickers)


def load_ticker_stopwords() -> Set[str]:
    """
    Load ticker stopwords from JSON file.
    Returns a set of uppercase words that should NOT be considered valid tickers.
    """
    import json

    script_dir = os.path.dirname(os.path.abspath(__file__))
    stopwords_path = os.path.join(script_dir, 'inputs/ticker_stopwords.json')

    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Get stopwords from the flat list
        stopwords = set(data.get('stopwords', []))

        # Ensure all are uppercase
        stopwords = {word.upper() for word in stopwords}

        logger.info(f"Loaded {len(stopwords)} ticker stopwords")
        return stopwords

    except FileNotFoundError:
        logger.warning(f"Ticker stopwords file not found at {stopwords_path}")
        logger.warning("Using empty stopwords set - no filtering will occur")
        return set()
    except Exception as e:
        logger.warning(f"Error loading ticker stopwords: {e}")
        return set()


def filter_ticker_stopwords(ticker_list: List[str], stopwords: Set[str]) -> List[str]:
    """
    Filter out false positive tickers from a list.

    Args:
        ticker_list: List of ticker symbols
        stopwords: Set of words that should not be considered tickers

    Returns:
        Filtered list of valid tickers
    """
    if not ticker_list or len(ticker_list) == 0:
        return []

    return [ticker for ticker in ticker_list if ticker.upper() not in stopwords]


def apply_ticker_stopword_filter(df: DataFrameType, stopwords: Set[str]) -> DataFrameType:
    """
    Apply ticker stopword filtering to a DataFrame with mentioned_tickers column.
    Updates mentioned_tickers and n_tickers columns.

    Args:
        df: DataFrame with mentioned_tickers column (list of tickers)
        stopwords: Set of words to filter out

    Returns:
        DataFrame with filtered tickers
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas not available")

    df = df.copy()

    # Filter the ticker lists
    df['mentioned_tickers'] = df['mentioned_tickers'].apply(
        lambda x: filter_ticker_stopwords(x, stopwords) if isinstance(x, list) else []
    )

    # Update ticker count
    df['n_tickers'] = df['mentioned_tickers'].apply(len)

    filtered_count = len(df[df['n_tickers'] > 0])
    logger.info(f"After stopword filtering: {filtered_count} rows with valid tickers")

    return df

def score_texts(texts):
    """
    Run FinBERT on a list of texts and return structured sentiment info.

    Parameters
    ----------
    texts : list of str
        The input texts to classify.

    Returns
    -------
    results : list of dict
        Each dict has:
        - sentiment_label : str       ('positive', 'neutral', 'negative')
        - sentiment_score : float     (p_pos - p_neg in [-1, 1])
        - p_pos, p_neu, p_neg : float (probabilities)
    """
    # This calls the HF pipeline once for the whole batch
    outputs = sentiment_pipe(texts)

    results = []
    for out in outputs:
        # out is a list like:
        # [{'label': 'positive', 'score': 0.7}, {'label': 'neutral', 'score': 0.2}, {'label': 'negative', 'score': 0.1}]
        # Normalize label names to lowercase to be robust to variations
        probs = {d["label"].lower(): float(d["score"]) for d in out}

        p_pos = probs.get("positive", 0.0)
        p_neg = probs.get("negative", 0.0)
        p_neu = probs.get("neutral", 0.0)

        # Continuous sentiment score in [-1, 1]
        sentiment_score = p_pos - p_neg

        # Discrete label = argmax over the three probabilities
        sentiment_label = max(probs, key=probs.get)

        results.append({
            "sentiment_label": sentiment_label,
            "sentiment_score": sentiment_score,
            "p_pos": p_pos,
            "p_neu": p_neu,
            "p_neg": p_neg
        })

    return results

def fetch_historical_prices(ticker, start_date, end_date) -> DataFrameType:
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)
    return hist


def check_market_moved_before_date(df: DataFrameType, target_date: str) -> dict:
    """
    Determines whether a stock has already made a significant move before a given sentiment-spike date.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: Open, High, Low, Close, Volume (indexed by date)
    target_date : str
        The sentiment-spike date to check (format: 'YYYY-MM-DD')

    Returns:
    --------
    dict : Dictionary containing all intermediate signals and the final market_moved_flag,
           or None if calculation cannot be performed
    """
    if not PANDAS_AVAILABLE or not NUMPY_AVAILABLE:
        raise ImportError("pandas and numpy are required for this function")

    try:
        # Handle multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        # Make a copy to avoid modifying original
        df = df.copy()

        # Ensure df has a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Normalize the index to remove timezone info for easier comparison
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Also normalize to date only (remove time component)
        df.index = df.index.normalize()

        # Convert target_date to datetime (timezone-naive)
        target_dt = pd.to_datetime(target_date).normalize()

        # Check if target_date exists in the dataframe
        if target_dt not in df.index:
            return None

        # Get data up to and including target_date
        df_until_target = df[df.index <= target_dt].copy()

        # Check minimum required history (need at least 20 days for rolling stats)
        if len(df_until_target) < 20:
            return None

        # 1. Calculate daily returns
        df_until_target['returns'] = df_until_target['Close'].pct_change()

        # 2. Compute 3-day percentage price change (t-3 to t) with z-score normalization
        if len(df_until_target) < 3:
            return None
        close_t = df_until_target.loc[target_dt, 'Close']
        close_t_minus_3 = df_until_target['Close'].iloc[-4] if len(df_until_target) >= 4 else np.nan
        pct_change_3d = ((close_t - close_t_minus_3) / close_t_minus_3) * 100 if not pd.isna(close_t_minus_3) else np.nan

        # Compute rolling 3-day % change series for z-score (normalized against stock's own volatility)
        df_until_target['pct_change_3d_series'] = df_until_target['Close'].pct_change(periods=3) * 100
        pct_3d_rolling_mean = df_until_target['pct_change_3d_series'].rolling(window=20, min_periods=20).mean()
        pct_3d_rolling_std = df_until_target['pct_change_3d_series'].rolling(window=20, min_periods=20).std()
        df_until_target['pct_3d_z'] = (df_until_target['pct_change_3d_series'] - pct_3d_rolling_mean) / pct_3d_rolling_std
        pct_3d_z = df_until_target.loc[target_dt, 'pct_3d_z']

        # 3. Compute abnormal return z-score (20-day rolling)
        rolling_mean = df_until_target['returns'].rolling(window=20, min_periods=20).mean()
        rolling_std = df_until_target['returns'].rolling(window=20, min_periods=20).std()
        df_until_target['ret_z'] = (df_until_target['returns'] - rolling_mean) / rolling_std
        ret_z = df_until_target.loc[target_dt, 'ret_z']

        # 4. Compute volatility expansion (3-day vs 20-day std) as a rolling series
        df_until_target['std_3d'] = df_until_target['returns'].rolling(window=3).std()
        df_until_target['std_20d'] = df_until_target['returns'].rolling(window=20).std()
        df_until_target['vol_expansion'] = df_until_target['std_3d'] / df_until_target['std_20d']
        vol_expansion = df_until_target.loc[target_dt, 'vol_expansion']

        # 5. Compute abnormal volume z-score (20-day rolling)
        vol_rolling_mean = df_until_target['Volume'].rolling(window=20, min_periods=20).mean()
        vol_rolling_std = df_until_target['Volume'].rolling(window=20, min_periods=20).std()
        df_until_target['vol_z'] = (df_until_target['Volume'] - vol_rolling_mean) / vol_rolling_std
        vol_z = df_until_target.loc[target_dt, 'vol_z']

        # 6. Compute ATR-14 using True Range
        df_until_target['prev_close'] = df_until_target['Close'].shift(1)
        hl = df_until_target['High'] - df_until_target['Low']
        hc = (df_until_target['High'] - df_until_target['prev_close']).abs()
        lc = (df_until_target['Low'] - df_until_target['prev_close']).abs()
        df_until_target['tr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df_until_target['atr_14'] = df_until_target['tr'].rolling(window=14, min_periods=14).mean()
        atr_14 = df_until_target.loc[target_dt, 'atr_14']

        # 7. Compute ATR-based move (3-day absolute price move / ATR-14) as a rolling series
        df_until_target['abs_3d_move'] = df_until_target['Close'].diff(periods=3).abs()
        df_until_target['atr_move'] = df_until_target['abs_3d_move'] / df_until_target['atr_14']
        atr_move = df_until_target.loc[target_dt, 'atr_move']

        # 8. Combine signals into market_moved_flag
        # Check last 5 trading days (same as visualization logic)
        recent_days = 5
        df_recent = df_until_target.tail(recent_days)

        market_moved_flag = False

        # Check if ANY of the last 5 days triggered any threshold
        # OPTIMIZED FOR SHARPE RATIO (grid search results)
        # Z_THRESHOLD = 2.0, VOL_EXPANSION = 1.75, ATR_MOVE = 2.0
        if (df_recent['pct_3d_z'].abs() >= 2.0).any():
            market_moved_flag = True
        if (df_recent['ret_z'].abs() >= 2.0).any():
            market_moved_flag = True
        if (df_recent['vol_z'].abs() >= 2.0).any():
            market_moved_flag = True
        if (df_recent['vol_expansion'] >= 1.75).any():
            market_moved_flag = True
        if (df_recent['atr_move'] >= 2.0).any():
            market_moved_flag = True

        # 9. Return all intermediate values and final flag
        return {
            'target_date': target_date,
            'pct_change_3d': round(float(pct_change_3d), 2) if not pd.isna(pct_change_3d) else None,
            'pct_3d_z': round(float(pct_3d_z), 2) if not pd.isna(pct_3d_z) else None,
            'ret_z': round(float(ret_z), 2) if not pd.isna(ret_z) else None,
            'vol_z': round(float(vol_z), 2) if not pd.isna(vol_z) else None,
            'vol_expansion': round(float(vol_expansion), 2) if not pd.isna(vol_expansion) else None,
            'atr_14': round(float(atr_14), 2) if not pd.isna(atr_14) else None,
            'atr_move': round(float(atr_move), 2) if not pd.isna(atr_move) else None,
            'market_moved_flag': market_moved_flag
        }

    except Exception as e:
        print(f"Error in check_market_moved_before_date: {str(e)}")
        return None


def visualize_market_move_analysis(df: DataFrameType, target_date: str, ticker: str = "Stock", window_days: int = 40):
    """
    Visualizes the market move analysis - compact PowerPoint-friendly layout.
    Only highlights triggers from the last 5 trading days.
    Uses z-score normalization for 3-day % change to adapt to stock volatility.
    """
    if not PANDAS_AVAILABLE or not NUMPY_AVAILABLE:
        raise ImportError("pandas and numpy are required for this function")

    # Import matplotlib here to avoid loading it if not needed
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # Prepare data
    df_plot = df.copy()
    if isinstance(df_plot.columns, pd.MultiIndex):
        df_plot.columns = [col[0] if isinstance(col, tuple) else col for col in df_plot.columns]
    if not isinstance(df_plot.index, pd.DatetimeIndex):
        df_plot.index = pd.to_datetime(df_plot.index)
    if df_plot.index.tz is not None:
        df_plot.index = df_plot.index.tz_localize(None)
    df_plot.index = df_plot.index.normalize()
    target_dt = pd.to_datetime(target_date).normalize()

    # Get window
    target_idx = df_plot.index.get_loc(target_dt)
    start_idx = max(0, target_idx - window_days + 1)
    df_window = df_plot.iloc[start_idx:target_idx + 1].copy()
    num_days = len(df_window)

    # Calculate metrics
    df_window['returns'] = df_window['Close'].pct_change()
    df_window['rolling_mean'] = df_window['returns'].rolling(window=20, min_periods=20).mean()
    df_window['rolling_std'] = df_window['returns'].rolling(window=20, min_periods=20).std()
    df_window['ret_z'] = (df_window['returns'] - df_window['rolling_mean']) / df_window['rolling_std']
    df_window['vol_rolling_mean'] = df_window['Volume'].rolling(window=20, min_periods=20).mean()
    df_window['vol_rolling_std'] = df_window['Volume'].rolling(window=20, min_periods=20).std()
    df_window['vol_z'] = (df_window['Volume'] - df_window['vol_rolling_mean']) / df_window['vol_rolling_std']
    df_window['prev_close'] = df_window['Close'].shift(1)
    hl = df_window['High'] - df_window['Low']
    hc = (df_window['High'] - df_window['prev_close']).abs()
    lc = (df_window['Low'] - df_window['prev_close']).abs()
    df_window['tr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df_window['atr_14'] = df_window['tr'].rolling(window=14, min_periods=14).mean()
    df_window['pct_change_3d'] = df_window['Close'].pct_change(periods=3) * 100
    # Compute 3-day % change z-score (normalized against rolling history)
    pct_3d_rolling_mean = df_window['pct_change_3d'].rolling(window=20, min_periods=20).mean()
    pct_3d_rolling_std = df_window['pct_change_3d'].rolling(window=20, min_periods=20).std()
    df_window['pct_3d_z'] = (df_window['pct_change_3d'] - pct_3d_rolling_mean) / pct_3d_rolling_std
    df_window['abs_3d_move'] = df_window['Close'].diff(periods=3).abs()
    df_window['atr_move'] = df_window['abs_3d_move'] / df_window['atr_14']
    df_window['std_3d'] = df_window['returns'].rolling(window=3).std()
    df_window['std_20d'] = df_window['returns'].rolling(window=20).std()
    df_window['vol_expansion'] = df_window['std_3d'] / df_window['std_20d']

    # Recent window for highlighting (last 5 days)
    recent_days = 5
    df_recent = df_window.tail(recent_days)
    recent_dates = set(df_recent.index)

    triggered_recent = {
        'pct_3d_z': df_recent[df_recent['pct_3d_z'].abs() >= 2]['pct_3d_z'],
        'ret_z': df_recent[df_recent['ret_z'].abs() >= 2]['ret_z'],
        'vol_z': df_recent[df_recent['vol_z'].abs() >= 2]['vol_z'],
        'vol_expansion': df_recent[df_recent['vol_expansion'] >= 1.75]['vol_expansion'],
        'atr_move': df_recent[df_recent['atr_move'] >= 2.0]['atr_move']
    }
    any_triggered = any(len(v) > 0 for v in triggered_recent.values())

    # COMPACT LAYOUT: 2x3 grid (PowerPoint friendly)
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # Colors (original scheme)
    c_price = '#2C3E50'
    c_pct = '#E67E22'
    c_thresh = '#FF6B6B'
    c_normal = '#4ECDC4'
    c_volume = '#9B59B6'
    c_triggered = '#FF6B6B'

    def format_ax(ax, title):
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
        # Shade recent period
        if len(df_recent) > 0:
            ax.axvspan(df_recent.index[0], df_recent.index[-1], alpha=0.15, color='#FFD700', zorder=0)

    # 1. PRICE CHART with dual y-axis (top-left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(df_window.index, df_window['Close'], color=c_price, linewidth=1.5, label=f'{ticker} Price')
    ax1.axvline(x=target_dt, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='Target Date')
    ax1.set_ylabel('Price ($)', fontsize=8, color=c_price)
    ax1.tick_params(axis='y', labelcolor=c_price)
    format_ax(ax1, f'{ticker} Price & 3d% Z-Score')

    # Secondary y-axis for 3-day % change Z-score (normalized)
    ax1b = ax1.twinx()
    ax1b.plot(df_window.index, df_window['pct_3d_z'], color=c_pct, linewidth=1.2, alpha=0.8, label='3d% Z-Score')
    ax1b.axhline(y=2, color=c_thresh, linestyle=':', linewidth=1.2, alpha=0.8, label='Threshold (±2σ)')
    ax1b.axhline(y=-2, color=c_thresh, linestyle=':', linewidth=1.2, alpha=0.8)
    ax1b.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax1b.set_ylabel('3d% Z-Score', fontsize=8, color=c_pct)
    ax1b.tick_params(axis='y', labelcolor=c_pct, labelsize=7)
    # Adjust y-limits
    pct_range = max(abs(df_window['pct_3d_z'].min()), abs(df_window['pct_3d_z'].max()), 3) * 1.2
    ax1b.set_ylim(-pct_range, pct_range)
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=7, framealpha=0.9)

    # 2. VOLATILITY EXPANSION (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(df_window.index, df_window['vol_expansion'], color=c_price, linewidth=1.2, marker='o', markersize=2, label='Vol Expansion')
    ax2.axhline(y=1.75, color=c_thresh, linestyle='--', linewidth=1, alpha=0.7, label='Threshold (1.75)')
    ax2.axhline(y=1, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Baseline (1.0)')
    # Highlight ONLY recent triggers
    recent_vol_exp_triggered = df_recent[df_recent['vol_expansion'] >= 1.75]
    if len(recent_vol_exp_triggered) > 0:
        ax2.scatter(recent_vol_exp_triggered.index, recent_vol_exp_triggered['vol_expansion'],
                   color=c_triggered, s=50, zorder=5, edgecolors='white', linewidths=0.5, label='Triggered')
    ax2.set_ylabel('Ratio', fontsize=8)
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='upper left', fontsize=6, framealpha=0.9)
    format_ax(ax2, 'Volatility Expansion (≥1.75)')

    # 3. RETURN Z-SCORE (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    colors_rz = [c_triggered if (idx in recent_dates and pd.notna(v) and abs(v) >= 2)
                 else c_normal for idx, v in zip(df_window.index, df_window['ret_z'])]
    ax3.bar(df_window.index, df_window['ret_z'], width=0.8, color=colors_rz, alpha=0.7, label='Return Z')
    ax3.axhline(y=2, color=c_thresh, linestyle='--', linewidth=1, alpha=0.7, label='Threshold (±2)')
    ax3.axhline(y=-2, color=c_thresh, linestyle='--', linewidth=1, alpha=0.7)
    ax3.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax3.set_ylabel('Z-Score', fontsize=8)
    ax3.legend(loc='upper left', fontsize=6, framealpha=0.9)
    format_ax(ax3, 'Return Z-Score (±2)')

    # 4. VOLUME Z-SCORE (bottom-center)
    ax4 = fig.add_subplot(gs[1, 1])
    colors_vz = [c_triggered if (idx in recent_dates and pd.notna(v) and abs(v) >= 2)
                 else c_volume for idx, v in zip(df_window.index, df_window['vol_z'])]
    ax4.bar(df_window.index, df_window['vol_z'], width=0.8, color=colors_vz, alpha=0.7, label='Volume Z')
    ax4.axhline(y=2, color=c_thresh, linestyle='--', linewidth=1, alpha=0.7, label='Threshold (±2)')
    ax4.axhline(y=-2, color=c_thresh, linestyle='--', linewidth=1, alpha=0.7)
    ax4.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax4.set_ylabel('Z-Score', fontsize=8)
    ax4.legend(loc='upper left', fontsize=6, framealpha=0.9)
    format_ax(ax4, 'Volume Z-Score (±2)')

    # 5. ATR MOVE (bottom-right)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(df_window.index, df_window['atr_move'], color=c_price, linewidth=1.2, marker='o', markersize=2, label='ATR Move')
    ax5.axhline(y=2.0, color=c_thresh, linestyle='--', linewidth=1, alpha=0.7, label='Threshold (2.0)')
    # Highlight ONLY recent triggers
    recent_atr_triggered = df_recent[df_recent['atr_move'] >= 2.0]
    if len(recent_atr_triggered) > 0:
        ax5.scatter(recent_atr_triggered.index, recent_atr_triggered['atr_move'],
                   color=c_triggered, s=50, zorder=5, edgecolors='white', linewidths=0.5, label='Triggered')
    ax5.set_ylabel('ATR Multiple', fontsize=8)
    ax5.set_ylim(bottom=0)
    ax5.legend(loc='upper left', fontsize=6, framealpha=0.9)
    format_ax(ax5, 'ATR Move (≥2.0)')

    # SUMMARY BOX - in the title area
    if any_triggered:
        status = "Market moved"
        status_color = '#FFCCCC'
    else:
        status = "No recent move"
        status_color = '#90EE90'

    fig.suptitle(f'{ticker} Market Analysis  •  Target: {target_date}  •  {status}',
                 fontsize=11, fontweight='bold', y=0.98,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=status_color, alpha=0.8))

    # Legend for recent period
    fig.text(0.99, 0.01, f'Yellow = Last {recent_days} days (trigger window)',
             fontsize=7, ha='right', va='bottom', style='italic', color='#666')

    plt.subplots_adjust(left=0, right=1, top=0.90, bottom=0.08, hspace=0.35, wspace=0.55)
    plt.show()

    return check_market_moved_before_date(df, target_date)

def adjust_to_trading_day(date_str):
    """Adjust weekend dates to the previous Friday."""
    dt = pd.to_datetime(date_str)
    # Saturday = 5, Sunday = 6
    if dt.weekday() == 5:  # Saturday -> Friday
        dt = dt - timedelta(days=1)
    elif dt.weekday() == 6:  # Sunday -> Friday
        dt = dt - timedelta(days=2)
    return dt.strftime('%Y-%m-%d')


# =============================================================================
# SENTIMENT_DAILY TABLE - CANONICAL DAILY AGGREGATION
# =============================================================================

def create_sentiment_daily_table(conn) -> bool:
    """
    Create the SENTIMENT_DAILY table structure in Oracle.
    This table serves as the canonical source for daily sentiment aggregations.

    Returns:
        bool: True if table created successfully, False otherwise
    """
    if not ORACLE_AVAILABLE:
        raise ImportError("oracledb not available")

    try:
        cursor = conn.cursor()

        # Drop existing table if any
        try:
            cursor.execute("DROP TABLE SENTIMENT_DAILY")
            logger.info("Dropped existing SENTIMENT_DAILY table")
        except:
            pass

        # Create table with comprehensive structure
        create_sql = """
        CREATE TABLE SENTIMENT_DAILY (
            TICKER VARCHAR2(20) NOT NULL,
            SENTIMENT_DATE DATE NOT NULL,
            SENTIMENT_MEAN NUMBER(15,6),
            SENTIMENT_MEDIAN NUMBER(15,6),
            SENTIMENT_STD NUMBER(15,6),
            MENTIONS NUMBER(10,0),
            TOTAL_UPVOTES NUMBER(15,2),
            AVG_UPVOTES NUMBER(15,4),
            WEIGHTED_SENTIMENT NUMBER(15,6),
            POSITIVE_COUNT NUMBER(10,0),
            NEUTRAL_COUNT NUMBER(10,0),
            NEGATIVE_COUNT NUMBER(10,0),
            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT PK_SENTIMENT_DAILY PRIMARY KEY (TICKER, SENTIMENT_DATE)
        )
        """
        cursor.execute(create_sql)

        # Create index for fast lookups
        cursor.execute("CREATE INDEX IDX_SENTIMENT_DAILY_DATE ON SENTIMENT_DAILY(SENTIMENT_DATE)")
        cursor.execute("CREATE INDEX IDX_SENTIMENT_DAILY_TICKER ON SENTIMENT_DAILY(TICKER)")

        conn.commit()
        cursor.close()

        logger.info("Created SENTIMENT_DAILY table with indexes")
        return True

    except Exception as e:
        logger.error(f"Failed to create SENTIMENT_DAILY table: {e}")
        return False


def populate_sentiment_daily(conn, start_date: str = None, end_date: str = None) -> int:
    """
    Populate SENTIMENT_DAILY table by aggregating from SENTIMENT_RESULTS.

    This function aggregates raw sentiment data (many rows per ticker per day)
    into a single row per (ticker, date) with computed statistics.

    Args:
        conn: Oracle database connection
        start_date: Start date (YYYY-MM-DD), defaults to min date in SENTIMENT_RESULTS
        end_date: End date (YYYY-MM-DD), defaults to max date in SENTIMENT_RESULTS

    Returns:
        int: Number of rows inserted, or -1 if failed
    """
    if not ORACLE_AVAILABLE or not PANDAS_AVAILABLE:
        raise ImportError("oracledb and pandas are required")

    try:
        cursor = conn.cursor()

        # Build date filter clause
        date_filter = ""
        params = {}
        if start_date:
            date_filter += " AND TRUNC(TO_DATE('1970-01-01', 'YYYY-MM-DD') + CREATED_UTC / 86400) >= TO_DATE(:start_dt, 'YYYY-MM-DD')"
            params['start_dt'] = start_date
        if end_date:
            date_filter += " AND TRUNC(TO_DATE('1970-01-01', 'YYYY-MM-DD') + CREATED_UTC / 86400) <= TO_DATE(:end_dt, 'YYYY-MM-DD')"
            params['end_dt'] = end_date

        # Aggregation query with comprehensive metrics
        # NOTE: CREATED_UTC is Unix seconds (e.g., 1704140129)
        insert_sql = f"""
        INSERT INTO SENTIMENT_DAILY (
            TICKER, SENTIMENT_DATE, SENTIMENT_MEAN, SENTIMENT_MEDIAN, SENTIMENT_STD,
            MENTIONS, TOTAL_UPVOTES, AVG_UPVOTES, WEIGHTED_SENTIMENT,
            POSITIVE_COUNT, NEUTRAL_COUNT, NEGATIVE_COUNT
        )
        SELECT
            TICKER,
            TRUNC(TO_DATE('1970-01-01', 'YYYY-MM-DD') + CREATED_UTC / 86400) as SENTIMENT_DATE,
            AVG(FINAL_SENTIMENT_SCORE) as SENTIMENT_MEAN,
            MEDIAN(FINAL_SENTIMENT_SCORE) as SENTIMENT_MEDIAN,
            STDDEV(FINAL_SENTIMENT_SCORE) as SENTIMENT_STD,
            COUNT(*) as MENTIONS,
            SUM(NORMALIZED_UPVOTES) as TOTAL_UPVOTES,
            AVG(NORMALIZED_UPVOTES) as AVG_UPVOTES,
            SUM(FINAL_SENTIMENT_SCORE * NORMALIZED_UPVOTES) / NULLIF(SUM(NORMALIZED_UPVOTES), 0) as WEIGHTED_SENTIMENT,
            SUM(CASE WHEN FINAL_SENTIMENT_LABEL = 'positive' THEN 1 ELSE 0 END) as POSITIVE_COUNT,
            SUM(CASE WHEN FINAL_SENTIMENT_LABEL = 'neutral' THEN 1 ELSE 0 END) as NEUTRAL_COUNT,
            SUM(CASE WHEN FINAL_SENTIMENT_LABEL = 'negative' THEN 1 ELSE 0 END) as NEGATIVE_COUNT
        FROM SENTIMENT_RESULTS
        WHERE 1=1 {date_filter}
        GROUP BY
            TICKER,
            TRUNC(TO_DATE('1970-01-01', 'YYYY-MM-DD') + CREATED_UTC / 86400)
        """

        cursor.execute(insert_sql, params)
        rows_inserted = cursor.rowcount
        conn.commit()
        cursor.close()

        logger.info(f"Populated SENTIMENT_DAILY with {rows_inserted:,} rows")
        return rows_inserted

    except Exception as e:
        logger.error(f"Failed to populate SENTIMENT_DAILY: {e}")
        import traceback
        traceback.print_exc()
        return -1


def refresh_sentiment_daily(start_date: str = None, end_date: str = None) -> int:
    """
    Full refresh of SENTIMENT_DAILY table: drop, create, and populate.

    Args:
        start_date: Start date (YYYY-MM-DD), optional
        end_date: End date (YYYY-MM-DD), optional

    Returns:
        int: Number of rows inserted, or -1 if failed
    """
    conn = get_oracle_connection()
    if not conn:
        logger.error("Failed to connect to Oracle")
        return -1

    try:
        # Create table structure
        if not create_sentiment_daily_table(conn):
            return -1

        # Populate with aggregated data
        rows = populate_sentiment_daily(conn, start_date, end_date)

        return rows

    finally:
        conn.close()


def get_daily_sentiment_from_oracle(conn, ticker: str = None,
                                     start_date: str = None,
                                     end_date: str = None) -> DataFrameType:
    """
    Load daily sentiment data from SENTIMENT_DAILY table.

    This is the canonical way to get aggregated daily sentiment data.
    Falls back to direct aggregation if SENTIMENT_DAILY doesn't exist.

    Args:
        conn: Oracle database connection
        ticker: Filter by specific ticker (optional)
        start_date: Start date filter (YYYY-MM-DD)
        end_date: End date filter (YYYY-MM-DD)

    Returns:
        DataFrame with daily sentiment data
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas not available")

    # Build query with filters
    where_clauses = []
    params = {}

    if ticker:
        where_clauses.append("TICKER = :ticker")
        params['ticker'] = ticker
    if start_date:
        where_clauses.append("SENTIMENT_DATE >= TO_DATE(:start_dt, 'YYYY-MM-DD')")
        params['start_dt'] = start_date
    if end_date:
        where_clauses.append("SENTIMENT_DATE <= TO_DATE(:end_dt, 'YYYY-MM-DD')")
        params['end_dt'] = end_date

    where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

    query = f"""
    SELECT
        TICKER, SENTIMENT_DATE, SENTIMENT_MEAN, SENTIMENT_MEDIAN, SENTIMENT_STD,
        MENTIONS, TOTAL_UPVOTES, AVG_UPVOTES, WEIGHTED_SENTIMENT,
        POSITIVE_COUNT, NEUTRAL_COUNT, NEGATIVE_COUNT
    FROM SENTIMENT_DAILY
    WHERE {where_clause}
    ORDER BY TICKER, SENTIMENT_DATE
    """

    try:
        df = pd.read_sql_query(query, conn, params=params)
        logger.info(f"Loaded {len(df):,} rows from SENTIMENT_DAILY")
        return df
    except Exception as e:
        # Table might not exist - fall back to direct aggregation
        logger.warning(f"SENTIMENT_DAILY query failed: {e}")
        logger.info("Falling back to direct aggregation from SENTIMENT_RESULTS")
        return _aggregate_sentiment_directly(conn, ticker, start_date, end_date)


def _aggregate_sentiment_directly(conn, ticker: str = None,
                                   start_date: str = None,
                                   end_date: str = None) -> DataFrameType:
    """
    Fallback: aggregate sentiment directly from SENTIMENT_RESULTS if SENTIMENT_DAILY doesn't exist.
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas not available")

    where_clauses = []
    params = {}

    if ticker:
        where_clauses.append("TICKER = :ticker")
        params['ticker'] = ticker
    if start_date:
        where_clauses.append("TRUNC(TO_DATE('1970-01-01', 'YYYY-MM-DD') + CREATED_UTC / 86400) >= TO_DATE(:start_dt, 'YYYY-MM-DD')")
        params['start_dt'] = start_date
    if end_date:
        where_clauses.append("TRUNC(TO_DATE('1970-01-01', 'YYYY-MM-DD') + CREATED_UTC / 86400) <= TO_DATE(:end_dt, 'YYYY-MM-DD')")
        params['end_dt'] = end_date

    where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

    query = f"""
    SELECT
        TICKER,
        TRUNC(TO_DATE('1970-01-01', 'YYYY-MM-DD') + CREATED_UTC / 86400) as SENTIMENT_DATE,
        AVG(FINAL_SENTIMENT_SCORE) as SENTIMENT_MEAN,
        MEDIAN(FINAL_SENTIMENT_SCORE) as SENTIMENT_MEDIAN,
        STDDEV(FINAL_SENTIMENT_SCORE) as SENTIMENT_STD,
        COUNT(*) as MENTIONS,
        SUM(NORMALIZED_UPVOTES) as TOTAL_UPVOTES,
        AVG(NORMALIZED_UPVOTES) as AVG_UPVOTES,
        SUM(FINAL_SENTIMENT_SCORE * NORMALIZED_UPVOTES) / NULLIF(SUM(NORMALIZED_UPVOTES), 0) as WEIGHTED_SENTIMENT,
        SUM(CASE WHEN FINAL_SENTIMENT_LABEL = 'positive' THEN 1 ELSE 0 END) as POSITIVE_COUNT,
        SUM(CASE WHEN FINAL_SENTIMENT_LABEL = 'neutral' THEN 1 ELSE 0 END) as NEUTRAL_COUNT,
        SUM(CASE WHEN FINAL_SENTIMENT_LABEL = 'negative' THEN 1 ELSE 0 END) as NEGATIVE_COUNT
    FROM SENTIMENT_RESULTS
    WHERE {where_clause}
    GROUP BY
        TICKER,
        TRUNC(TO_DATE('1970-01-01', 'YYYY-MM-DD') + CREATED_UTC / 86400)
    ORDER BY TICKER, SENTIMENT_DATE
    """

    df = pd.read_sql_query(query, conn, params=params)
    logger.info(f"Aggregated {len(df):,} rows directly from SENTIMENT_RESULTS")
    return df