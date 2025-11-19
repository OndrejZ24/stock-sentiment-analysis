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






