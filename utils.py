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
from dotenv import load_dotenv

load_dotenv()

import praw

reddit = praw.Reddit(
    client_id=os.getenv("api-client_id"),
    client_secret=os.getenv("api-client_secret"),
    username=os.getenv("api-username"),
    password=os.getenv("api-password"),
    user_agent="stock-sentiment-script"
)

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
    import praw
    REDDIT_AVAILABLE = True
except ImportError:
    logger.warning("praw not available. Reddit functions will not work.")
    REDDIT_AVAILABLE = False
    praw = None

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
# Broadened regex:
# - $cashtags allow 1-5 letters (e.g., $A, $aapl)
# - Bare tickers require 2-5 letters (avoid single-letter words like "I")
# - Optional dot or hyphen class suffix (e.g., BRK.B, BRK-B, AAPL.U)
# Word boundaries replaced with lookarounds to avoid consuming punctuation
TICKER_REGEX = r'(?<!\w)(?:\$[A-Za-z]{1,5}|[A-Za-z]{2,5})(?:[.-][A-Za-z]{1,2})?(?!\w)'

# spaCy setup (optional)
USE_SPACY = False
try:
    if USE_SPACY:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    else:
        nlp = None
        SPACY_AVAILABLE = False
except (ImportError, OSError):
    USE_SPACY = False
    nlp = None
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

# Reddit Functions
def fetch_and_insert_posts_comments(reddit, subreddit_name, last_fetched_utc, conn, cursor, comments_max=50):
    """Fetch posts and comments and insert them immediately into the DB."""
    subreddit = reddit.subreddit(subreddit_name)
    max_utc = last_fetched_utc

    for post in subreddit.hot(limit=None):
        if post.created_utc <= last_fetched_utc:
            continue
    
        cursor.execute("SELECT id FROM current_posts WHERE id = :id", {"id": post.id})
        if cursor.fetchone():
            continue
    
        cursor.execute("SELECT id FROM current_posts WHERE id = :id", {"id": post.id})
        if cursor.fetchone():
            continue

        post_params = {
            "author": str(post.author) if post.author else None,
            "title": post.title,
            "created_utc": post.created_utc,
            "id": post.id,
            "is_original_content": 1 if post.is_original_content else 0,
            "score": post.score,
            "body": post.selftext,
            "subreddit": str(post.subreddit),
            "upvote_ratio": post.upvote_ratio,
            "url": post.url
        }

        conn, cursor = safe_execute(conn, cursor, """
            INSERT INTO current_posts (
                author, title, created_utc, id, is_original_content,
                score, body, subreddit, upvote_ratio, url
            ) VALUES (
                :author, :title, :created_utc, :id, :is_original_content,
                :score, :body, :subreddit, :upvote_ratio, :url
            )
        """, post_params)

        max_utc = max(max_utc, post.created_utc)
        time.sleep(1)

        post.comments.replace_more(limit=0)
        count = 0
        for comment in post.comments.list():
            if count >= comments_max:
                break
            if comment.created_utc <= last_fetched_utc:
                continue
            
            cursor.execute("SELECT id FROM current_comments WHERE id = :id", {"id": comment.id})
            if cursor.fetchone():
                continue
            
            
            cursor.execute("SELECT id FROM current_comments WHERE id = :id", {"id": comment.id})
            if cursor.fetchone():
                continue
            
            comment_params = {
                "author": str(comment.author) if comment.author else None,
                "created_utc": comment.created_utc,
                "id": comment.id,
                "parent_post_id": post.id,
                "score": comment.score,
                "body": comment.body,
                "subreddit": str(post.subreddit)
            }

            conn, cursor = safe_execute(conn, cursor, """
                INSERT INTO current_comments (
                    author, created_utc, id, parent_post_id, score, body, subreddit
                ) VALUES (
                    :author, :created_utc, :id, :parent_post_id, :score, :body, :subreddit
                )
            """, comment_params)

            count += 1
            time.sleep(2)

    return conn, cursor, max_utc


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
    posts["parent_post_id"] = posts.get("parent_post_id", None)
    comments["url"] = comments.get("url", None)
    
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
    logger.info("Dropped invalid/short texts. Remaining rows: %d", len(df))
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
    logger.info("Deduplicated and normalized types.")
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
    df["score_log1p"] = np.log1p(df["score"].astype(float))
    
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
    logger.info("Detecting tickers in text (this may take a while)...")
    
    if TQDM_AVAILABLE:
        df["mentioned_tickers"] = df["text"].progress_apply(
            lambda x: detect_tickers_in_text(x, ticker_set)
        )
    else:
        df["mentioned_tickers"] = df["text"].apply(
            lambda x: detect_tickers_in_text(x, ticker_set)
        )
    
    df["n_tickers"] = df["mentioned_tickers"].apply(lambda lst: len(lst))
    
    # Add exchange information
    # Create a mapping of ticker -> exchange
    ticker_to_exchange = dict(zip(tickers_df["ticker"], tickers_df["exchange"]))
    
    def get_ticker_exchanges(ticker_list):
        """Determine which exchanges are represented in the ticker list."""
        if not ticker_list or len(ticker_list) == 0:
            return ""
        
        exchanges = set()
        for ticker in ticker_list:
            exchange = ticker_to_exchange.get(ticker)
            if exchange:
                exchanges.add(exchange)
        
        if len(exchanges) == 0:
            return ""
        elif len(exchanges) == 1:
            return list(exchanges)[0]
        else:
            return "BOTH"
    
    df["ticker_exchanges"] = df["mentioned_tickers"].apply(get_ticker_exchanges)
    
    return df


def normalize_text_for_sentiment(text: str, keep_tickers: bool = True) -> str:
    """
    Light normalization targeted for VADER / transformer models:
    - removes URLs
    - removes markdown links
    - strips excessive punctuation / whitespace
    - keeps $TICKER or TICKER optionally
    """
    if not isinstance(text, str):
        return ""
    
    s = text
    # remove URLs
    s = re.sub(r'http\S+', ' ', s)
    s = re.sub(r'\[([^\]]+)\]\((?:http\S+)\)', r'\1', s)  # markdown links -> keep text
    
    # optionally keep ticker tokens as-is; otherwise lowercase everything
    # Replace non-alphanumeric except $ (for $TICKER) and whitespace
    if keep_tickers:
        s = re.sub(r'[^A-Za-z0-9\$\s]', ' ', s)
    else:
        s = re.sub(r'[^A-Za-z0-9\s]', ' ', s)
        s = s.lower()
    
    s = re.sub(r'\s+', ' ', s).strip()
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
        "praw": REDDIT_AVAILABLE,
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
    
    # Financial terms to preserve (common sentiment-relevant words)
    financial_keep_words = {
        'buy', 'sell', 'hold', 'bull', 'bear', 'bullish', 'bearish',
        'up', 'down', 'rise', 'fall', 'drop', 'crash', 'moon', 'rocket',
        'high', 'low', 'target', 'price', 'value', 'worth', 'cheap', 'expensive',
        'calls', 'puts', 'options', 'long', 'short', 'profit', 'loss', 'gain',
        'strong', 'weak', 'good', 'bad', 'great', 'terrible', 'excellent',
        'positive', 'negative', 'optimistic', 'pessimistic'
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
        # Clean word for comparison (remove punctuation except $)
        word_clean = re.sub(r'[^\w$]', '', word).lower()
        word_upper = word.upper()
        
        # Keep if:
        # 1. It's a ticker symbol
        # 2. It's a financial term we want to preserve
        # 3. It's not a stopword
        # 4. It's a number or contains digits (financial amounts, percentages)
        if (preserve_tickers and word_upper in tickers or
            word_clean in financial_keep_words or
            word_clean not in english_stopwords or
            any(char.isdigit() for char in word_clean) or
            '$' in word):
            filtered_words.append(word)
    
    return ' '.join(filtered_words)


def remove_stopwords_spacy(text: str, preserve_tickers: bool = True) -> str:
    """
    Alternative stopword removal using spaCy (more advanced).
    Use this if you have spaCy installed and want better language processing.
    """
    if not isinstance(text, str):
        return ""
    
    try:
        import spacy
        
        # Try to load English model
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            # Fall back to NLTK method
            return remove_financial_stopwords(text, preserve_tickers)
        
        # Financial terms to preserve
        financial_keep_words = {
            'buy', 'sell', 'hold', 'bull', 'bear', 'bullish', 'bearish',
            'up', 'down', 'rise', 'fall', 'drop', 'crash', 'moon', 'rocket',
            'high', 'low', 'target', 'price', 'value', 'worth', 'cheap', 'expensive',
            'calls', 'puts', 'options', 'long', 'short', 'profit', 'loss', 'gain',
            'strong', 'weak', 'good', 'bad', 'great', 'terrible', 'excellent',
            'positive', 'negative', 'optimistic', 'pessimistic'
        }
        
        # Extract tickers if preserving them
        tickers = set()
        if preserve_tickers:
            ticker_matches = re.findall(TICKER_REGEX, text)
            tickers = {match.upper() for match in ticker_matches}
        
        # Process with spaCy
        doc = nlp(text)
        filtered_tokens = []
        
        for token in doc:
            token_text = token.text
            token_lower = token.text.lower()
            token_upper = token.text.upper()
            
            # Keep if:
            # 1. It's a ticker symbol
            # 2. It's a financial term we want to preserve
            # 3. It's not a stopword
            # 4. It's not punctuation (except $ signs)
            # 5. It contains digits
            if (preserve_tickers and token_upper in tickers or
                token_lower in financial_keep_words or
                not token.is_stop or
                token.like_num or
                '$' in token_text or
                (not token.is_punct or '$' in token_text)):
                filtered_tokens.append(token_text)
        
        return ' '.join(filtered_tokens)
        
    except ImportError:
        logger.warning("spaCy not available, falling back to NLTK method")
        return remove_financial_stopwords(text, preserve_tickers)





