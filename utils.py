#!/usr/bin/env python3
"""
Stock Sentiment Analysis Utilities
A robust utility module for stock sentiment analysis with graceful dependency handling.
"""

import os
import time
import re
import json
import urllib.request
from io import StringIO
import logging
from typing import List, Optional, Tuple, Set, Dict, Any, Union, TYPE_CHECKING, TypeAlias

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core imports with graceful fallbacks
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

# Provide a typing-friendly alias that doesn't break at runtime when pandas is missing
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

# Optional imports with graceful fallback
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
TICKER_FILEPATH = "tickers.csv"
# Broadened regex:
# - $cashtags allow 1-5 letters (e.g., $A, $aapl)
# - Bare tickers require 2-5 letters (avoid single-letter words like "I")
# - Optional dot or hyphen class suffix (e.g., BRK.B, BRK-B, AAPL.U)
# Word boundaries replaced with lookarounds to avoid consuming punctuation
TICKER_REGEX = r'(?<!\w)(?:\$[A-Za-z]{1,5}|[A-Za-z]{2,5})(?:[.-][A-Za-z]{1,2})?(?!\w)'
CLEANED_TABLE = "cleaned_posts_comments"
USE_SPACY = False

# spaCy setup (optional)
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


# Database Functions
def create_sample_data():
    """Create sample Reddit data for testing when database is not available."""
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas not available")
    
    sample_posts_data = {
        'ID': ['post_1', 'post_2', 'post_3'],
        'TITLE': [
            'Tesla stock discussion TSLA',
            'Apple earnings $AAPL analysis', 
            'Microsoft vs Google comparison'
        ],
        'BODY': [
            'I think TSLA will go to the moon! Tesla is revolutionary.',
            'AAPL earnings beat expectations. Apple stock looks strong.',
            'MSFT and GOOGL are both solid picks for long term investment.'
        ],
        'CREATED_UTC': [1638360000, 1638363600, 1638367200],
        'SCORE': [45, 78, 23],
        'SUBREDDIT': ['stocks', 'investing', 'StockMarket']
    }
    
    sample_comments_data = {
        'ID': ['comment_1', 'comment_2', 'comment_3'],
        'BODY': [
            'I agree about $TSLA! Tesla is the future.',
            'AAPL has been my best performer this year.',
            'MSFT is undervalued compared to other tech stocks.'
        ],
        'CREATED_UTC': [1638360300, 1638363900, 1638367500],
        'SCORE': [12, 8, 15],
        'SUBREDDIT': ['stocks', 'investing', 'StockMarket'],
        'PARENT_POST_ID': ['post_1', 'post_2', 'post_3']
    }
    
    posts_df = pd.DataFrame(sample_posts_data)
    comments_df = pd.DataFrame(sample_comments_data)
    
    logger.info(f"Created sample data: {posts_df.shape} posts, {comments_df.shape} comments")
    return posts_df, comments_df

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("api-client_id"),
    client_secret=os.getenv("api-client_secret"),
    user_agent="stock_market_scraper",
    username=os.getenv("api-username"),
    password=os.getenv("api-password")
    )

def get_oracle_connection():
    """Get a new Oracle DB connection from environment variables."""
    if not ORACLE_AVAILABLE:
        raise ImportError("oracledb not available")
    
    try:
        # Using your project's environment variable naming convention
        dsn = os.getenv('db-dsn')  # Your project uses 'db-dsn'
        username = os.getenv('db-username')  # Your project uses 'db-username'
        password = os.getenv('db-password')  # Your project uses 'db-password'
        
        if not username or not password or not dsn:
            raise ValueError("Oracle credentials not found in environment variables. Expected: db-dsn, db-username, db-password")
        
        conn = oracledb.connect(user=username, password=password, dsn=dsn)
        logger.info("Oracle connection successful!")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to Oracle: {e}")
        return None


def safe_execute(conn, cursor, sql, params):
    """Execute a SQL statement safely with automatic reconnects."""
    if not ORACLE_AVAILABLE:
        raise ImportError("oracledb not available")
        
    while True:
        try:
            cursor.execute(sql, params)
            conn.commit()
            break
        except Exception as e:
            logger.error(f"DB error: {e}")
            logger.info(f"Retrying in {RETRY_DELAY}s...")
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
                logger.error("Reconnect failed, retrying again...")
    return conn, cursor


# Reddit Functions
def fetch_and_insert_posts_comments(reddit, subreddit_name, last_fetched_utc, conn, cursor, comments_max=50):
    """Fetch posts and comments and insert them immediately into the DB."""
    if not REDDIT_AVAILABLE:
        raise ImportError("praw not available. Please install it with: pip install praw")
    if not ORACLE_AVAILABLE:
        raise ImportError("oracledb not available")
        
    subreddit = reddit.subreddit(subreddit_name)
    max_utc = last_fetched_utc

    for post in subreddit.hot(limit=None):
        if post.created_utc <= last_fetched_utc:
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
def read_raw_from_db(conn) -> Tuple[DataFrameType, DataFrameType]:
    """
    Read posts and comments from Oracle using same pattern as your preprocessing script.
    Adjust queries if you want more rows.
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas not available. Please install it with: pip install pandas")
    
    query_posts = "SELECT * FROM historical_posts WHERE ROWNUM <= 100000"
    query_comments = "SELECT * FROM historical_comments WHERE ROWNUM <= 100000"

    logger.info("Reading posts and comments from DB...")
    df_posts = pd.read_sql_query(query_posts, conn)
    df_comments = pd.read_sql_query(query_comments, conn)
    logger.info("Read posts: %d rows, comments: %d rows", len(df_posts), len(df_comments))
    return df_posts, df_comments


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


def load_ticker_dictionary(filepath: str = TICKER_FILEPATH) -> DataFrameType:
    """
    Loads ticker CSV with columns: ticker, company_name (case-insensitive).
    The CSV must be prepared beforehand (you can export from Yahoo/other).
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas not available")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Ticker CSV not found at: {filepath}")
    
    tickers_df = pd.read_csv(filepath, dtype=str)
    tickers_df.columns = [c.strip().lower().replace(" ", "_") for c in tickers_df.columns]
    
    # Handle different possible column names for ticker symbol
    ticker_col_candidates = ["ticker", "act_symbol", "symbol", "actsymbol"]
    ticker_col = None
    for candidate in ticker_col_candidates:
        if candidate in tickers_df.columns:
            ticker_col = candidate
            break
    
    if ticker_col is None:
        raise ValueError(f"tickers CSV must contain one of these columns: {ticker_col_candidates}")
    
    # Rename to standard 'ticker' column
    if ticker_col != "ticker":
        tickers_df = tickers_df.rename(columns={ticker_col: "ticker"})
    
    tickers_df["ticker"] = tickers_df["ticker"].str.upper().str.strip()
    
    # optional company_name
    if "company_name" in tickers_df.columns:
        tickers_df["company_name"] = tickers_df["company_name"].astype(str)
    
    logger.info("Loaded %d tickers from %s", len(tickers_df), filepath)
    return tickers_df


def _is_json_source(path_or_url: str) -> bool:
    try:
        p = str(path_or_url).lower()
        return p.endswith('.json') or 'company_tickers.json' in p
    except Exception:
        return False


def load_ticker_dictionary_sec(source: str) -> DataFrameType:
    """
    Load tickers from the SEC JSON dataset at https://www.sec.gov/files/company_tickers.json
    Input can be a local file path or the SEC URL.
    Output columns: ticker (upper), company_name (title), cik_str (int)
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas not available")

    # Read JSON from local or URL
    def _read_json(src: str) -> Dict[str, Any]:
        if src.startswith('http://') or src.startswith('https://'):
            # Use urllib with a user-agent for SEC
            req = urllib.request.Request(src, headers={'User-Agent': 'Mozilla/5.0 (compatible; sentiment/1.0)'})
            with urllib.request.urlopen(req) as resp:
                data = resp.read()
            return json.loads(data)
        else:
            with open(src, 'r', encoding='utf-8') as f:
                return json.load(f)

    obj = _read_json(source)
    # The JSON is a dict of index -> {cik_str, ticker, title}
    df = pd.DataFrame.from_dict(obj, orient='index')
    # Normalize columns
    cols = {c: str(c).strip().lower() for c in df.columns}
    df = df.rename(columns=cols)
    if 'ticker' not in df.columns:
        raise ValueError("SEC JSON missing 'ticker' field")
    if 'title' in df.columns and 'company_name' not in df.columns:
        df = df.rename(columns={'title': 'company_name'})
    df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
    if 'company_name' in df.columns:
        df['company_name'] = df['company_name'].astype(str)
    # Deduplicate and keep only unique tickers
    df = df.dropna(subset=['ticker']).drop_duplicates(subset=['ticker']).reset_index(drop=True)
    logger.info("Loaded %d tickers from SEC JSON: %s", len(df), source)
    return df


def load_ticker_dictionary_auto(source: str) -> DataFrameType:
    """Auto-detect ticker source by extension/content and load appropriately."""
    if _is_json_source(source):
        return load_ticker_dictionary_sec(source)
    return load_ticker_dictionary(source)


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
    last_err: Optional[Exception] = None
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
    """Apply ticker detection to the text column."""
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


def maybe_lemmatize(df: DataFrameType) -> DataFrameType:
    """
    Optional: use spaCy to lemmatize sentiment_ready_text. Controlled by USE_SPACY.
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas not available")
    
    if not USE_SPACY or not SPACY_AVAILABLE or nlp is None:
        return df
    
    df = df.copy()
    
    def lemmatize_text(txt):
        if not isinstance(txt, str) or txt.strip() == "":
            return ""
        doc = nlp(txt)
        return " ".join([tok.lemma_ for tok in doc if not tok.is_punct and not tok.is_space])
    
    if TQDM_AVAILABLE:
        df["sentiment_ready_text"] = df["sentiment_ready_text"].progress_apply(lemmatize_text)
    else:
        df["sentiment_ready_text"] = df["sentiment_ready_text"].apply(lemmatize_text)
    
    return df


def run_enrichment_pipeline_memory(conn=None, tickers_source: str = "tickers.csv") -> DataFrameType:
    """
    Execute the full enrichment pipeline in-memory and return the cleaned DataFrame.
    No database writes are performed.
    
    Parameters
    ----------
    conn : optional, Oracle connection. If None, will open a new connection.
    tickers_source : path or URL to your ticker source (CSV or SEC JSON)
    
    Returns
    -------
    pd.DataFrame
        Cleaned, enriched dataframe ready for sentiment analysis.
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas not available")
    
    internal_conn = conn
    close_conn_when_done = False
    if internal_conn is None:
        internal_conn = get_oracle_connection()
        close_conn_when_done = True

    if internal_conn is None:
        raise ConnectionError("Cannot connect to Oracle DB. Aborting enrichment pipeline.")

    try:
        # Step 1: Read raw posts + comments
        query_posts = "SELECT * FROM historical_posts WHERE ROWNUM <= 5000"
        query_comments = "SELECT * FROM historical_comments WHERE ROWNUM <= 5000"

        logger.info("Reading posts and comments from DB...")
        df_posts = pd.read_sql_query(query_posts, internal_conn)
        df_comments = pd.read_sql_query(query_comments, internal_conn)
        logger.info("Posts: %d rows, Comments: %d rows", len(df_posts), len(df_comments))

        # Step 2: Harmonize schema
        df = harmonize_schema(df_posts, df_comments)

        # Step 3: Cleaning & filtering
        df = drop_invalid_texts(df)
        df = deduplicate_and_normalize_types(df)

        # Step 4: Enrichment - temporal & engagement
        df = add_temporal_features(df)
        df = add_engagement_features(df)

        # Step 5: Ticker detection
        tickers_df = load_ticker_dictionary_auto(tickers_source)
        df = apply_ticker_detection(df, tickers_df)

        # Optional: filter rows that mention tickers only
        # df = df[df['n_tickers'] > 0].reset_index(drop=True)

        # Step 6: Text normalization for sentiment
        df = apply_text_normalization(df, keep_tickers=True)
        df = maybe_lemmatize(df)

        # Step 7: Final housekeeping
        df = df.reset_index(drop=True)
        logger.info("Enrichment pipeline completed. Final rows: %d", len(df))

        return df

    finally:
        if close_conn_when_done and internal_conn is not None:
            try:
                internal_conn.close()
            except Exception:
                pass


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


def install_suggestions():
    """Print installation suggestions for missing dependencies."""
    suggestions = []
    
    if not PANDAS_AVAILABLE:
        suggestions.append("pip install pandas")
    if not NUMPY_AVAILABLE:
        suggestions.append("pip install numpy")
    if not TQDM_AVAILABLE:
        suggestions.append("pip install tqdm")
    if not ORACLE_AVAILABLE:
        suggestions.append("pip install oracledb")
    if not REDDIT_AVAILABLE:
        suggestions.append("pip install praw")
    if not DOTENV_AVAILABLE:
        suggestions.append("pip install python-dotenv")
    
    if suggestions:
        print("\nTo install missing dependencies, run:")
        print("-" * 40)
        for suggestion in suggestions:
            print(suggestion)
        print("\nOr install all at once:")
        print("pip install pandas numpy tqdm oracledb praw python-dotenv")
    else:
        print("\n✅ All dependencies are available!")


if __name__ == "__main__":
    print("Stock Sentiment Analysis Utils")
    print("=" * 50)
    check_dependencies()
    install_suggestions()
    
    # Test basic functions
    print("\nTesting basic functions...")
    try:
        # Test ticker detection
        test_text = "Check out $AAPL and GOOGL stocks today!"
        ticker_set = {"AAPL", "GOOGL", "MSFT"}
        result = detect_tickers_in_text(test_text, ticker_set)
        print(f"✅ Ticker detection: {result}")
        
        # Test text normalization
        normalized = normalize_text_for_sentiment(test_text, keep_tickers=True)
        print(f"✅ Text normalization: '{normalized}'")
        
        print("\n✅ Basic functions working correctly!")
        
    except Exception as e:
        print(f"❌ Error testing functions: {e}")





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





