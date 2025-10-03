import time
import os
import oracledb
from dotenv import load_dotenv

BATCH_SIZE = 500
RETRY_DELAY = 10

def get_oracle_connection():
    """Get a new Oracle DB connection from environment variables."""
    try:
        load_dotenv()
        cs = os.getenv("db-dsn")
        connection = oracledb.connect(
            user=os.getenv("db-username"),
            password=os.getenv("db-password"),
            dsn=cs
        )
        print("Connection successful!")
        return connection
    except oracledb.DatabaseError as e:
        print(f"Error connecting to Oracle DB: {e}")
        return None

def get_subreddit_posts_and_comments(reddit, subreddit_name, last_fetched_utc, limit=None, comments_max=50):
    """Fetch posts and comments from a subreddit that are newer than last_fetched_utc."""
    subreddit = reddit.subreddit(subreddit_name)
    posts_data = []
    comments_data = []

    for post in subreddit.hot(limit=limit):
        if post.created_utc <= last_fetched_utc:
            continue

        post_info = {
            "author": str(post.author) if post.author else None,
            "title": post.title,
            "created_utc": post.created_utc,
            "id": post.id,
            "is_original_content": post.is_original_content,
            "score": post.score,
            "selftext": post.selftext,
            "subreddit": str(post.subreddit),
            "upvote_ratio": post.upvote_ratio,
            "url": post.url
        }
        posts_data.append(post_info)
        time.sleep(1)

        post.comments.replace_more(limit=0)
        count = 0
        for comment in post.comments.list():
            if count >= comments_max:
                break
            if comment.created_utc <= last_fetched_utc:
                continue

            comment_info = {
                "author": str(comment.author) if comment.author else None,
                "created_utc": comment.created_utc,
                "id": comment.id,
                "parent_post_id": post.id,
                "score": comment.score,
                "selftext": comment.body,
                "subreddit": str(post.subreddit)
            }
            comments_data.append(comment_info)
            count += 1
            time.sleep(2)

    return posts_data, comments_data

def insert_in_batches(conn, cursor, sql, rows, batch_size=BATCH_SIZE):
    """Insert rows in batches with automatic reconnect if DB drops."""
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        while True:
            try:
                cursor.executemany(sql, batch)
                conn.commit()
                print(f"Inserted {len(batch)} rows (up to {i+len(batch)} total)")
                break
            except oracledb.DatabaseError as e:
                print(f"Error inserting batch: {e}")
                print(f"Attempting to reconnect in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
                new_conn = get_oracle_connection()
                if new_conn:
                    conn = new_conn
                    cursor = conn.cursor()
                else:
                    print("Reconnect failed, retrying after delay...")