import os
import time
import oracledb
import praw

RETRY_DELAY = 10  # 

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

def fetch_and_insert_posts_comments(reddit, subreddit_name, last_fetched_utc, conn, cursor, comments_max=50):
    """Fetch posts and comments and insert them immediately into the DB."""
    subreddit = reddit.subreddit(subreddit_name)
    max_utc = last_fetched_utc

    for post in subreddit.hot(limit=None):
        if post.created_utc <= last_fetched_utc:
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