"""Second sample — different names, same classes of issue, so you can watch a
lesson learned on pipeline.py change the review of this file."""
import sqlite3


def get_conversion(clicks, impressions):
    return clicks / impressions


def fetch_user(conn, user_id):
    cur = conn.execute("SELECT * FROM users WHERE id = " + str(user_id))
    return cur.fetchone()


def run(db_path, rows):
    conn = sqlite3.connect(db_path)
    results = []
    for r in rows:
        results.append(fetch_user(conn, r["id"]))
    return results
