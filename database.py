"""
database.py
------------
Conversation/diagnosis logging layer (Chapter 3.6 - Database Design).

The project report specifies MySQL for structured data. To keep this
runnable out-of-the-box without requiring a MySQL server to be installed,
this module defaults to SQLite (same schema, same SQL dialect subset) but
will transparently use MySQL if connection details are supplied via
environment variables. This mirrors common deployment practice: SQLite for
development/demo, MySQL for production.

Set these environment variables to use MySQL instead of SQLite:
    DB_ENGINE=mysql
    DB_HOST, DB_USER, DB_PASSWORD, DB_NAME
"""

import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQLITE_PATH = os.path.join(BASE_DIR, "data", "chatbot.db")

USE_MYSQL = os.environ.get("DB_ENGINE", "sqlite").lower() == "mysql"

if USE_MYSQL:
    import mysql.connector


def get_connection():
    if USE_MYSQL:
        return mysql.connector.connect(
            host=os.environ.get("DB_HOST", "localhost"),
            user=os.environ.get("DB_USER", "root"),
            password=os.environ.get("DB_PASSWORD", ""),
            database=os.environ.get("DB_NAME", "hardware_chatbot"),
        )
    return sqlite3.connect(SQLITE_PATH)


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    if USE_MYSQL:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(64) NOT NULL,
                user_message TEXT NOT NULL,
                fault_category VARCHAR(64),
                confidence FLOAT,
                method VARCHAR(32),
                source VARCHAR(16) DEFAULT 'text',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INT AUTO_INCREMENT PRIMARY KEY,
                conversation_id INT,
                was_helpful BOOLEAN,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dialogue_sessions (
                session_id VARCHAR(64) PRIMARY KEY,
                state_json TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
        """)
    else:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_message TEXT NOT NULL,
                fault_category TEXT,
                confidence REAL,
                method TEXT,
                source TEXT DEFAULT 'text',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                was_helpful INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dialogue_sessions (
                session_id TEXT PRIMARY KEY,
                state_json TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

    conn.commit()
    cur.close()
    conn.close()


def log_conversation(session_id, user_message, fault_category, confidence, method, source="text"):
    conn = get_connection()
    cur = conn.cursor()
    placeholder = "%s" if USE_MYSQL else "?"
    cur.execute(
        f"""INSERT INTO conversations
            (session_id, user_message, fault_category, confidence, method, source)
            VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})""",
        (session_id, user_message, fault_category, confidence, method, source),
    )
    conn.commit()
    new_id = cur.lastrowid
    cur.close()
    conn.close()
    return new_id


def log_feedback(conversation_id, was_helpful: bool):
    conn = get_connection()
    cur = conn.cursor()
    placeholder = "%s" if USE_MYSQL else "?"
    cur.execute(
        f"""INSERT INTO feedback (conversation_id, was_helpful)
            VALUES ({placeholder}, {placeholder})""",
        (conversation_id, int(was_helpful)),
    )
    conn.commit()
    cur.close()
    conn.close()


def get_recent_conversations(limit=20):
    conn = get_connection()
    cur = conn.cursor()
    placeholder = "%s" if USE_MYSQL else "?"
    cur.execute(
        f"""SELECT id, session_id, user_message, fault_category, confidence, method, created_at
            FROM conversations ORDER BY id DESC LIMIT {placeholder}""",
        (limit,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_fault_frequency():
    """Used for the analytics/admin view: counts how often each fault is diagnosed."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT fault_category, COUNT(*) as cnt
        FROM conversations
        WHERE fault_category IS NOT NULL
        GROUP BY fault_category
        ORDER BY cnt DESC
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_dialogue_state(session_id):
    """Returns the raw JSON string of saved state for a session, or None."""
    conn = get_connection()
    cur = conn.cursor()
    placeholder = "%s" if USE_MYSQL else "?"
    cur.execute(
        f"SELECT state_json FROM dialogue_sessions WHERE session_id = {placeholder}",
        (session_id,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row[0] if row else None


def save_dialogue_state(session_id, state_json):
    conn = get_connection()
    cur = conn.cursor()
    placeholder = "%s" if USE_MYSQL else "?"
    if USE_MYSQL:
        cur.execute(
            f"""INSERT INTO dialogue_sessions (session_id, state_json)
                VALUES ({placeholder}, {placeholder})
                ON DUPLICATE KEY UPDATE state_json = {placeholder}""",
            (session_id, state_json, state_json),
        )
    else:
        cur.execute(
            f"""INSERT INTO dialogue_sessions (session_id, state_json, updated_at)
                VALUES ({placeholder}, {placeholder}, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id) DO UPDATE SET
                    state_json = excluded.state_json,
                    updated_at = CURRENT_TIMESTAMP""",
            (session_id, state_json),
        )
    conn.commit()
    cur.close()
    conn.close()


def clear_dialogue_state(session_id):
    conn = get_connection()
    cur = conn.cursor()
    placeholder = "%s" if USE_MYSQL else "?"
    cur.execute(
        f"DELETE FROM dialogue_sessions WHERE session_id = {placeholder}",
        (session_id,),
    )
    conn.commit()
    cur.close()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at: {'MySQL' if USE_MYSQL else SQLITE_PATH}")
