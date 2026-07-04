import sqlite3
from datetime import datetime

DB_NAME = "isl_learning.db"

# =============================
# DATABASE CONNECTION
# =============================
def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

# =============================
# INITIALIZE DATABASE
# =============================
def init_db():
    conn = get_connection()
    c = conn.cursor()

    # USERS TABLE
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'user'
    )
    """)

    # RESULTS TABLE
    c.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        gesture TEXT,
        predicted TEXT,
        confidence REAL,
        score REAL,
        correctness TEXT,
        timestamp TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)

    conn.commit()

    # Create default admin if not exists
    c.execute("SELECT * FROM users WHERE username = ?", ("admin",))
    admin_exists = c.fetchone()

    if not admin_exists:
        c.execute("""
        INSERT INTO users (username, password, role)
        VALUES (?, ?, ?)
        """, ("admin", "admin123", "admin"))
        conn.commit()

    conn.close()

# =============================
# USER AUTH FUNCTIONS
# =============================
def register_user(username, password):
    conn = get_connection()
    c = conn.cursor()

    try:
        c.execute("""
        INSERT INTO users (username, password, role)
        VALUES (?, ?, ?)
        """, (username, password, "user"))
        conn.commit()
        conn.close()
        return True, "User registered successfully!"
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Username already exists."

def login_user(username, password):
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
    SELECT id, username, role FROM users
    WHERE username = ? AND password = ?
    """, (username, password))

    user = c.fetchone()
    conn.close()
    return user

# =============================
# SAVE RESULT
# =============================
def save_result(user_id, gesture, predicted, confidence, score, correctness):
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
    INSERT INTO results (user_id, gesture, predicted, confidence, score, correctness, timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        gesture,
        predicted,
        confidence,
        score,
        correctness,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()

# =============================
# USER-SPECIFIC RESULTS
# =============================
def get_results(user_id):
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
    SELECT * FROM results
    WHERE user_id = ?
    ORDER BY id DESC
    """, (user_id,))

    rows = c.fetchall()
    conn.close()
    return rows

def get_average_score(user_id):
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
    SELECT AVG(score) FROM results
    WHERE user_id = ?
    """, (user_id,))

    avg = c.fetchone()[0]
    conn.close()
    return round(avg, 2) if avg else 0

def get_completed_gestures(user_id):
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
    SELECT DISTINCT gesture FROM results
    WHERE user_id = ? AND correctness = 'Correct'
    """, (user_id,))

    rows = c.fetchall()
    conn.close()
    return [r[0] for r in rows]

# =============================
# ADMIN FUNCTIONS
# =============================
def get_all_users():
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
    SELECT id, username, role FROM users
    ORDER BY id ASC
    """)

    rows = c.fetchall()
    conn.close()
    return rows

def get_all_results():
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
    SELECT results.id, users.username, results.gesture, results.predicted,
           results.confidence, results.score, results.correctness, results.timestamp
    FROM results
    JOIN users ON results.user_id = users.id
    ORDER BY results.id DESC
    """)

    rows = c.fetchall()
    conn.close()
    return rows

# =============================
# INIT DB
# =============================
init_db()