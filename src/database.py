import sqlite3

DB = "predictions.db"


def init_db():
    """Create the table once if it doesn't exist."""
    conn = sqlite3.connect(DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT,
        category TEXT,
        urgency TEXT,
        confidence REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit()
    conn.close()


def log_prediction(text, category, urgency, confidence):
    conn = sqlite3.connect(DB)
    conn.execute(
        "INSERT INTO predictions(text,category,urgency,confidence) VALUES(?,?,?,?)",
        (text, category, urgency, confidence))
    conn.commit()
    conn.close()


def get_history(limit=20):
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT id, text, category, urgency, confidence, created_at "
        "FROM predictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return rows