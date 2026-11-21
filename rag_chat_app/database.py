import sqlite3
import os
import hashlib
import hmac

DB_PATH = "rag_chat_app/app_data.db"

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT,
            api_key TEXT
        )
    ''')
    
    # Sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            agent_id TEXT DEFAULT 'default_agent',
            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Messages table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            role TEXT,
            content TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            memoize_status TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        )
    ''')
    
    conn.commit()
    conn.close()

PBKDF2_ITERATIONS = 200_000


def hash_password(password, salt=None):
    """Salted PBKDF2-HMAC-SHA256, stored as 'pbkdf2$<iterations>$<salt hex>$<hash hex>'."""
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password, stored):
    if stored.startswith("pbkdf2$"):
        _, iterations, salt_hex, digest_hex = stored.split("$")
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(iterations))
        return hmac.compare_digest(digest.hex(), digest_hex)
    # legacy unsalted SHA-256 rows (upgraded on next successful login)
    return hmac.compare_digest(hashlib.sha256(password.encode()).hexdigest(), stored)

def register_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def login_user(username, password):
    """Returns (id, username) or None."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, password_hash FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    user = None
    if row and verify_password(password, row[2]):
        user = (row[0], row[1])
        if not row[2].startswith("pbkdf2$"):
            cursor.execute("UPDATE users SET password_hash = ? WHERE id = ?", (hash_password(password), row[0]))
            conn.commit()
    conn.close()
    return user


def create_session(user_id, agent_id='default_agent'):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO sessions (user_id, agent_id) VALUES (?, ?)", (user_id, agent_id))
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id

def add_message(session_id, role, content, memoize_status=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO messages (session_id, role, content, memoize_status) VALUES (?, ?, ?, ?)", 
                   (session_id, role, content, memoize_status))
    conn.commit()
    conn.close()

def get_chat_history(session_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content, timestamp, memoize_status FROM messages WHERE session_id = ? ORDER BY timestamp ASC", 
                   (session_id,))
    history = cursor.fetchall()
    conn.close()
    return history
