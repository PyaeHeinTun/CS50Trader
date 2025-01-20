import sqlite3
from flask import g

DATABASE = 'database.db'

def get_db() -> sqlite3.Connection:
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

def create_necessary_table() -> None:
    get_db().cursor().execute("""CREATE TABLE IF NOT EXISTS users
                              (
                              id INTEGER PRIMARY KEY,
                              username TEXT NOT NULL,
                              password TEXT NOT NULL,
                              balance REAL NOT NULL DEFAULT 100.0,
                              created_at INTEGER NOT NULL
                              )
                              """)
    get_db().cursor().execute('''CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY,
                        entry_price REAL,
                        exit_price REAL,
                        is_short INTEGER,
                        created_at TEXT,
                        updated_at TEXT,
                        leverage INTEGER,
                        stake_ammount REAL,
                        quantity INTEGER,
                        pair TEXT,
                        is_completed INTEGER
                    )''')
