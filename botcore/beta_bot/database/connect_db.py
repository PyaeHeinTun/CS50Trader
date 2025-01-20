import sqlite3

def create_necessary_table(cursor) -> None:
    cursor.execute("""CREATE TABLE IF NOT EXISTS users
                              (
                              id INTEGER PRIMARY KEY,
                              username TEXT NOT NULL,
                              password TEXT NOT NULL,
                              balance REAL NOT NULL DEFAULT 0.0,
                              created_at INTEGER NOT NULL
                              )
                              """)
    cursor.execute('''CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY,
                        entry_price REAL,
                        exit_price REAL,
                        is_short INTEGER,
                        created_at INTEGER,
                        updated_at INTEGER,
                        leverage INTEGER,
                        stake_ammount REAL,
                        quantity INTEGER,
                        pair TEXT,
                        is_completed INTEGER
                    )''')


def connectDB() -> tuple:
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    # Create a table
    create_necessary_table(cursor)

    return cursor, conn

def closeDB(cursor: sqlite3.Cursor, conn: sqlite3.Connection):
    cursor.close()
    conn.close()
