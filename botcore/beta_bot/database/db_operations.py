import sqlite3
from beta_bot.model.base import Trade
import time

def create_trade(cursor: sqlite3.Cursor, conn: sqlite3.Connection, entry_price, exit_price, is_short, created_at, updated_at, leverage, stake_ammount,quantity, pair, is_completed):
    cursor.execute("BEGIN TRANSACTION")
    cursor.execute("INSERT INTO trades (entry_price, exit_price,is_short,created_at,updated_at,leverage,stake_ammount,quantity,pair,is_completed) VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (entry_price, exit_price, is_short, created_at, updated_at, leverage, stake_ammount, quantity, pair, is_completed,))
    cursor.execute("COMMIT")
    conn.commit()

def find_completed_trade(cursor: sqlite3.Cursor, conn: sqlite3.Connection):
    cursor.execute("SELECT * FROM trades WHERE is_completed=1")
    rows = cursor.fetchall()
    return rows


def find_last_trade(cursor: sqlite3.Cursor, conn: sqlite3.Connection):
    cursor.execute("SELECT * FROM trades ORDER BY created_at DESC LIMIT 1")
    rows = cursor.fetchall()
    return rows


def update_pending_to_completed_trade(cursor: sqlite3.Cursor, conn: sqlite3.Connection, id):
    cursor.execute("BEGIN TRANSACTION")
    cursor.execute(
        "UPDATE trades SET is_completed = ?, updated_at = ? WHERE id = ?",
        (1, int(time.time()), id)
    )
    cursor.execute("COMMIT")
    conn.commit()
    return


def update_trade_current_price(cursor: sqlite3.Cursor, conn: sqlite3.Connection, current_price: float,pair:str):
    cursor.execute("BEGIN TRANSACTION")
    cursor.execute(
        "UPDATE trades SET exit_price=? WHERE is_completed = 0 AND pair=?", (current_price,pair,))
    cursor.execute("COMMIT")
    conn.commit()

def delete_trade(cursor: sqlite3.Cursor, conn: sqlite3.Connection, id):
    cursor.execute("BEGIN TRANSACTION")
    cursor.execute("DELETE FROM trades WHERE id = ?", (id,))
    cursor.execute("COMMIT")
    conn.commit()


def find_current_trade(cursor: sqlite3.Cursor, conn: sqlite3.Connection,pair:str):
    cursor.execute("SELECT * FROM trades WHERE is_completed=0 AND pair=?",(pair,))
    trades_list = cursor.fetchall()
    return trades_list

def map_tuple_into_trade(data: tuple):
    return Trade(
        trade_id=data[0],
        entry_price=data[1],
        exit_price=data[2],
        is_short=data[3],
        created_at=data[4],
        updated_at=data[5],
        leverage=data[6],
        stake_ammount=data[7],
        quantity=data[8],
        pair=data[9],
        is_completed=data[10],
    )
