import pandas as pd
from beta_bot.model.base import Trade
from datetime import datetime
from beta_bot.database import base as database
from beta_bot.helper import base as helper
import sqlite3
import time
import ccxt
from beta_bot.temp_storage import TempStorage, temp_storage_data

async def main_trade_func(dataframe: pd.DataFrame, future_predictions: dict, pair: str) -> list[Trade]:
    cursor: sqlite3.Cursor = temp_storage_data[TempStorage.cursor]
    conn: sqlite3.Connection = temp_storage_data[TempStorage.conn]

    leverage = temp_storage_data[TempStorage.config]['trade_params']['leverage']
    stake_ammount = temp_storage_data[TempStorage.config]['trade_params']['stake_ammount']
    current_price = dataframe.iloc[-1].squeeze()['close']
    trades_list = _find_existing_trade(cursor, conn,pair)
    future_predictions = future_predictions[pair]

    # If current trades exits
    if (len(trades_list) != 0):
        _update_current_price_in_trade(cursor, conn, current_price,pair)

        for trade in trades_list:
            if (_custom_exit(dataframe, trade, future_predictions) & (trade.pair == pair)):
                await _close_position(cursor, conn, trade)

        return trades_list

    else:
        signal, should_trade = _populate_trade_entry(
            dataframe, future_predictions)
        if (should_trade):
            quantity = (stake_ammount * leverage) / current_price
            new_trade = _get_trade_object(
                signal, current_price, leverage, pair, stake_ammount,quantity)
            await _create_position(cursor, conn, new_trade)
        return trades_list


def _custom_exit(dataframe: pd.DataFrame, trade: Trade, future_predictions: dict) -> bool:
    # Calculate trade metrics
    current_profit = trade.calculate_profit_ratio()
    current_profit_roi = current_profit["roi"]
    current_price = dataframe.iloc[-1]["close"]
    trade_open_rate = trade.entry_price

    # Initialize exit reason
    exit_reason = None

    # **2. Signal-Based Exits**
    should_exit = should_exit_trade(future_predictions, trade.is_short)
    if should_exit:
        if current_profit_roi > 0:
            exit_reason = "signal_exit_profit"  # Exit based on signal while in profit
        else:
            exit_reason = "signal_exit_loss"  # Exit based on signal while in loss

    # Confirm exit if a reason is identified
    return _confirm_trade_exit(exit_reason)


def should_exit_trade(indices: dict, is_short: int) -> bool:
    # For long positions (is_short == 0)
    if is_short == 0:
        consistent_exit_signals = (
            ((indices[0]['class'] == -1) and (indices[1]['class'] == -1))
            or
            (indices[0]['trend'] == -1)
        )
    # For short positions (is_short == 1)
    elif is_short == 1:
        consistent_exit_signals = (
            ((indices[0]['class'] == 1) and (indices[1]['class'] == 1))
            or
            (indices[0]['trend'] == 1)
        )

    # Exit if any of the signals match the condition for exiting the trade
    return consistent_exit_signals

def _confirm_trade_exit(exit_reason: str) -> bool:
    valid_exit_reasons = {"trailing_stop", "signal_exit_profit", "signal_exit_loss"}
    return exit_reason in valid_exit_reasons

    
def _populate_trade_entry(dataframe: pd.DataFrame, future_predictions: dict) -> bool:
    cursor: sqlite3.Cursor = temp_storage_data[TempStorage.cursor]
    conn: sqlite3.Connection = temp_storage_data[TempStorage.conn]

    # return return ("none", False)
    return should_enter_trade(future_predictions)

def should_enter_trade(indices: dict) -> bool:
    # Evaluate all indices for long and short signals
    consistent_buy_signals = all(data["class"] == 1 for data in indices.values())
    consistent_up_trend = all(data["trend"] == 1 for data in indices.values())

    consistent_sell_signals = all(data["class"] == -1 for data in indices.values())
    consistent_down_trend = all(data["trend"] == -1 for data in indices.values())

    if consistent_buy_signals and consistent_up_trend:
        return (0,True)
    if consistent_sell_signals and consistent_down_trend:
        return (1,True)
    return ("none",False)

async def _create_position(cursor: sqlite3.Cursor, conn: sqlite3.Connection, trade: Trade):
    order_book = temp_storage_data[f'order_book{trade.pair}']

    if (trade.is_short == 1):
        database.create_trade(cursor, conn, order_book['bids'][0][0], trade.exit_price,
                                1, trade.created_at, trade.updated_at, trade.leverage, trade.stake_ammount,trade.quantity, trade.pair, trade.is_completed)
    else:
        database.create_trade(cursor, conn, order_book['asks'][0][0], trade.exit_price,
                                0, trade.created_at, trade.updated_at, trade.leverage, trade.stake_ammount,trade.quantity, trade.pair, trade.is_completed)
    return trade


async def _close_position(cursor: sqlite3.Cursor, conn: sqlite3.Connection, trade: Trade):
    is_dry_run = temp_storage_data[TempStorage.config]['dry_run']
    exchange: ccxt.binance = temp_storage_data[TempStorage.exchange]

    database.update_pending_to_completed_trade(cursor, conn, trade.id)
    return trade


def _update_current_price_in_trade(cursor: sqlite3.Cursor, conn: sqlite3.Connection, current_price: float,pair:str):
    database.update_trade_current_price(cursor, conn, current_price,pair)
    return


def _find_existing_trade(cursor: sqlite3.Cursor, conn: sqlite3.Connection,pair:str):
    trades_list = database.find_current_trade(cursor, conn,pair)

    trades_list = [database.map_tuple_into_trade(
        value[0]) for value in zip(trades_list)]

    return trades_list

def _get_last_trade(cursor: sqlite3.Cursor, conn: sqlite3.Connection):
    trades_list = database.find_last_trade(cursor, conn)

    trades_list = [database.map_tuple_into_trade(
        value[0]) for value in zip(trades_list)]

    return trades_list

def _get_trade_object(is_short: int, current_price, leverage, pair, stake_ammount,quantity) -> Trade:
    return Trade(
        trade_id=0,
        updated_at=int(time.time()),
        created_at=int(time.time()),
        entry_price=current_price,
        exit_price=current_price,
        is_short=is_short,
        leverage=leverage,
        pair=pair,
        stake_ammount=stake_ammount,
        quantity=quantity,
        is_completed=0,
    )
