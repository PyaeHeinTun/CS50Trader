from typing import Self
from .db_helper import get_db
import time
from datetime import datetime,timedelta

class Trade:
    def __init__(self, pair, trade_id, entry_price, exit_price, is_short, updated_at, created_at, leverage, stake_ammount,quantity,is_completed) -> None:
        self.id = trade_id
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.is_short = is_short
        self.updated_at = updated_at
        self.created_at = created_at
        self.leverage = leverage
        self.stake_ammount = stake_ammount
        self.quantity = quantity
        self.pair = pair
        self.is_completed = is_completed

    def calculate_profit_ratio(self) -> dict:
        # entry_price, exit_price, is_short, leverage, stake_ammount
        quantity = (self.stake_ammount*self.leverage)/self.entry_price
        initial_margin = quantity * self.entry_price * (1/self.leverage)
        pnl = 0
        roi = 0
        if (self.is_short == False):
            pnl = (self.exit_price - self.entry_price) * quantity
        else:
            pnl = (self.entry_price - self.exit_price) * quantity

        roi = pnl / initial_margin
        return {
            "roi": round(roi * 100, 2),
            "pnl": round(pnl, 2)
        }
    
    @staticmethod
    def get_trades_for_n_days(days:int):        
        days_difference_in_seconds = int(time.time()) - (days * 24 * 60 * 60)
        trades = get_db().cursor().execute("SELECT * FROM trades WHERE updated_at>=? AND is_completed=1",(days_difference_in_seconds,)).fetchall()
        trades_list = [Trade.map_tuple_into_trade(data) for data in trades]
        return trades_list
    
    @staticmethod
    def get_open_trades() -> list:
        trades = get_db().cursor().execute("SELECT * FROM trades WHERE is_completed=0").fetchall()
        trades_list = [Trade.map_tuple_into_trade(data) for data in trades]
        return trades_list
    
    @staticmethod
    def get_trade_history(startTime:int,pair:str,page:int):
        page = page - 1
        print(page)
        offset = page * 5
        startDate = datetime.fromtimestamp(startTime).strftime("%Y-%m-%d")
        if pair=="All":
            trades = get_db().cursor().execute("SELECT *,DATE(updated_at, 'unixepoch') AS day FROM trades WHERE is_completed=1 AND day==? LIMIT 5 OFFSET ?",(startDate,offset,)).fetchall()
            total_trades = get_db().cursor().execute("SELECT COUNT(*),DATE(updated_at, 'unixepoch') AS day FROM trades WHERE is_completed=1 AND day==?",(startDate,)).fetchall()
        else:
            trades = get_db().cursor().execute("SELECT *,DATE(updated_at, 'unixepoch') AS day FROM trades WHERE is_completed=1 AND day==? AND pair=? LIMIT 5 OFFSET ?",(startDate,pair,offset,)).fetchall()
            total_trades = get_db().cursor().execute("SELECT COUNT(*),DATE(updated_at, 'unixepoch') AS day FROM trades WHERE is_completed=1 AND day==? AND pair=?",(startDate,pair,)).fetchall()

        trades_list = [Trade.map_tuple_into_trade(data) for data in trades]
        return trades_list,int(total_trades[0][0])

    @staticmethod
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
    
    def to_json(self) -> dict:
        return {
            "trade_id" : self.id,
            "pair" :  self.pair,
            "action" : "LONG" if self.is_short==0 else "SHORT",
            "entryPrice": self.entry_price,
            "exitPrice" : self.exit_price,
            "profitLoss" : self.calculate_profit_ratio()['pnl'],
            "profitLossClass" : "text-green-500" if self.calculate_profit_ratio()['roi'] > 0 else "text-red-500",
            "actionClass" :  "text-green-600 font-bold" if self.is_short==0 else "text-red-600 font-bold",
            "timestamp" :  datetime.fromtimestamp(int(self.created_at)).strftime("%Y-%m-%d %H:%M"),
        }