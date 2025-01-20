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
