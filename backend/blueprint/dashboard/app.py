from collections import defaultdict
from flask import Blueprint, request, jsonify, render_template, redirect, url_for
from flask_jwt_extended import (
    create_access_token, jwt_required, get_jwt_identity,decode_token
)
from model.user_model import User
from model.trade_model import Trade
from datetime import datetime,timedelta
import pytz

app = Blueprint('dashboard_app', __name__)

def get_profit_for_each_day(day_list:list,trades_list:list[Trade]) -> list[float]:
    profits_by_date = defaultdict(int)
    for trade in trades_list:
        trade_date = datetime.utcfromtimestamp(int(trade.updated_at)).strftime('%Y-%m-%d')
        profits_by_date[trade_date] += trade.calculate_profit_ratio()['pnl']
    return [profits_by_date[day] for day in day_list]

def generate_day_list(current_date, days) -> list:
    return [(current_date - timedelta(days=days-1-i)).strftime('%Y-%m-%d') for i in range(days)]


@app.route("/chart",methods=['POST'])
def getChart() -> dict:
    if request.method == 'POST':
        token = request.form.get("token")
        if not token:
            return jsonify({"title":"Token is Empty","message": "Must provide token in form to check."}), 400 
        
        decoded_token = decode_token(token)
        username = decoded_token['sub']
        user = User.get_user_by_username(username)
        if not user:
            return jsonify({"title":"User could not find in database.","message": "Must provide valid token in form."}), 400 
        
        current_day = datetime.now()
        day_7_day_list = generate_day_list(current_day,7)
        day_7_trade_list = Trade.get_trades_for_n_days(days=7)
        day_7_trade_list_won = [trade for trade in day_7_trade_list if trade.calculate_profit_ratio()['roi']>0]
        day_7_profit_loss = sum([trade.calculate_profit_ratio()['pnl'] for trade in day_7_trade_list])

        day_14_day_list = generate_day_list(current_day,14)
        day_14_trade_list = Trade.get_trades_for_n_days(days=14)
        day_14_trade_list_won = [trade for trade in day_14_trade_list if trade.calculate_profit_ratio()['roi']>0]
        day_14_profit_loss = sum([trade.calculate_profit_ratio()['pnl'] for trade in day_14_trade_list])

        chartData = {
            '7days': {
                "labels": day_7_day_list,
                "data": get_profit_for_each_day(day_7_day_list,day_7_trade_list),
                "metrics" : {
                    "profitLoss": day_7_profit_loss,
                    "winRate": (len(day_7_trade_list_won)/len(day_7_trade_list))*100,
                    "totalTrades": len(day_7_trade_list),
                    "holdings": (user.balance + day_7_profit_loss) if (user.balance + day_7_profit_loss) > 0 else 0,
                },
            },
            '14days': {
                "labels": day_14_day_list,
                "data": get_profit_for_each_day(day_14_day_list,day_14_trade_list),
                "metrics" : {
                    "profitLoss": day_14_profit_loss,
                    "winRate": (len(day_14_trade_list_won)/len(day_14_trade_list))*100,
                    "totalTrades": len(day_14_trade_list),
                    "holdings": (user.balance + day_14_profit_loss) if (user.balance + day_14_profit_loss) > 0 else 0,
                },
            }
        }
        return jsonify({
            "chart" : chartData
        }),200

    return jsonify({"message": "Method not allowed."}), 404

@app.route("/open_trades",methods=["POST"])
def getOpenTrades() -> dict:
    if request.method == "POST":
        token = request.form.get("token")
        if not token:
            return jsonify({"title":"Token is Empty","message": "Must provide token in form to check."}), 400 
        
        trade_list = [trade.to_json() for trade in Trade.get_open_trades()]
        return jsonify({"open_trades":trade_list}),200
    return jsonify({"message": "Method not allowed."}), 404
