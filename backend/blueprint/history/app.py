from collections import defaultdict
from flask import Blueprint, request, jsonify, render_template, redirect, url_for
from flask_jwt_extended import (
    create_access_token, jwt_required, get_jwt_identity,decode_token
)
from model.user_model import User
from model.trade_model import Trade
from datetime import datetime,timedelta
import json
import pytz
import math

app = Blueprint('history_app', __name__)

@app.route("pair_list")
def getPairList() -> dict:
    coin_list = ["All"]
    coin_list = coin_list + read_config()['exchange']['pair_whitelist']
    return jsonify({"pair_list" : coin_list})

@app.route("trade_history",methods=["POST"])
def getTradeHistory() -> dict:
    if request.method == "POST":
        token = request.form.get("token")
        selectDate = request.form.get("selectDate")
        tradingPair = request.form.get("tradingPair")
        page = request.form.get("page","1")

        if not token or (selectDate == "null") or not tradingPair or not page:
            return jsonify({"title":"Form Input is Empty","message": "Must provide Input in form to check."}), 400 

        try:
            trade_list,total_trades = Trade.get_trade_history(int(selectDate),tradingPair,int(page))
            trade_list = [trade.to_json() for trade in trade_list]
        except:
            return jsonify({"title":"Success","message": "Something went wrong."}), 404
        return jsonify(
            {
                "title":"Success",
                "message":"Successfully rendered",
                "history":trade_list,
                "pagination": {
                    "currentPage" : int(page),
                    "hasNextPage" : int(page) < math.ceil(total_trades/5),
                    "hasPreviousPage" : int(page) > 1,
                    "totalPage" : 1 if math.ceil(total_trades/5) == 0 else math.ceil(total_trades/5),
                }
            }),200
    return jsonify({"title":"Success","message": "Method not allowed."}), 404

def read_config() -> dict:
    with open('botcore/config.json') as f:
        config_data = json.load(f)
    return config_data