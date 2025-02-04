import asyncio
import time
import pandas as pd
from beta_bot.helper import base as helper
from beta_bot.controller import base as controller
from datetime import datetime, timedelta
from beta_bot.database import base as database
from beta_bot.temp_storage import temp_storage_data, TempStorage
import ccxt.pro as ccxt

async def watch_ohlcv_info(exchange: ccxt.binance, symbol, timeframe, limit) -> pd.DataFrame:
    data = await exchange.watch_ohlcv(
        symbol,
        timeframe,
        limit=limit,
        params={
            'enableRateLimit': True,
            'price': 'index'
        },
    )
    df = pd.DataFrame(
        data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    return df


async def fetch_ohlcv_info(exchange: ccxt.binance, symbol, timeframe, limit) -> pd.DataFrame:
    # Fetch 10,000 candles in chunks of 1500 candles
    num_candles_to_fetch = limit
    candles_per_request = 1000
    index = 0
    dataframe = pd.DataFrame()
    since = None

    while index < num_candles_to_fetch:
        # Fetch candles for this chunk
        candles = await exchange.fetch_ohlcv(
            symbol, timeframe, limit=candles_per_request, since=since)
        # Convert candles to DataFrame
        df = pd.DataFrame(candles, columns=[
                          'date', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['date'], unit='ms')
        # Append the candles to the DataFrame
        dataframe = pd.concat([df, dataframe], ignore_index=True)
        # Update start index for the next request
        index += candles_per_request
        since = candles[0][0]-(60000*(candles_per_request))
    return dataframe

timeframe_conditions = {
    "1m": lambda current_time: (current_time.second > 1 & current_time.second < 45),
    "3m": lambda current_time: (current_time.second > 1 & current_time.second < 45) and current_time.minute % 3 == 0,
    "5m": lambda current_time: (current_time.second > 1 & current_time.second < 45) and current_time.minute % 5 == 0,
    "15m": lambda current_time: (current_time.second > 1 & current_time.second < 45) and current_time.minute % 15 == 0,
    "30m": lambda current_time: (current_time.second > 1 & current_time.second < 45) and current_time.minute % 30 == 0,
    "45m": lambda current_time: (current_time.second > 1 & current_time.second < 45) and current_time.minute % 45 == 0,
    "1h": lambda current_time: current_time.minute == 0 and (current_time.second > 1 & current_time.second < 45),
    "2h": lambda current_time: current_time.minute == 0 and (current_time.second > 1 & current_time.second < 45) and current_time.hour % 2 == 0,
    "3h": lambda current_time: current_time.minute == 0 and (current_time.second > 1 & current_time.second < 45) and current_time.hour % 3 == 0,
    "4h": lambda current_time: current_time.minute == 0 and (current_time.second > 1 & current_time.second < 45) and current_time.hour % 4 == 0,
    "10h": lambda current_time: current_time.minute == 0 and (current_time.second > 1 & current_time.second < 45) and current_time.hour % 5 == 0,
    "6h": lambda current_time: current_time.minute == 0 and (current_time.second > 1 & current_time.second < 45) and current_time.hour % 6 == 0,
    "8h": lambda current_time: current_time.minute == 0 and (current_time.second > 1 & current_time.second < 45) and current_time.hour % 8 == 0,
    "12h": lambda current_time: current_time.minute == 0 and (current_time.second > 1 & current_time.second < 45) and current_time.hour % 12 == 0,
}

current_trade_list = {}
async def fetch_data(exchange: ccxt.binance, symbol, timeframe, limit):
    cursor, conn = database.connectDB()
    temp_storage_data[TempStorage.cursor] = cursor
    temp_storage_data[TempStorage.conn] = conn
    temp_storage_data[TempStorage.exchange] = exchange
    temp_storage_data[TempStorage.highestProfitRoi][symbol] = 0
    current_trade_list[symbol] = []

    while True:
        _command_for_run = temp_storage_data[TempStorage.command_for_run]
        try:
            speed_counter_start = datetime.now()
            current_time = datetime.utcnow()
            temp_storage_data[TempStorage.conditionToAddNew][symbol] = timeframe_conditions.get(
                timeframe)

            # Initial Fetch Data
            if ((current_time.minute in [0])) or (f"order_book{symbol}" not in temp_storage_data):
                if (f"order_book{symbol}" not in temp_storage_data):
                    temp_storage_data[TempStorage.should_count_for_current][symbol] = False
                    temp_storage_data[TempStorage.dataframe][symbol] = await fetch_ohlcv_info(exchange, symbol, timeframe, limit)
            temp_storage_data[TempStorage.current_data][symbol] = await watch_ohlcv_info(exchange, symbol, timeframe, limit)
            temp_storage_data[f"order_book{symbol}"] = await exchange.watch_order_book(symbol)
            # When New Data is arrived add to the existing dataframe
            isDateChangeFromPrevious = temp_storage_data[TempStorage.dataframe][
                symbol].iloc[-1]['date'] != temp_storage_data[TempStorage.current_data][symbol].iloc[-1]['date']
            if (isDateChangeFromPrevious and temp_storage_data[TempStorage.should_count_for_current][symbol]):
                last_index = len(
                    temp_storage_data[TempStorage.dataframe][symbol])
                temp_storage_data[TempStorage.dataframe][symbol].loc[
                    last_index] = temp_storage_data[TempStorage.current_data][symbol].iloc[-1]
                temp_storage_data[TempStorage.dataframe][symbol].drop(
                    temp_storage_data[TempStorage.dataframe][symbol].index[0], inplace=True)
                temp_storage_data[TempStorage.dataframe][symbol].reset_index(
                    drop=True, inplace=True)
                length_for_predict = len(
                    temp_storage_data[TempStorage.dataframe][symbol]) - 1
                temp_storage_data[TempStorage.future_prediction][symbol] = controller.predict_future(
                    temp_storage_data[TempStorage.dataframe][symbol][:length_for_predict], symbol)
                temp_storage_data[TempStorage.should_count_for_current][symbol] = False
            elif not isDateChangeFromPrevious:
                temp_storage_data[TempStorage.dataframe][symbol].iloc[-1] = temp_storage_data[TempStorage.current_data][symbol].iloc[-1]
                temp_storage_data[TempStorage.should_count_for_current][symbol] = True
            else:
                temp_storage_data[TempStorage.dataframe][symbol].iloc[-1] = temp_storage_data[TempStorage.current_data][symbol].iloc[-1]
                temp_storage_data[TempStorage.should_count_for_current][symbol] = True

            # If Not Predicted, Then Predict
            if (not symbol in temp_storage_data[TempStorage.future_prediction]):
                length_for_predict = len(
                    temp_storage_data[TempStorage.dataframe][symbol]) - 1
                temp_storage_data[TempStorage.future_prediction][symbol] = controller.predict_future(
                    temp_storage_data[TempStorage.dataframe][symbol][:length_for_predict], symbol)

            dataframe:pd.DataFrame = temp_storage_data[TempStorage.dataframe][symbol]
            # Strategy Manager
            trade_datas = await controller.main_trade_func(dataframe, temp_storage_data[TempStorage.future_prediction], symbol)
            current_trade_list[symbol] = trade_datas
            speed_counter_stop = datetime.now()
            # helper.logger_test(
            #     temp_storage_data[TempStorage.future_prediction], speed_counter_start, speed_counter_stop)
            helper.logger(current_trade_list, speed_counter_start, speed_counter_stop)
        except Exception as e:
            print(f"Error in fetch_data: {e}")
            await asyncio.sleep(1)

async def fetch_multiple_coins() -> None:
    exchange = ccxt.binance()
    config = temp_storage_data[TempStorage.config]
    # exchange.set_sandbox_mode(True)
    
    leverage = config['trade_params']['leverage']
    exchange.options['defaultType'] = config['exchange']['type']
    timeframe = config['timeframe']
    limit = config['exchange']['ohlcv_candle_limit']
    symbols = config['exchange']['pair_whitelist']

    markets = await exchange.load_markets()
    tasks = []
    for symbol in symbols:
        temp_storage_data[TempStorage.tickSize][symbol] = float(markets[symbol]['info']['filters'][0]['tickSize'])
        temp_storage_data[TempStorage.dataframe][symbol] = pd.DataFrame()
        task = asyncio.create_task(fetch_data(
            exchange, symbol, timeframe, limit))
        tasks.append(task)

    await asyncio.gather(*tasks)
