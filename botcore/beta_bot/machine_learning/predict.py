# Step 1 Import
import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import TypeVar,Type,Callable
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
from beta_bot.temp_storage import temp_storage_data,TempStorage

from enum import Enum

class FeatureName(Enum):
    rsi = "RSI"
    wt = "WT"
    cci = "CCI"
    adx = "ADX"

def rescale(src, old_min, old_max, new_min, new_max) -> np.ndarray:
    return new_min + (new_max - new_min) * (src - old_min) / np.maximum((old_max - old_min), 10e-10)

def n_rsi(src, n1, n2) -> pd.Series:
    rsi = ta.rsi(src, n1)
    ema_rsi = ta.ema(rsi, n2)
    return rescale(ema_rsi, 0, 100, 0, 1)

def calculate_cci_improved(source1: pd.Series, source2: pd.Series, source3: pd.Series, length) -> np.ndarray:
    source1: np.ndarray = source1.to_numpy()
    source2: np.ndarray = source2.to_numpy()
    source3: np.ndarray = source3.to_numpy()
    windows = np.lib.stride_tricks.sliding_window_view(
        source1, window_shape=(length,))
    source3_modify = source3[:, np.newaxis][length-1:]
    sums = np.sum(np.abs(windows - source3_modify), axis=1)

    mad = sums / length

    mad_series = pd.Series(index=range(len(source1)), dtype=float)
    mad_series[-len(mad):] = mad
    mad_series = mad_series.to_numpy()
    source2_clean = np.nan_to_num(source2, nan=0.0)
    mad_series_clean = np.nan_to_num(mad_series, nan=0.0)
    mcci = np.divide(source2_clean, mad_series_clean, out=np.zeros_like(source2_clean), where=(mad_series_clean != 0))
    # mcci = source2/mad_series/0.015
    return mcci

def n_cci(dataframe, n1, n2) -> pd.Series:
    df = dataframe.copy()
    source = df['close']

    df['mas'] = ta.sma(source, n1)
    df['diffs'] = source - df['mas']
    df['cci'] = pd.Series(calculate_cci_improved(
        dataframe['open'], df['diffs'], df['mas'], n1))

    df['ema_cci'] = ta.ema(df['cci'], n2)

    normalized_wt_diff = pd.Series(normalize_optimized(0, 1, df['ema_cci']))
    return normalized_wt_diff

def normalize_optimized(min_val, max_val, source: pd.Series):
    source = source.to_numpy()
    historic_min = 10e10
    historic_max = -10e10
    # source.fillna(historic_min)
    src_filled_min = np.nan_to_num(source, historic_min)
    # source.fillna(historic_max)
    src_filled_max = np.nan_to_num(source, historic_max)
    historic_min = np.minimum.accumulate(src_filled_min)
    historic_max = np.maximum.accumulate(src_filled_max)
    division_value = np.maximum((historic_max - historic_min), 10e-10)
    normalized_src = (min_val + (max_val - min_val) *
                      (source - historic_min)) / division_value
    return normalized_src

def n_wt(src, n1, n2) -> np.ndarray:
    ema1 = ta.ema(src, n1)
    ema2 = ta.ema(np.abs(src - ema1), n1)
    ci = (src - ema1) / (0.015 * ema2)
    wt1 = ta.ema(ci, n2)
    wt2 = ta.sma(wt1, 4)
    diff = wt1 - wt2
    normalized_wt_diff = pd.Series(normalize_optimized(0, 1, pd.Series(diff)))
    return normalized_wt_diff

def calculate_tr_optimized(high, low, close) -> np.ndarray:
    previos_close = np.roll(close, 1)

    diff_h_n_l = high - low
    abs_value_h_n_c = np.abs(high - previos_close)
    abs_value_h_n_c[0] = abs(high[0] - 0)
    abs_value_l_n_c = np.abs(low - previos_close)
    abs_value_l_n_c[0] = abs(low[0] - 0)
    tr = np.maximum(np.maximum(diff_h_n_l, abs_value_h_n_c), abs_value_l_n_c)
    return tr

def calculate_directionalMovementPlus_optimized(high, low) -> np.ndarray:
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)

    diff_h_n_ph = high - prev_high
    diff_h_n_ph[0] = high[0] - 0
    diff_pl_n_l = prev_low - low
    diff_pl_n_l[0] = 0 - low[0]
    dmp_value = np.maximum(diff_h_n_ph, 0) * (diff_h_n_ph > diff_pl_n_l)
    return dmp_value

def calculate_negMovement_optimized(high, low) -> np.ndarray:
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)

    diff_h_n_ph = high - prev_high
    diff_h_n_ph[0] = high[0] - 0
    diff_pl_n_l = prev_low - low
    diff_pl_n_l[0] = 0 - low[0]
    negMovement = np.maximum(diff_pl_n_l, 0) * (diff_pl_n_l > diff_h_n_ph)
    return negMovement

def n_adx_optimized(highSrc: pd.Series, lowSrc: pd.Series, closeSrc: pd.Series, n1: int):
    length = n1
    highSrc_numpy = highSrc.to_numpy()
    lowSrc_numpy = lowSrc.to_numpy()
    closeSrc_numpy = closeSrc.to_numpy()

    tr = calculate_tr_optimized(highSrc_numpy, lowSrc_numpy, closeSrc_numpy)
    directionalMovementPlus = calculate_directionalMovementPlus_optimized(
        highSrc_numpy, lowSrc_numpy)
    negMovement = calculate_negMovement_optimized(highSrc_numpy, lowSrc_numpy)

    trSmooth = np.zeros_like(closeSrc_numpy)
    trSmooth[0] = np.nan
    for i in range(0, len(tr)):
        trSmooth[i] = trSmooth[i-1] - trSmooth[i-1] / length + tr[i]

    smoothDirectionalMovementPlus = np.zeros_like(closeSrc)
    smoothDirectionalMovementPlus[0] = np.nan
    for i in range(0, len(directionalMovementPlus)):
        smoothDirectionalMovementPlus[i] = smoothDirectionalMovementPlus[i-1] - \
            smoothDirectionalMovementPlus[i-1] / \
            length + directionalMovementPlus[i]

    smoothnegMovement = np.zeros_like(closeSrc)
    smoothnegMovement[0] = np.nan
    for i in range(0, len(negMovement)):
        smoothnegMovement[i] = smoothnegMovement[i-1] - \
            smoothnegMovement[i-1] / length + negMovement[i]

    diPositive = smoothDirectionalMovementPlus / trSmooth * 100
    diNegative = smoothnegMovement / trSmooth * 100
    dx = np.abs(diPositive - diNegative) / (diPositive + diNegative) * 100
    dx_series = pd.Series(dx)

    adx = dx_series.copy()
    adx.iloc[:length] = adx.rolling(length).mean().iloc[:length]
    adx = adx.ewm(alpha=(1.0/length), adjust=False).mean()
    return rescale(adx, 0, 100, 0, 1)

def chooseFeatureName(name: FeatureName, dataframe, paramsA, paramsB):
    df = dataframe.copy()
    source = df['open']
    hlc3 = (df['high'] + df['low'] + df['open']) / 3
    if (name == FeatureName.rsi.name):
        return n_rsi(source, paramsA, paramsB)
    if (name == FeatureName.wt.name):
        return n_wt(hlc3, paramsA, paramsB)
    if (name == FeatureName.cci.name):
        return n_cci(df, paramsA, paramsB)
    if (name == FeatureName.adx.name):
        return n_adx_optimized(df['high'], df['low'], df['open'], paramsA)

def highestvalue(_src,_len) -> np.ndarray:
    rolling_windows = np.lib.stride_tricks.sliding_window_view(_src, window_shape=(_len,))
    max_previous_value = np.max(rolling_windows, axis=1)
    max_previous_value = np.concatenate([np.full(_len-1, np.nan), max_previous_value])
    return max_previous_value

def lowestvalue(_src,_len) -> np.ndarray:
    rolling_windows = np.lib.stride_tricks.sliding_window_view(_src, window_shape=(_len,))
    max_previous_value = np.min(rolling_windows, axis=1)
    max_previous_value = np.concatenate([np.full(_len-1, np.nan), max_previous_value])
    return max_previous_value

def change_occurred(arr) -> np.ndarray:
    differences = np.diff(arr)
    differences = np.concatenate([np.full(1, 0), differences])
    return differences != 0

def set_h_l_value(high:np.ndarray,low:np.ndarray,direction:np.ndarray):
    price_now_1 = low[0]
    price_now_2 = high[0]
    price_now = low[0]

    price_index = 0
    price_index_1 = 0
    price_index_2 = 0

    price_index_array_1 = np.zeros_like(high)
    price_index_array_2 = np.zeros_like(high)
    price_index_array = np.zeros_like(high)

    for i in range(len(high)):
        if direction[i] != direction[i-1]:
            price_now_1 = price_now_2
            price_index_1 = price_index_2
            price_index_array_1[i] = price_index_1
            price_now_2 = price_now
            price_index_2 = price_index
            price_index_array_2[i] = price_index_2

        if direction[i] > 0:
            if (high[i] > price_now_2):
                price_now_2 = high[i]
                price_index_2 = i
                price_index_array_2[i] = price_index_2
                price_now = low[i]
                price_index = i
                price_index_array[i] = price_index

            if (low[i] < price_now):
                price_now = low[i]
                price_index = i
                price_index_array[i] = price_index

        if direction[i] < 0:
            if (low[i] < price_now_2):
                price_now_2 = low[i]
                price_index_2 = i
                price_index_array_2[i] = price_index_2
                price_now = high[i]
                price_index = i
                price_index_array[i] = price_index

            if (high[i] > price_now):
                price_now = high[i]
                price_index = i
                price_index_array[i] = price_index
    
    return price_index_array_1,price_index_array_2

def findBarSince(src: np.ndarray) -> np.ndarray:
    true_indices = np.where(src)[0]
    if len(true_indices) == 0:
        return np.zeros_like(src, dtype=np.int64)
    index_distance_array = np.full_like(src, fill_value=-1, dtype=np.int64)
    for i in range(len(true_indices)):
        if i == len(true_indices) - 1:
            index_distance_array[true_indices[i]:] = np.arange(len(src) - true_indices[i])
        else:
            index_distance_array[true_indices[i]:true_indices[i + 1]] = np.arange(true_indices[i + 1] - true_indices[i])
    index_distance_array[index_distance_array == -1] = 0
    return index_distance_array

def find_ytrain(z1:np.ndarray,z2:np.ndarray,direction:np.ndarray,high:np.ndarray,low:np.ndarray):
    cutted_z1 = z1[direction != np.roll(direction,1)]
    cutted_z2 = z2[direction != np.roll(direction,1)]
    cutted_direction = direction[direction != np.roll(direction,1)]

    iterator = zip(cutted_z1,cutted_direction)
    result = np.zeros_like(direction)
    previous_index = 0
    for i,value in enumerate(iterator):
        result[previous_index:int(value[0])+1] = value[1]
        
        previous_index = int(value[0])+1

    while previous_index < len(z1)-1:
        data_index_remainder = previous_index
        result_remainder = result[data_index_remainder:] 
        direction_remainder = direction[data_index_remainder:]
        high_remainder = high[data_index_remainder:]
        low_remainder = low[data_index_remainder:]

        max_index = np.argmax(high_remainder)+1
        min_index = np.argmin(low_remainder)+1

        if(result[data_index_remainder-1] == 1):
            result_remainder[:max_index] = -1
            result_remainder[max_index:] = 0
            result_remainder = np.roll(result_remainder,1)
            result_remainder[0] = -1
            result[data_index_remainder:]  = result_remainder
            previous_index = data_index_remainder + max_index
        if(result[data_index_remainder-1] == -1):
            result_remainder[:min_index] = 1   
            result_remainder[min_index:] = 0
            result_remainder = np.roll(result_remainder,1)
            result_remainder[0] = 1
            result[data_index_remainder:]  = result_remainder
            previous_index = data_index_remainder + min_index
        

    # 1 into -1 and -1 into 1 Conversion
    mask_1 = result == 1
    mask_minus1 = result == -1
    result[mask_1] = -1
    result[mask_minus1] = 1
    return result

def zigzagpp(_high,_low,depth,deviation,backstep,symbol) -> tuple:
    tick_size = temp_storage_data[TempStorage.tickSize][symbol]
    df = pd.DataFrame()
    df['high'] = pd.Series(_high)
    df['low'] = pd.Series(_low)

    df['highest'] = highestvalue(df['high'],depth)
    hr_condition = np.logical_not(((df['highest'] - df['high']) > (deviation * tick_size)).shift(1))
    hr_condition = np.array(hr_condition, dtype=np.float64)
    df['hr'] = findBarSince(hr_condition)

    df['lowest'] = lowestvalue(df['low'],depth)
    lr_condition = np.logical_not(((df['low'] - df['lowest']) > (deviation * tick_size)).shift(1))
    lr_condition = np.array(lr_condition, dtype=np.float64)
    df['lr'] = findBarSince(lr_condition)

    difference_of_hr_lr_condition = np.logical_not(df['hr'] > df['lr'])
    difference_of_hr_lr_condition = np.array(difference_of_hr_lr_condition,dtype=np.bool_)
    difference_of_hr_lr = findBarSince(difference_of_hr_lr_condition) >= backstep
    df['direction'] = np.where(difference_of_hr_lr,-1,1)

    price_index_1 , price_index_2 = set_h_l_value(df['high'].values,df['low'].values,df['direction'].values)

    price_index_1 = pd.Series(price_index_1)
    price_index_2 = pd.Series(price_index_2)

    return price_index_1,price_index_2,df['direction']

def label_market_trend(df: pd.DataFrame,symbol:str ,future_count: int = 2) -> pd.DataFrame:
    z1,z2,direction = zigzagpp(df['high'],df['low'],2,2,2,symbol)
    df['y_train'] = pd.Series(find_ytrain(z1.values,z2.values,direction.values,df['high'].values,df['low'].values)).shift(-1)
    df.loc[df['y_train'] == 0,['y_train']] = np.nan
    return df['y_train']

def extract_features(dataframe: pd.DataFrame, training_params,symbol:str):
    df = dataframe.copy()
    future_count = training_params['future_count']
    feature_count = training_params['feature_count']

    for i in range(1, feature_count+1):
        df[f'f{i}'] = chooseFeatureName(training_params[f'f{i}']['name'], df,
                                        training_params[f'f{i}']['paramsA'], training_params[f'f{i}']['paramsB'])
    df['y_train'] = label_market_trend(df=df,future_count=future_count,symbol=symbol)
    return df

def train_model(df,training_params) -> VotingClassifier:
    test_number = 0

    dataframe = df.copy()
    dataframe = dataframe
    dataframe.dropna(inplace=True)
    feature_count = training_params["feature_count"]+1
    feature_columns = [f'f{i}' for i in range(1, feature_count)]
    df_features = dataframe[feature_columns]
    df_y = dataframe['y_train']

    # X_train, X_test, y_train, y_test = train_test_split(df_features, df_y, test_size=0.001, random_state=42)
    X_train = df_features[:len(df_features)-test_number]
    X_test = df_features[len(df_features)-test_number:]
    y_train = df_y[:len(df_y)-test_number]
    y_test = df_y[len(df_y)-test_number:]
    # Build Model
    knn = KNeighborsClassifier(n_neighbors=1)
    cat_model = CatBoostClassifier(
        iterations=50,
        verbose=False,
        depth=2,
        learning_rate=0.01,
        loss_function='Logloss',
        rsm=0.95,
        border_count=64,
        eval_metric='AUC',
    )

    model = VotingClassifier(
        estimators=[
            ('knn', knn),
            ('cat', cat_model),
        ],
        voting='soft'
    )
    model.fit(X_train, y_train)
    # Predict Model
    # if test_number > 0:
    #     y_pred = model.predict(X_test)
    #     print(f"Accuracy Test : {accuracy_score(y_test,y_pred)}")
    #     print(classification_report(y_test, y_pred))
    return model

def fractalFilters(predict_value: pd.Series):
    isDifferentSignalType = predict_value.ne(predict_value.shift())
    return isDifferentSignalType

def predict_future(dataframe: pd.DataFrame, training_params) -> pd.DataFrame:
    df = dataframe.copy()

    df['predicted_value'], df['predicted_proba'] = train_model(
        df, training_params)
    df['isDifferentSignalType'] = fractalFilters(df['predicted_value'])

    dataframe['predicted_value'] = df['predicted_value']
    dataframe['predicted_proba'] = df['predicted_proba']
    dataframe['buy_signal'] = (df['predicted_value'] > 0) & (
        df['isDifferentSignalType'])
    dataframe['sell_signal'] = (df['predicted_value'] < 0) & (
        df['isDifferentSignalType'])

    return dataframe

T = TypeVar('T')

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs) -> None:
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class CoreML(metaclass=Singleton):
    """Core Machine Learning class implementing a singleton pattern."""
    def __init__(self,symbol:str) -> None:
        self.model = None
        self.symbol = symbol
        self.extracted_data = None

    def extract_features(self,df:pd.DataFrame,training_params: dict) -> pd.DataFrame:
        self.extracted_data = extract_features(dataframe=df, training_params=training_params,symbol=self.symbol)
        return self.extracted_data

    def train(self,df: pd.DataFrame, training_params: dict) -> None:
        """Train the model using the provided data and parameters."""
        self.extract_features(df,training_params)
        self.model = train_model(self.extracted_data, training_params)

        feature_count = training_params["feature_count"]+1
        feature_columns = [f'f{i}' for i in range(1, feature_count)]
        data_for_predict = self.extracted_data[feature_columns][-3:-1]
        return self.predict(data_for_predict)

    def predict(self,data_for_predit:pd.DataFrame) -> list:
        if not self.model:
            raise Exception("Must be train first.")
        current_predict_class = self.model.predict(data_for_predit)
        current_predict_probability = self.model.predict_proba(data_for_predit)
        current_predict_probability = np.amax(
        current_predict_probability, axis=1, keepdims=True)
        return current_predict_class