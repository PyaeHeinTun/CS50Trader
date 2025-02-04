from beta_bot.machine_learning.predict import CoreML
from beta_bot.machine_learning.trend import adaptiveTrendFinder_jit
import pandas as pd
from beta_bot.temp_storage import temp_storage_data,TempStorage

def get_signal(dataframe:pd.DataFrame,symbol:str) -> tuple:
    config = temp_storage_data[TempStorage.config]
    dataframe['trend'] = adaptiveTrendFinder_jit(dataframe,[2])
    predicted_class, predicted_probability = CoreML(symbol).train(dataframe,config['training_params'])

    # + for up - for down
    if (dataframe['trend'].iloc[-1] > 0):
        current_trend = 1 
    else:
        current_trend = -1
    return current_trend , predicted_class , predicted_probability