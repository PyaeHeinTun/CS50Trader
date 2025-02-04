import pandas as pd
from beta_bot.machine_learning.core_ml import get_signal
from beta_bot.temp_storage import TempStorage,temp_storage_data

def predict_future(dataframe: pd.DataFrame, symbol) -> dict:
    df = dataframe.copy()

    current_trend , predicted_class , predicted_probability = get_signal(df,symbol)
    return {
        0 : {
            "index": 0,
            "class": predicted_class[0],
            "trend": current_trend,
            "probability": predicted_probability[0],
        },
        1 : {
            "index": 1,
            "class": predicted_class[1],
            "trend": current_trend,
            "probability": predicted_probability[1],
        }
    }
