# Step 1 Import
import pandas as pd
import pandas_ta as ta
import numpy as np
import math
from numba import njit

from numba import njit
import numpy as np

@njit
def calculate_trend_direction_jit(index, close, periods) -> tuple:
    if index >= periods[-1]:
        devMultiplier = 2.0
        highestPearsonR = -np.inf
        detectedPeriod = 0
        detectedSlope = 0
        detectedIntrcpt = 0
        detectedStdDev = 0
        for period in periods:
            stdDev, pearsonR, slope, intercept = calc_dev_jit(period, close, index)
            if pearsonR > highestPearsonR:
                highestPearsonR = pearsonR
                detectedPeriod = period
                detectedSlope = slope
                detectedIntrcpt = intercept
                detectedStdDev = stdDev
        if highestPearsonR == -np.inf:
            raise Exception("Cannot find highest PearsonR")
        startPrice = np.exp(detectedIntrcpt + detectedSlope * (detectedPeriod - 1))
        endPrice = np.exp(detectedIntrcpt)
        trend_direction = endPrice - startPrice
        return trend_direction, detectedPeriod, highestPearsonR

    return 0, 0, 0


@njit
def calc_dev_jit(length, close, index) -> tuple:
    log_source = close[index + 1 - length: index + 1]
    indices = np.arange(1, length + 1)
    sum_x = np.sum(indices)
    sum_xx = np.sum(indices ** 2)
    sum_y = np.sum(log_source)
    sum_yx = np.sum(indices * log_source)

    denominator = length * sum_xx - sum_x ** 2
    if denominator == 0:
        return 0, 0, 0, 0

    slope = (length * sum_yx - sum_x * sum_y) / denominator
    average = sum_y / length
    intercept = average - (slope * sum_x / length) + slope

    deviations = log_source - intercept - slope * (indices - 1)
    sum_dev = np.sum(deviations ** 2)
    if length <= 1 or sum_dev == 0:
        std_dev = 0
    else:
        std_dev = np.sqrt(sum_dev / (length - 1))

    dxt = log_source - average
    dyt = slope * (indices - 1) - (intercept + slope * (length - 1) * 0.5)
    sum_dxx = np.sum(dxt ** 2)
    sum_dyy = np.sum(dyt ** 2)
    sum_dyx = np.sum(dxt * dyt)

    divisor = sum_dxx * sum_dyy
    if divisor <= 0:
        pearson_r = 0
    else:
        pearson_r = sum_dyx / np.sqrt(divisor)

    return std_dev, pearson_r, slope, intercept

def adaptiveTrendFinder_jit(df: pd.DataFrame, periods):
    close = np.log(df['close'].to_numpy())
    trend_directions = np.zeros(len(close))
    detected_periods = np.zeros(len(close))
    highest_pearson_rs = np.zeros(len(close))
    for i in range(len(close)):
        trend_direction, detected_period, highest_pearson_r = calculate_trend_direction_jit(i, close, periods)
        trend_directions[i] = trend_direction
        detected_periods[i] = detected_period
        highest_pearson_rs[i] = highest_pearson_r
    df['trend_direction'] = -trend_directions
    df['detected_period'] = detected_periods
    df['highest_pearson_r'] = highest_pearson_rs
    return df['trend_direction']