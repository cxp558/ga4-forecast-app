# models/exp_smoothing_model.py

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def run_exp_smoothing_forecast(df, target='sessions', forecast_days=90):
    """
    Fits Exponential Smoothing model and forecasts the specified number of days.
    Returns combined historical + forecast DataFrame.
    """
    df = df.sort_index().copy()

    model = ExponentialSmoothing(df[target], trend='add', seasonal=None)
    model_fit = model.fit()

    forecast_values = model_fit.forecast(forecast_days)

    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

    forecast_df = pd.DataFrame({
        target: np.nan,
        'forecast': forecast_values.values
    }, index=future_dates)

    df['forecast'] = np.nan
    combined_df = pd.concat([df, forecast_df])

    return combined_df
