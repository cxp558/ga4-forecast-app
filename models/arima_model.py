### models/arima_model.py
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def run_arima_forecast(df, target, forecast_days):
    df = df.sort_index().copy()

    model = ARIMA(df[target], order=(5, 1, 0))
    model_fit = model.fit()

    forecast_result = model_fit.get_forecast(steps=forecast_days)
    forecast_values = forecast_result.predicted_mean

    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({target: np.nan, 'forecast': forecast_values.values}, index=future_dates)

    df['forecast'] = np.nan
    return pd.concat([df, forecast_df])
