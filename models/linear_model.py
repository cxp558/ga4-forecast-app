### models/linear_model.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def run_linear_regression_forecast(df, target, forecast_days):
    df = df.sort_index().copy()
    df['time_index'] = np.arange(len(df))

    X = df[['time_index']]
    y = df[target]

    model = LinearRegression()
    model.fit(X, y)

    future_index = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)
    forecast_values = model.predict(future_index)

    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({target: np.nan, 'forecast': forecast_values}, index=future_dates)

    df['forecast'] = np.nan
    return pd.concat([df, forecast_df])
