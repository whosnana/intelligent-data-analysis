import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing

def forecast_sales(df: pd.DataFrame, brand: str, periods_ahead: int = 3):
    bdf = df[df['brand'] == brand].copy()
    bdf = bdf.set_index('date').sort_index()

    ts = bdf['sales']

    model = ExponentialSmoothing(
        ts,
        trend='add',
        seasonal=None,
        initialization_method="estimated"
    ).fit()

    forecast = model.forecast(periods_ahead)

    plt.figure(figsize=(8,4))
    plt.plot(ts.index, ts.values, label='Факт')
    plt.plot(forecast.index, forecast.values, 'r--', label='Прогноз')
    plt.title(f'Прогноз продажів бренду {brand}')
    plt.xlabel('Місяць')
    plt.ylabel('Продажі')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return forecast
