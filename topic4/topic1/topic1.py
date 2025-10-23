import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import ExponentialSmoothing


months = pd.date_range(start="2022-01-01", periods=36, freq="M")
np.random.seed(42)


trend = np.linspace(100, 160, 36)
season = 10 * np.sin(2 * np.pi * np.arange(36) / 12)
noise = np.random.normal(0, 4, 36)
sales = trend + season + noise

data = pd.DataFrame({"Month": months, "Sales": sales})
data.set_index("Month", inplace=True)


plt.figure(figsize=(10,5))
plt.plot(data.index, data["Sales"], marker='o')
plt.title("Динамічний часовий ряд продажів косметичного бренду")
plt.xlabel("Місяць")
plt.ylabel("Продажі (тис. грн)")
plt.grid(True)
plt.show()


decomp = seasonal_decompose(data["Sales"], model='additive', period=12)
decomp.plot()
plt.suptitle("Декомпозиція часового ряду (Trend, Seasonality, Residuals)")
plt.show()



data["MA"] = data["Sales"].rolling(window=3, center=True).mean()


model_es = ExponentialSmoothing(data["Sales"], trend="add", seasonal="add", seasonal_periods=12)
fit_es = model_es.fit()
data["Forecast"] = fit_es.fittedvalues


plt.figure(figsize=(10,5))
plt.plot(data.index, data["Sales"], label="Фактичні дані", alpha=0.6)
plt.plot(data.index, data["MA"], label="Ковзне середнє", linewidth=2)
plt.plot(data.index, data["Forecast"], label="Експоненційне згладжування", linewidth=2, linestyle='--')
plt.title("Згладжування часового ряду продажів бренду")
plt.xlabel("Місяць")
plt.ylabel("Продажі (тис. грн)")
plt.legend()
plt.show()


forecast = fit_es.forecast(6)
plt.figure(figsize=(10,5))
plt.plot(data.index, data["Sales"], label="Історичні дані")
plt.plot(pd.date_range(data.index[-1], periods=7, freq="M")[1:], forecast, label="Прогноз", color="red")
plt.title("Прогноз продажів косметичного бренду на 6 місяців вперед")
plt.xlabel("Місяць")
plt.ylabel("Продажі (тис. грн)")
plt.legend()
plt.show()
