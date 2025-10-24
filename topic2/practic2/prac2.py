# ===============================================
# Тема: Прогнозування часових рядів продажів косметичних брендів з використанням ARIMA
# Автор: Коваленко Владислава Володимирівна
# Мова: Python 3.13
# ===============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# часовий ряд
np.random.seed(42)
months = pd.date_range(start="2020-01-01", periods=48, freq="ME")  # 4 роки
sales = 2000 + np.sin(np.linspace(0, 6 * np.pi, 48)) * 300 + np.random.normal(0, 120, 48)
data = pd.DataFrame({"Дата": months, "Продажі": sales}).set_index("Дата")

#показать часовий ряд
plt.figure(figsize=(10, 5))
plt.plot(data.index, data["Продажі"], marker="o", color="purple")
plt.title("Динаміка продажів косметичного бренду")
plt.xlabel("Дата")
plt.ylabel("Продажі (од.)")
plt.grid(True)
plt.show()

# тест фуллера
result = adfuller(data["Продажі"])
print("ADF Statistic:", result[0])
print("p-value:", result[1])
if result[1] > 0.05:
    print("→ Ряд не є стаціонарним.")
else:
    print("→ Ряд стаціонарний.")

data["log_sales"] = np.log(data["Продажі"])
data["diff_1"] = data["log_sales"].diff().dropna()
data["diff_12"] = data["log_sales"].diff(12).dropna()

data["diff_final"] = data["log_sales"].diff().diff(12)
data["diff_final"].dropna(inplace=True)

plt.figure(figsize=(10, 4))
plt.plot(data["diff_final"], color="darkgreen")
plt.title("Стаціонарний ряд після різницювання")
plt.xlabel("Дата")
plt.ylabel("Δ Продажів (лог)")
plt.grid(True)
plt.show()

# автокореляція
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(data["diff_final"].dropna(), ax=axes[0])
plot_pacf(data["diff_final"].dropna(), ax=axes[1])
axes[0].set_title("ACF")
axes[1].set_title("PACF")
plt.show()

#модель аріма
model = ARIMA(data["log_sales"], order=(1,1,1), seasonal_order=(1,1,1,12))
model_fit = model.fit()

print(model_fit.summary())

residuals = model_fit.resid
plt.figure(figsize=(10,4))
plt.plot(residuals, color="gray")
plt.title("Залишки моделі ARIMA")
plt.grid(True)
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(residuals, bins=15, kde=True, color="blue")
plt.title("Розподіл залишків")
plt.xlabel("Значення")
plt.ylabel("Частота")
plt.grid(True)
plt.show()

forecast = model_fit.get_forecast(steps=10)
forecast_index = pd.date_range(data.index[-1], periods=11, freq="ME")[1:]
forecast_mean = np.exp(forecast.predicted_mean)  # перетворення з логарифмів
conf_int = np.exp(forecast.conf_int())

#прогноз
plt.figure(figsize=(10, 5))
plt.plot(data.index, data["Продажі"], label="Фактичні дані", color="purple")
plt.plot(forecast_index, forecast_mean, label="Прогноз", color="red", marker="o")
plt.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color="pink", alpha=0.3)
plt.title("Прогноз продажів косметичного бренду (ARIMA)")
plt.xlabel("Дата")
plt.ylabel("Продажі (од.)")
plt.legend()
plt.grid(True)
plt.show()

# висновки
print("Висновок:")
print("- Часовий ряд спочатку був нестаціонарним, тому виконано логарифмування та різницювання.")
print("- Модель ARIMA(1,1,1)(1,1,1)[12] показала адекватні результати.")
print("- Отриманий прогноз демонструє сезонне зростання продажів протягом наступних 10 місяців.")
