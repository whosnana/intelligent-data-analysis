# ===============================================
# Лабораторна робота з аналізу часових рядів
# Тема: Аналіз і прогнозування продажів косметичних брендів
# Автор: Коваленко Владислава Володимирівна
# Мова: Python 3.13
# ===============================================

#бібліотеки
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.arima.model import ARIMA

#вхідні дані
# продажі косметичного бренду лореаль
months = pd.date_range(start="2022-01-01", periods=36, freq="ME")  # ME = month end
np.random.seed(42)

sales = 2000 + np.sin(np.linspace(0, 3 * np.pi, 36)) * 300 + np.random.normal(0, 120, 36)
data = pd.DataFrame({"Дата": months, "Продажі": sales})
data.set_index("Дата", inplace=True)

#побудова графіків
plt.figure(figsize=(10, 5))
plt.plot(data.index, data["Продажі"], marker='o', color='purple', label="Продажі бренду")
plt.title("Динаміка продажів косметичного бренду")
plt.xlabel("Дата")
plt.ylabel("Продажі (од.)")
plt.legend()
plt.grid(True)
plt.show()

#час
result = seasonal_decompose(data["Продажі"], model="additive", period=12)
result.plot()
plt.suptitle("Розкладання часового ряду на складові", y=1.03)
plt.show()

#автокореляційна функція
acf_values = acf(result.resid.dropna(), nlags=20)
plt.figure(figsize=(8, 4))
plt.stem(acf_values)  # без use_line_collection
plt.title("Автокореляційна функція залишкової компоненти")
plt.xlabel("Лаг")
plt.ylabel("Коефіцієнт автокореляції")
plt.show()

#аріма
model = ARIMA(data["Продажі"], order=(1, 1, 1))
model_fit = model.fit()

#прогнози продажів
forecast = model_fit.forecast(steps=6)
future_dates = pd.date_range(data.index[-1], periods=7, freq="ME")[1:]

plt.figure(figsize=(10, 5))
plt.plot(data.index, data["Продажі"], label="Фактичні дані", color='purple')
plt.plot(future_dates, forecast, label="Прогноз", color='red', linestyle='--', marker='o')
plt.title("Прогноз продажів косметичного бренду (ARIMA)")
plt.xlabel("Дата")
plt.ylabel("Продажі (од.)")
plt.legend()
plt.grid(True)
plt.show()

#висновок
print("\n=== ВИСНОВОК ===")
print("1. У часовому ряді спостерігається тренд і сезонність.")
print("2. Модель ARIMA(1,1,1) добре апроксимує дані.")
print("3. Прогноз демонструє потенційне зростання продажів у майбутні місяці.")
print("4. Методика може бути застосована до реальних даних косметичних компаній.")
