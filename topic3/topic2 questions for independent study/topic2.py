import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from scipy.signal import savgol_filter


np.random.seed(42)
months = pd.date_range(start="2023-01-01", periods=24, freq="ME")
trend = np.linspace(100, 180, 24)
seasonal = 15 * np.sin(np.linspace(0, 3 * np.pi, 24))
noise = np.random.normal(0, 5, 24)
sales = trend + seasonal + noise
data = pd.DataFrame({"Month": months, "Sales": sales}).set_index("Month")


plt.figure(figsize=(8, 4))
plt.plot(data.index, data["Sales"], marker="o", color="teal")
plt.title("Динаміка продажів косметичного бренду")
plt.ylabel("Продажі (тис. грн)")
plt.grid(True)
plt.show()

decomp = seasonal_decompose(data["Sales"], model='additive', period=12)
decomp.plot()
plt.suptitle("Декомпозиція часового ряду продажів")
plt.show()


model = ARIMA(data["Sales"], order=(1,1,1))
fit = model.fit()
print(fit.summary())


forecast = fit.forecast(steps=6)
plt.figure(figsize=(8, 4))
plt.plot(data.index, data["Sales"], label="Реальні продажі", color="blue")
plt.plot(pd.date_range(data.index[-1], periods=7, freq="ME")[1:], forecast, label="Прогноз", color="red")
plt.title("Прогноз продажів косметичного бренду на 6 місяців уперед")
plt.ylabel("Продажі (тис. грн)")
plt.legend()
plt.grid(True)
plt.show()

marketing_spend = sales * 0.4 + np.random.normal(0, 10, 24)
corr = np.corrcoef(sales, marketing_spend)[0, 1]
print(f"Коефіцієнт кореляції між маркетинговими витратами та продажами: {corr:.3f}")

plt.scatter(marketing_spend, sales, color="purple")
plt.title("Залежність продажів від маркетингових витрат")
plt.xlabel("Витрати на маркетинг (тис. грн)")
plt.ylabel("Продажі (тис. грн)")
plt.grid(True)
plt.show()

data["Smoothed"] = savgol_filter(data["Sales"], window_length=5, polyorder=2)

plt.figure(figsize=(8, 4))
plt.plot(data.index, data["Sales"], color="gray", label="Оригінальний ряд")
plt.plot(data.index, data["Smoothed"], color="red", label="Згладжений ряд")
plt.title("Згладжування часових коливань продажів косметики")
plt.legend()
plt.grid(True)
plt.show()
