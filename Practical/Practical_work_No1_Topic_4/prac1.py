# ===============================================
# Лабораторна робота з аналізу часових рядів
# Тема: Аналіз і прогнозування продажів косметичних брендів
# Автор: Коваленко Владислава Володимирівна
# Мова: Python 3.13
# ===============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf

months = pd.date_range(start="2022-01-01", periods=36, freq="M")
np.random.seed(42)
sales = 100 + np.linspace(0, 60, 36) + 10*np.sin(np.arange(36)/2) + np.random.normal(0, 4, 36)
df = pd.DataFrame({"Month": months, "Sales": sales})
df.to_csv("cosmetic_sales.csv", index=False)

plt.figure(figsize=(9, 4))
plt.plot(df["Month"], df["Sales"], marker="o", color="purple")
plt.title("Продажі косметичного бренду у часі")
plt.xlabel("Місяць")
plt.ylabel("Продажі (тис. грн)")
plt.grid(True)
plt.tight_layout()
plt.savefig("1_trend.png", dpi=150)
plt.close()

values = df["Sales"].values
up, down = 0, 0
trend = 0
for i in range(1, len(values)):
    if values[i] > values[i-1]:
        if trend != 1:
            up += 1
            trend = 1
    elif values[i] < values[i-1]:
        if trend != -1:
            down += 1
            trend = -1
print(f"Висхідних серій: {up}, низхідних: {down}")
if up > down:
    print("Тенденція присутня — продажі зростають.")
else:
    print("Тенденція відсутня або нестійка.")

decomp = seasonal_decompose(df["Sales"], model='additive', period=12)
fig = decomp.plot()
fig.set_size_inches(9, 6)
plt.suptitle("Декомпозиція продажів косметичного бренду")
plt.tight_layout()
plt.savefig("2_decomposition.png", dpi=150)
plt.close()

residuals = decomp.resid.dropna()
acf_vals = acf(residuals, nlags=24)
plt.figure(figsize=(8, 4))
plt.stem(range(len(acf_vals)), acf_vals)
plt.title("Автокореляційна функція випадкової складової")
plt.xlabel("Лаг")
plt.ylabel("ACF")
plt.tight_layout()
plt.savefig("3_acf.png", dpi=150)
plt.close()

if np.abs(acf_vals[1]) < 0.3:
    print("Розкладання коректне — випадкова складова стаціонарна.")
else:
    print("Можливо, залишилась частина тренду — потрібна перевірка.")
