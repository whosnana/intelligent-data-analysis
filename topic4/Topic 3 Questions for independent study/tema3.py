import numpy as np
import matplotlib.pyplot as plt

profitability = np.array([1.6, 0.8, 1.2, 0.5, 0.9, 1.1, 1.1, 0.6, 1.5, 0.8, 0.9, 1.2, 0.5, 1.3, 0.8, 1.2])

t = np.arange(1, len(profitability) + 1)

plt.figure(figsize=(8, 4))
plt.plot(t, profitability, marker='o', linestyle='-', color='teal')
plt.title("Динаміка прибутковості косметичного бренду")
plt.xlabel("Період (місяці)")
plt.ylabel("Прибутковість (умовні одиниці)")
plt.grid(True)
plt.show()

plt.figure(figsize=(5, 5))
plt.scatter(profitability[:-1], profitability[1:], color='purple')
plt.title("Залежність прибутковості y(t+1) від y(t)")
plt.xlabel("y(t)")
plt.ylabel("y(t+1)")
plt.grid(True)
plt.show()

y_mean = np.mean(profitability)
numerator = np.sum((profitability[:-1] - y_mean) * (profitability[1:] - y_mean))
denominator = np.sum((profitability - y_mean)**2)
r1 = numerator / denominator

print(f"Коефіцієнт автокореляції першого порядку: r1 = {r1:.3f}")
if abs(r1) < 0.3:
    print("→ Автокореляція слабка: прибутковість між періодами майже не пов’язана.")
elif abs(r1) < 0.7:
    print("→ Автокореляція середня: прибутковість має помірну залежність від попереднього місяця.")
else:
    print("→ Автокореляція сильна: прибутковість бренду стабільно залежить від попередніх значень.")


trend = 0.05
n = 20
noise = np.random.normal(0, 0.1, n)
Y = np.zeros(n)
Y[0] = 1.0

for t in range(1, n):
    Y[t] = Y[t-1] + trend + noise[t]

plt.figure(figsize=(8, 4))
plt.plot(range(n), Y, marker='o', color='darkgreen')
plt.title("Випадкове блукання з трендом (прибутковість бренду)")
plt.xlabel("Період (місяці)")
plt.ylabel("Значення показника Y(t)")
plt.grid(True)
plt.show()

tau = 5
y_t = Y[-1]
forecast = y_t + tau * trend
print(f"Прогноз на {tau} періодів уперед: y(t+{tau}) = {forecast:.3f}")

forecast_error = Y[-1] - (Y[-2] + trend)
mse = np.mean((Y[1:] - (Y[:-1] + trend))**2)

print(f"Прогнозна помилка останнього кроку: {forecast_error:.3f}")
print(f"Середня квадратична похибка (СКП): {mse:.4f}")