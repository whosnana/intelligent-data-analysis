import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

data = pd.DataFrame({
    'stabilized_glucose': [4.1, 4.3, 4.5, 4.8, 5.0, 5.2, 5.5, 5.8, 6.0, 6.3],
    'hemoglobin':         [122, 124, 128, 133, 138, 142, 145, 147, 149, 150]
})

x = data['stabilized_glucose'].values
y = data['hemoglobin'].values

def model_exp(x, a, b, c):
    return a * np.exp(b * x) + c

popt, _ = curve_fit(model_exp, x, y, p0=(1.0, 0.05, 100.0), maxfev=10000)
a, b, c = popt
y_pred = model_exp(x, *popt)

r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"a = {a:.4f}, b = {b:.4f}, c = {c:.4f}")
print(f"R^2 = {r2:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"Модель: hemoglobin ≈ {a:.3f} * exp({b:.3f} * glucose) + {c:.3f}")
