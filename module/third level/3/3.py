import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

data = pd.DataFrame({
    'brand': [
        'LuxeSkin', 'BioCare', 'PureGlow', 'Natura', 'AromaLine',
        'Velvet', 'BeautyLab', 'EcoDerm', 'SkinPro', 'HerbalMix',
        'MagicTouch', 'Glamify', 'DermSoft', 'VitalSkin', 'RoseMist'
    ],
    'ad_spend': [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32],
    'revenue': [40, 52, 65, 78, 90, 102, 112, 118, 121, 122, 123, 124, 124.5, 125, 125.2]
})

x = data['ad_spend'].values
y = data['revenue'].values

def model_exp(x, a, b, c):
    return a * np.exp(b * x) + c

popt, pcov = curve_fit(model_exp, x, y, p0=(1.0, 0.05, 30.0), maxfev=10000)
a, b, c = popt

y_pred = model_exp(x, *popt)

r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"a = {a:.4f}, b = {b:.4f}, c = {c:.4f}")
print(f"R^2 = {r2:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"Модель: revenue ≈ {a:.3f} * exp({b:.3f} * ad_spend) + {c:.3f}")
