import pandas as pd
from scipy.stats import pearsonr

data = pd.DataFrame({
    'lipoproteins': [3.1, 3.3, 3.5, 3.8, 4.0, 4.1, 4.4, 4.6, 4.9, 5.1],
    'hemoglobin':   [125, 128, 131, 135, 138, 140, 143, 146, 148, 150]
})

x = data['lipoproteins']
y = data['hemoglobin']

r, p = pearsonr(x, y)

print(f"Коефіцієнт кореляції Пірсона r = {r:.4f}")
print(f"p-value = {p:.4e}")

abs_r = abs(r)
if abs_r < 0.3:
    strength = "слабкий"
elif abs_r < 0.7:
    strength = "середній"
else:
    strength = "сильний"

direction = "прямий (зі зростанням ліпопротеїнів зростає гемоглобін)" if r > 0 else "обернений (зі зростанням ліпопротеїнів зменшується гемоглобін)"

print(f"Інтерпретація: {strength} {direction} зв'язок.")
print("Статистично значущий?" , "так" if p < 0.05 else "ні")
