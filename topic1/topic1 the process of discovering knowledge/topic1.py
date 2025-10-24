import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

marketing = np.array([40, 50, 60, 70, 80, 90, 100])
profit = np.array([10, 14, 18, 22, 27, 33, 38])

corr, p_value = pearsonr(marketing, profit)
print(f"Коефіцієнт кореляції: {corr:.2f}")

sns.regplot(x=marketing, y=profit, color="orchid", marker="o")
plt.title("Кореляція між витратами на рекламу та прибутком")
plt.xlabel("Реклама (тис. грн)")
plt.ylabel("Прибуток (тис. грн)")
plt.show()