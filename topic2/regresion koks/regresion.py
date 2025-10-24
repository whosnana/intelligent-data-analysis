import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt

np.random.seed(42)

n = 80
brands = ["L'Oréal", "Fenty", "Rare Beauty", "Maybelline", "Sephora"]
data = pd.DataFrame({
    "brand": np.random.choice(brands, n),
    "marketing_spend": np.random.uniform(50, 150, n),
    "roi": np.random.uniform(0.3, 0.9, n),
    "sustainability": np.random.randint(1, 6, n),
    "innovation": np.random.randint(1, 10, n),
    "time_to_decline": np.random.randint(6, 60, n),
    "decline_event": np.random.choice([0, 1], n, p=[0.4, 0.6])
})

data.to_csv("cosmetic_brand_data.csv", index=False)
print("Файл 'cosmetic_brand_data.csv' збережено.")

cph = CoxPHFitter()
cph.fit(data[["time_to_decline", "decline_event", "marketing_spend", "roi", "sustainability", "innovation"]],
        duration_col="time_to_decline", event_col="decline_event")

summary = cph.summary
print(summary)

cph.plot()
plt.title("Вплив факторів на ризик зниження ефективності бренду")
plt.tight_layout()
plt.savefig("1_cox_coefficients.png", dpi=150)
plt.close()

cph.plot_partial_effects_on_outcome(covariates='roi', values=[0.3, 0.5, 0.7, 0.9])
plt.title("Вплив ROI на ризик зниження ефективності бренду")
plt.tight_layout()
plt.savefig("2_cox_roi.png", dpi=150)
plt.close()

cph.plot_partial_effects_on_outcome(covariates='marketing_spend', values=[60, 100, 140])
plt.title("Вплив рекламного бюджету на стабільність бренду")
plt.tight_layout()
plt.savefig("3_cox_marketing.png", dpi=150)
plt.close()

print("\nІнтерпретація:")
for var in summary.index:
    hr = np.exp(summary.loc[var, "coef"])
    p = summary.loc[var, "p"]
    trend = "підвищує ризик" if hr > 1 else "зменшує ризик"
    print(f"{var}: HR={hr:.2f}, p={p:.3f} → {trend}")
