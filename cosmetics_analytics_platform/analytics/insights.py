import pandas as pd

def generate_brand_insight(row: pd.Series) -> str:
    brand = row["brand"]
    sales = row["avg_sales"]
    roi = row["avg_roi"]
    profit = row["avg_profit"]
    marketing = row["avg_marketing_spend"]
    retention = row["avg_retention"]
    innovation = row["avg_innovation"]
    sustainability = row["avg_sustainability"]
    eff = row["profit_per_marketing"]

    parts = []

    parts.append(
        f"{brand} має середні продажі {sales:.1f} тис. грн "
        f"та прибуток {profit:.1f} тис. грн на місяць."
    )

    parts.append(
        f"Витрати на рекламу становлять {marketing:.1f} тис. грн, "
        f"кожна 1 тис. грн реклами приносить {eff:.2f} тис. грн прибутку."
    )

    if roi >= 1.0:
        parts.append(f"ROI високий ({roi:.2f}) — інвестиції в маркетинг ефективні.")
    elif roi >= 0.8:
        parts.append(f"ROI середній ({roi:.2f}) — є потенціал для підвищення ефективності.")
    else:
        parts.append(f"ROI низький ({roi:.2f}) — варто переглянути маркетингову стратегію.")

    if retention >= 0.6:
        parts.append(f"Рівень утримання клієнтів добрий ({retention:.2f}).")
    else:
        parts.append(f"Рівень утримання клієнтів низький ({retention:.2f}).")

    if innovation >= 9:
        parts.append("Висока інноваційність — бренд задає тренди на ринку.")
    elif innovation >= 7:
        parts.append("Інноваційність достатньо висока, бренд підтримує зацікавленість.")
    else:
        parts.append("Інноваційність низька — ризик втрати конкурентності.")

    if sustainability >= 4.5:
        parts.append("Високий рівень екологічності та сталості бізнесу.")
    elif sustainability >= 3.5:
        parts.append("Бренд частково враховує сталий розвиток.")
    else:
        parts.append("Сталий розвиток виражений слабко, що може впливати на імідж.")

    return " ".join(parts)


def generate_all_insights(kpi_df: pd.DataFrame) -> pd.DataFrame:
    insights = []
    for _, row in kpi_df.iterrows():
        text = generate_brand_insight(row)
        insights.append({"brand": row["brand"], "insight": text})
    return pd.DataFrame(insights)


def generate_summary_insight(kpi_df: pd.DataFrame) -> str:
    top_roi = kpi_df.sort_values("avg_roi", ascending=False).iloc[0]
    top_profit = kpi_df.sort_values("avg_profit", ascending=False).iloc[0]
    low_roi = kpi_df.sort_values("avg_roi", ascending=True).iloc[0]

    return (
        f"Найкращий ROI має {top_roi['brand']} ({top_roi['avg_roi']:.2f}). "
        f"Найбільший прибуток показує {top_profit['brand']} ({top_profit['avg_profit']:.1f} тис. грн). "
        f"Найнижчий ROI у {low_roi['brand']} ({low_roi['avg_roi']:.2f}) — бренд потребує перегляду стратегії."
    )
