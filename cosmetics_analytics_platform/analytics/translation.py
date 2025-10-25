def translate_columns(df, context="kpi"):
    """Переклад назв колонок у таблицях Streamlit українською"""
    translations = {
        "kpi": {
            "brand": "Бренд",
            "avg_sales": "Середні продажі (тис. грн)",
            "avg_marketing_spend": "Середні витрати на рекламу (тис. грн)",
            "avg_profit": "Середній прибуток (тис. грн)",
            "avg_roi": "Середній ROI",
            "avg_online_share": "Частка онлайн-продажів",
            "avg_retention": "Утримання клієнтів (%)",
            "avg_innovation": "Індекс інноваційності",
            "avg_sustainability": "Індекс сталого розвитку",
            "profit_per_marketing": "Прибуток на 1 грн реклами"
        },
        "cox": {
            "covariate": "Змінна",
            "coef": "Коефіцієнт",
            "exp(coef)": "exp(Коеф.)",
            "coef lower 95%": "Нижня межа 95%",
            "coef upper 95%": "Верхня межа 95%",
            "exp(coef) lower 95%": "Нижня межа exp(Коеф.)",
            "exp(coef) upper 95%": "Верхня межа exp(Коеф.)",
            "cmp to": "Порівняно з",
            "z": "Z-статистика",
            "p": "P-значення",
            "-log2(p)": "-log₂(P)"
        },
        "clusters": {
            "brand": "Бренд",
            "avg_roi": "ROI",
            "avg_sales": "Продажі",
            "avg_profit": "Прибуток",
            "cluster": "Кластер"
        }
    }
    return df.rename(columns=translations.get(context, {}))
