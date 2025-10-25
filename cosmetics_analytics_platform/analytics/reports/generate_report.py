import pandas as pd
from analytics.loader import load_data
from analytics.kpi import brand_level_kpis
from analytics.segmentation import segment_brands
from analytics.survival import prepare_survival_data, run_cox_model

def generate_console_report(data_path: str = "data/brands_data.csv"):
    df = load_data(data_path)

    print("=== KPI по брендах ===")
    kpis = brand_level_kpis(df)
    print(kpis.round(3))

    print("\n=== Сегментація брендів (бізнес-моделі) ===")
    segmented = segment_brands(kpis.copy(), n_clusters=3)
    print(segmented[['brand', 'cluster']])

    print("\n=== Survival-аналіз (стійкість бізнес-моделі) ===")
    surv_df = prepare_survival_data(kpis)
    cph = run_cox_model(surv_df)
    print(cph.summary.round(3))

if __name__ == "__main__":
    generate_console_report()
