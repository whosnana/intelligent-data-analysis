import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

def prepare_survival_data(kpi_df: pd.DataFrame) -> pd.DataFrame:

    np.random.seed(42)
    survival_df = pd.DataFrame({
        'brand': kpi_df['brand'],
        'time_to_decline': np.random.randint(6, 36, size=len(kpi_df)),
        'decline_event': np.random.choice([0,1], size=len(kpi_df), p=[0.4,0.6]),
        'avg_roi': kpi_df['avg_roi'].values,
        'avg_marketing_spend': kpi_df['avg_marketing_spend'].values,
        'avg_innovation': kpi_df['avg_innovation'].values,
        'avg_sustainability': kpi_df['avg_sustainability'].values
    })
    return survival_df

def run_cox_model(survival_df: pd.DataFrame):
    cph = CoxPHFitter()
    cph.fit(
        survival_df.drop(columns=['brand']),
        duration_col='time_to_decline',
        event_col='decline_event'
    )
    return cph
