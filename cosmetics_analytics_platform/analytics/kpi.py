import pandas as pd

def brand_level_kpis(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby('brand').agg({
        'sales': 'mean',
        'marketing_spend': 'mean',
        'profit': 'mean',
        'roi': 'mean',
        'online_share': 'mean',
        'retention': 'mean',
        'innovation': 'mean',
        'sustainability': 'mean'
    }).rename(columns={
        'sales': 'avg_sales',
        'marketing_spend': 'avg_marketing_spend',
        'profit': 'avg_profit',
        'roi': 'avg_roi',
        'online_share': 'avg_online_share',
        'retention': 'avg_retention',
        'innovation': 'avg_innovation',
        'sustainability': 'avg_sustainability'
    })

    grouped['profit_per_marketing'] = grouped['avg_profit'] / grouped['avg_marketing_spend']

    return grouped.reset_index()
