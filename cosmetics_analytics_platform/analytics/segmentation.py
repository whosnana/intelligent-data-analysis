import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def segment_brands(kpi_df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    kpi_df = kpi_df.copy()

    numeric_cols = [
        "avg_sales", "avg_profit", "avg_marketing_spend",
        "avg_roi", "avg_retention", "avg_innovation",
        "avg_sustainability", "profit_per_marketing"
    ]

    kpi_df[numeric_cols] = kpi_df[numeric_cols].fillna(kpi_df[numeric_cols].mean())

    X = kpi_df[numeric_cols].values
    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kpi_df["cluster"] = kmeans.fit_predict(X_scaled)

    return kpi_df
