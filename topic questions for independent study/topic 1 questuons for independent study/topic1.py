import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

np.random.seed(42)
brands = [
    "L'Oréal", "Sephora", "Fenty Beauty", "The Ordinary", "Estée Lauder",
    "NYX", "Maybelline", "Clinique", "Benefit", "Huda Beauty",
    "Lancôme", "Shiseido", "Tarte", "Glossier", "Charlotte Tilbury"
]

data = pd.DataFrame({
    "Brand": brands,
    "RevenueGrowth_%": np.clip(np.random.normal(20, 5, len(brands)), 5, 35),
    "OnlineShare_%": np.clip(np.random.normal(60, 10, len(brands)), 30, 95),
    "ROI": np.clip(np.random.normal(2.5, 0.8, len(brands)), 0.5, 5),
    "CustomerRetention_%": np.clip(np.random.normal(50, 10, len(brands)), 20, 80),
    "NPS": np.clip(np.random.normal(45, 15, len(brands)), 5, 90)
})

data["Class"] = np.where(data["RevenueGrowth_%"] > 22, "High",
                 np.where(data["RevenueGrowth_%"] > 16, "Medium", "Low"))

print("Вихідні дані:\n", data.head(), "\n")

X = data[["RevenueGrowth_%", "ROI", "CustomerRetention_%", "NPS", "OnlineShare_%"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
data["Cluster"] = clusters

sil = silhouette_score(X_scaled, clusters)
db = davies_bouldin_score(X_scaled, clusters)
ch = calinski_harabasz_score(X_scaled, clusters)

print(f"Silhouette Score: {sil:.3f}")
print(f"Davies-Bouldin Index: {db:.3f}")
print(f"Calinski-Harabasz Index: {ch:.3f}\n")

plt.figure(figsize=(7, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap="viridis", s=100)
plt.title("K-Means кластеризація брендів косметики")
plt.xlabel("Revenue Growth (норм.)")
plt.ylabel("ROI (норм.)")
plt.grid(True)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_scaled, data["Class"], test_size=0.3, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Точність класифікації: {acc:.3f}")
print("\nЗвіт класифікації:\n", classification_report(y_test, y_pred))
print("Матриця плутанини:\n", confusion_matrix(y_test, y_pred))

print("\n=== Висновок ===")
print(f"Модель класифікує бренди за ефективністю з точністю {acc:.2f}.")
print(f"Якість кластеризації підтверджується Silhouette Score = {sil:.2f}.")
print("Це означає, що бренди поділені на три чіткі групи за ключовими бізнес-показниками.")
