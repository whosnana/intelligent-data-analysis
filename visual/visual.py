import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sns.set(style="whitegrid", palette="mako")
np.random.seed(42)


months = pd.date_range(start="2022-01-01", periods=36, freq="ME")
brands = ["L'Oréal", "Fenty", "Maybelline", "Sephora", "Rare Beauty"]


sales = 100 + np.linspace(0, 60, 36) + 10*np.sin(np.arange(36)/2) + np.random.normal(0, 4, 36)
marketing = 30 + 6*np.cos(np.arange(36)/3) + np.random.normal(0, 2, 36)
profit = sales * 0.32 + np.random.normal(0, 3, 36)
roi = profit / marketing

df = pd.DataFrame({"Month": months, "Sales": sales, "Marketing": marketing, "Profit": profit, "ROI": roi})
df.set_index("Month", inplace=True)


plt.figure(figsize=(9,5))
plt.plot(df.index, df["Sales"], color="royalblue", linewidth=2)
plt.title("Продажі косметичного бренду у часі")
plt.xlabel("Місяць")
plt.ylabel("Продажі (тис. грн)")
plt.savefig("1_prodazhi.png", dpi=150)
plt.close()

plt.figure(figsize=(9,5))
plt.bar(df.index, df["Marketing"], color="orchid")
plt.title("Динаміка рекламних витрат")
plt.xlabel("Місяць")
plt.ylabel("Рекламний бюджет (тис. грн)")
plt.savefig("2_reklama.png", dpi=150)
plt.close()


plt.figure(figsize=(9,5))
plt.fill_between(df.index, df["Profit"], color="lightgreen", alpha=0.7)
plt.title("Прибуток бренду за період")
plt.xlabel("Місяць")
plt.ylabel("Прибуток (тис. грн)")
plt.savefig("3_prybutok.png", dpi=150)
plt.close()


plt.figure(figsize=(6,5))
sns.regplot(x=df["Marketing"], y=df["Sales"], color="darkviolet")
plt.title("Залежність продажів від рекламного бюджету")
plt.xlabel("Реклама (тис. грн)")
plt.ylabel("Продажі (тис. грн)")
plt.savefig("4_korelyaciya.png", dpi=150)
plt.close()


plt.figure(figsize=(8,4))
plt.plot(df.index, df["ROI"], marker='o', color="gold")
plt.title("ROI косметичного бренду")
plt.xlabel("Місяць")
plt.ylabel("ROI (прибуток/реклама)")
plt.savefig("5_roi.png", dpi=150)
plt.close()


decomp = seasonal_decompose(df["Sales"], model='additive', period=12)
decomp.plot()
plt.suptitle("Декомпозиція продажів (тренд, сезонність, залишки)")
plt.savefig("6_dekompozyciya.png", dpi=150)
plt.close()

acf_values = acf(df["Sales"], nlags=24)
plt.figure(figsize=(7,4))
plt.stem(range(len(acf_values)), acf_values)
plt.title("Автокореляція продажів")
plt.xlabel("Затримка (lag)")
plt.ylabel("ACF")
plt.savefig("7_avtokorelyaciya.png", dpi=150)
plt.close()


model = ExponentialSmoothing(df["Sales"], trend="add", seasonal="add", seasonal_periods=12).fit()
forecast = model.forecast(6)
plt.figure(figsize=(9,5))
plt.plot(df.index, df["Sales"], label="Історія")
plt.plot(pd.date_range(df.index[-1], periods=7, freq="ME")[1:], forecast, 'r--', label="Прогноз")
plt.title("Прогноз продажів на 6 місяців")
plt.xlabel("Місяць")
plt.ylabel("Продажі (тис. грн)")
plt.legend()
plt.savefig("8_prognoz.png", dpi=150)
plt.close()


plt.figure(figsize=(9,5))
sns.regplot(x=np.arange(len(df)), y=df["Profit"], color="green")
plt.title("Тренд прибутковості косметичного бренду")
plt.xlabel("Час (місяці)")
plt.ylabel("Прибуток (тис. грн)")
plt.savefig("9_trend_prybutku.png", dpi=150)
plt.close()


brand_data = pd.DataFrame({
    "Brand": brands,
    "ROI": np.random.uniform(0.4, 0.9, len(brands)),
    "SalesGrowth": np.random.uniform(0.1, 0.5, len(brands))
})

scaler = StandardScaler()
scaled = scaler.fit_transform(brand_data[["ROI", "SalesGrowth"]])
kmeans = KMeans(n_clusters=3, random_state=42)
brand_data["Cluster"] = kmeans.fit_predict(scaled)

plt.figure(figsize=(7,5))
sns.scatterplot(data=brand_data, x="ROI", y="SalesGrowth", hue="Cluster", s=120, palette="viridis")
for i in range(len(brands)):
    plt.text(brand_data["ROI"][i]+0.005, brand_data["SalesGrowth"][i]+0.005, brands[i])
plt.title("Кластеризація косметичних брендів за ефективністю")
plt.xlabel("ROI")
plt.ylabel("Зростання продажів")
plt.savefig("10_klastery_brendy.png", dpi=150)
plt.close()

plt.figure(figsize=(7,5))
sns.barplot(x="Brand", y="ROI", data=brand_data, hue="Brand", palette="cool", legend=False)
plt.title("Індекс ROI для косметичних брендів")
plt.ylabel("ROI")
plt.savefig("11_roi_brendy.png", dpi=150)
plt.close()

plt.figure(figsize=(7,5))
sns.scatterplot(x=df["Marketing"], y=df["Profit"], color="deeppink", s=80)
sns.regplot(x=df["Marketing"], y=df["Profit"], scatter=False, color="black", line_kws={"linewidth":2})
plt.title("Вплив маркетингових витрат на прибуток")
plt.xlabel("Реклама (тис. грн)")
plt.ylabel("Прибуток (тис. грн)")
plt.savefig("12_marketing_prybutok.png", dpi=150)
plt.close()


months_short = months[-12:]
online_sales = np.random.uniform(45, 70, len(months_short))
offline_sales = 100 - online_sales

plt.figure(figsize=(8,5))
plt.stackplot(months_short, online_sales, offline_sales, labels=["Онлайн", "Офлайн"], colors=["#5B84B1", "#E69A8D"])
plt.title("Структура продажів бренду: онлайн та офлайн")
plt.xlabel("Місяць")
plt.ylabel("Частка продажів (%)")
plt.legend()
plt.savefig("13_online_offline.png", dpi=150)
plt.close()


df["MonthNum"] = df.index.month
avg_sales_by_month = df.groupby("MonthNum")["Sales"].mean()

plt.figure(figsize=(8,5))
sns.lineplot(x=avg_sales_by_month.index, y=avg_sales_by_month.values, marker="o", color="teal")
plt.title("Середні продажі за місяцями (сезонність)")
plt.xlabel("Місяць року")
plt.ylabel("Продажі (тис. грн)")
plt.xticks(range(1,13))
plt.grid(True)
plt.savefig("14_sezonnist.png", dpi=150)
plt.close()


profits = np.random.uniform(250, 600, len(brands))
plt.figure(figsize=(7,5))
sns.barplot(x=brands, y=profits, palette="mako")
plt.title("Порівняння прибутковості косметичних брендів")
plt.xlabel("Бренд")
plt.ylabel("Прибуток (тис. грн)")
plt.savefig("15_prybutok_brendy.png", dpi=150)
plt.close()


