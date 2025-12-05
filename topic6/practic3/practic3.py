import os
import random
from datetime import date, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from faker import Faker
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

os.makedirs("img", exist_ok=True)

random.seed(42)
np.random.seed(42)
fake = Faker("uk_UA")

categories_brands = {
    "Скінкейр": ["GlowCare", "PureSkin", "DermaSoft"],
    "Макіяж": ["LuxColor", "ProBeauty", "ChicFace"],
    "Волосся": ["HairPro", "SilkyLocks", "VolumeX"],
    "Парфуми": ["AromaSense", "NightMuse", "FreshDay"]
}

geo = {
    "Україна": {
        "Київська область": ["Київ", "Біла Церква"],
        "Львівська область": ["Львів", "Дрогобич"],
        "Одеська область": ["Одеса", "Чорноморськ"]
    },
    "Польща": {
        "Mazowieckie": ["Warszawa", "Radom"],
        "Małopolskie": ["Kraków", "Tarnów"]
    },
    "Німеччина": {
        "Bayern": ["München", "Nürnberg"],
        "Berlin": ["Berlin"]
    }
}

start = date(2020, 1, 1)
end = date(2024, 12, 31)
days_range = (end - start).days

orders = []
for _ in range(2000):
    exec_offset = random.randint(0, days_range)
    exec_date = start + timedelta(days=exec_offset)
    category = random.choice(list(categories_brands.keys()))
    brand = random.choice(categories_brands[category])
    country = random.choice(list(geo.keys()))
    region = random.choice(list(geo[country].keys()))
    city = random.choice(geo[country][region])
    client_last_name = fake.last_name()
    quantity = random.randint(1, 12)
    unit_price = random.uniform(15, 250)
    total_price = round(quantity * unit_price, 2)

    orders.append(
        {
            "exec_date": pd.to_datetime(exec_date),
            "product_brand": brand,
            "product_category": category,
            "client_last_name": client_last_name,
            "client_city": city,
            "client_region": region,
            "client_country": country,
            "quantity": quantity,
            "total_price": total_price,
        }
    )

df = pd.DataFrame(orders)

df["month"] = df["exec_date"].values.astype("datetime64[M]")

agg = (
    df.groupby(["product_brand", "product_category", "month"], as_index=False)
    .agg(total_quantity=("quantity", "sum"), total_price=("total_price", "sum"))
)

df_ts = agg.rename(columns={"month": "ds", "total_price": "y"})
df_ts["unique_id"] = df_ts["product_brand"] + " | " + df_ts["product_category"]
df_ts = df_ts[["unique_id", "ds", "y"]]

sf = StatsForecast(models=[AutoARIMA(season_length=12)], freq="MS", n_jobs=1)
fcst = sf.forecast(df=df_ts, h=12)

plt.figure(figsize=(10, 5))
top_brands = df.groupby("product_brand")["total_price"].sum().sort_values(ascending=False)
top_brands.plot(kind="bar")
plt.title("1. Сумарна ціна замовлень за марками")
plt.tight_layout()
plt.savefig("img/1.png", dpi=250)
plt.close()

plt.figure(figsize=(10, 5))
top_categories = df.groupby("product_category")["total_price"].sum().sort_values(ascending=False)
top_categories.plot(kind="bar")
plt.title("2. Сумарна ціна замовлень за категоріями")
plt.tight_layout()
plt.savefig("img/2.png", dpi=250)
plt.close()

plt.figure(figsize=(10, 5))
country_orders = df.groupby("client_country")["total_price"].sum().sort_values(ascending=False)
country_orders.plot(kind="bar")
plt.title("3. Сума замовлень за країнами клієнтів")
plt.tight_layout()
plt.savefig("img/3.png", dpi=250)
plt.close()

first_uid = df_ts["unique_id"].unique()[0]
hist = df_ts[df_ts["unique_id"] == first_uid].sort_values("ds")
pred = fcst[fcst["unique_id"] == first_uid].sort_values("ds")

plt.figure(figsize=(10, 5))
plt.plot(hist["ds"], hist["y"], label="Фактичні дані")
plt.plot(pred["ds"], pred["AutoARIMA"], label="Прогноз", linestyle="--")
plt.title(f"4. Прогноз сумарної ціни для: {first_uid}")
plt.tight_layout()
plt.legend()
plt.savefig("img/4.png", dpi=250)
plt.close()

pivot = df.pivot_table(values="total_price",
                       index="product_category",
                       columns="client_country",
                       aggfunc="sum",
                       fill_value=0)

plt.figure(figsize=(8, 6))
img = plt.imshow(pivot.values, aspect="auto")
plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
plt.yticks(range(len(pivot.index)), pivot.index)
plt.colorbar(img)
plt.title("5. Heatmap: категорії × країни")
plt.tight_layout()
plt.savefig("img/5.png", dpi=250)
plt.close()

print("Готово! 5 фото збережено у папці img/")
