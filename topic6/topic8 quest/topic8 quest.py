import warnings
warnings.filterwarnings("ignore")

import mysql.connector
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import contextmanager
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

DB_CONFIG = {
    "user": "root",
    "password": "1234Nana",
    "host": "localhost",
    "database": "cosmetic_analytics",
    "port": 3306
}

NUM_RECORDS = 3000

IMG_DIR = Path("img")
IMG_DIR.mkdir(exist_ok=True)

@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        yield conn
        conn.commit()
    except Exception as e:
        print(f"Помилка підключення до БД: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

def create_tables(conn):
    print("1. ЕТАП КОНСТРУЮВАННЯ: Створення сховища даних для косметичних брендів...")
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS brand_dataset;")
        cur.execute("""
            CREATE TABLE brand_dataset (
                id INT AUTO_INCREMENT PRIMARY KEY,
                brand_name VARCHAR(50),
                category VARCHAR(20),
                channel VARCHAR(20),
                region VARCHAR(30),
                avg_price DECIMAL(6,2),
                marketing_spend DECIMAL(10,2),
                sales DECIMAL(10,2),
                profit DECIMAL(10,2),
                roi DECIMAL(5,2),
                social_engagement INT,
                loyalty_index INT,
                effectiveness_class VARCHAR(20)
            );
        """)
    print("   -> Таблицю brand_dataset успішно створено.")

def generate_effectiveness(sales, roi, social_engagement, loyalty_index):
    score = 0
    if sales > 200000:
        score += 3
    elif sales > 120000:
        score += 2
    elif sales > 70000:
        score += 1
    if roi > 1.6:
        score += 3
    elif roi > 1.2:
        score += 2
    elif roi > 1.0:
        score += 1
    if social_engagement > 80:
        score += 2
    elif social_engagement > 50:
        score += 1
    if loyalty_index > 80:
        score += 2
    elif loyalty_index > 60:
        score += 1
    if score >= 8:
        return "High"
    elif score >= 4:
        return "Medium"
    else:
        return "Low"

def populate_smart_data(conn):
    print("2. ЕТАП ГЕНЕРАЦІЇ: Наповнення даними про косметичні бренди...")
    brands = ["Fenty", "LOréal", "Sephora", "Rare Beauty", "Maybelline", "Huda Beauty"]
    categories = ["mass", "premium", "luxury"]
    channels = ["online", "offline", "omnichannel"]
    regions = ["Europe", "Asia", "North America"]
    with conn.cursor() as cur:
        for _ in range(NUM_RECORDS):
            brand = random.choice(brands)
            if brand in ["Fenty", "Rare Beauty", "Huda Beauty"]:
                category = random.choice(["premium", "luxury"])
            elif brand in ["Maybelline"]:
                category = "mass"
            else:
                category = random.choice(categories)
            channel = random.choice(channels)
            region = random.choice(regions)
            if category == "luxury":
                avg_price = round(random.normalvariate(40, 8), 2)
                marketing_spend = round(random.normalvariate(250000, 50000), 2)
            elif category == "premium":
                avg_price = round(random.normalvariate(28, 5), 2)
                marketing_spend = round(random.normalvariate(180000, 40000), 2)
            else:
                avg_price = round(random.normalvariate(15, 3), 2)
                marketing_spend = round(random.normalvariate(120000, 30000), 2)
            base_demand = random.normalvariate(8000, 2000)
            if channel == "online":
                base_demand *= 1.1
            elif channel == "omnichannel":
                base_demand *= 1.25
            sales_units = max(1000, int(base_demand + random.normalvariate(0, 1500)))
            sales = float(sales_units * avg_price)
            margin_ratio = random.uniform(0.2, 0.45)
            profit = sales * margin_ratio - marketing_spend * random.uniform(0.7, 1.1)
            roi = 0.0
            if marketing_spend > 0:
                roi = profit / marketing_spend
            social_engagement = int(max(0, min(100, random.normalvariate(60, 20))))
            loyalty_index = int(max(0, min(100, random.normalvariate(65, 15))))
            effectiveness = generate_effectiveness(sales, roi, social_engagement, loyalty_index)
            if effectiveness == "High" and random.random() < 0.05:
                effectiveness = "Medium"
            if effectiveness == "Low" and random.random() < 0.05:
                effectiveness = random.choice(["Medium", "High"])
            cur.execute(
                """
                INSERT INTO brand_dataset
                (brand_name, category, channel, region, avg_price, marketing_spend,
                 sales, profit, roi, social_engagement, loyalty_index, effectiveness_class)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    brand,
                    category,
                    channel,
                    region,
                    avg_price,
                    marketing_spend,
                    sales,
                    profit,
                    roi,
                    social_engagement,
                    loyalty_index,
                    effectiveness,
                ),
            )
    print(f"   -> Згенеровано {NUM_RECORDS} записів по брендах.")

def analyze_data(conn):
    print("\n3. ЕТАП МОДЕЛЮВАННЯ ТА АНАЛІЗУ ДЛЯ КОСМЕТИЧНИХ БРЕНДІВ...")
    df = pd.read_sql("SELECT * FROM brand_dataset", conn)
    le_brand = LabelEncoder()
    le_cat = LabelEncoder()
    le_channel = LabelEncoder()
    le_region = LabelEncoder()
    df["brand_encoded"] = le_brand.fit_transform(df["brand_name"])
    df["category_encoded"] = le_cat.fit_transform(df["category"])
    df["channel_encoded"] = le_channel.fit_transform(df["channel"])
    df["region_encoded"] = le_region.fit_transform(df["region"])
    features = [
        "avg_price",
        "marketing_spend",
        "sales",
        "profit",
        "roi",
        "social_engagement",
        "loyalty_index",
        "category_encoded",
        "channel_encoded",
        "region_encoded",
    ]
    ukr_features = [
        "Сер.ціна",
        "Маркетинг",
        "Продажі",
        "Прибуток",
        "ROI",
        "Соц.залученість",
        "Лояльність",
        "Категорія",
        "Канал",
        "Регіон",
    ]
    print("   [1/2] Класифікація (Random Forest): Прогноз ефективності бренду...")
    X = df[features]
    y = df["effectiveness_class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("\n   >>> Результати класифікації ефективності:")
    print(classification_report(y_test, y_pred))
    print("   [2/2] Кластеризація (K-Means): Сегментація брендів за бізнес-моделями...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(
        df[["sales", "profit", "roi", "marketing_spend", "social_engagement", "loyalty_index"]]
    )
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)
    print("   -> Бренди розділені на 3 кластери (типи бізнес-моделей).")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    feature_imp = pd.Series(clf.feature_importances_, index=ukr_features).sort_values(ascending=False)
    sns.barplot(x=feature_imp, y=feature_imp.index)
    plt.title("Які фактори найбільше впливають на ефективність бренду?", fontsize=14)
    plt.xlabel("Важливість", fontsize=12)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "1.png")
    plt.close()
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=clf.classes_,
        yticklabels=clf.classes_,
    )
    plt.title("Матриця помилок для класифікації ефективності", fontsize=14)
    plt.ylabel("Реальний клас")
    plt.xlabel("Прогнозований клас")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "2.png")
    plt.close()
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        x="sales",
        y="profit",
        hue="cluster",
        style="effectiveness_class",
        data=df,
        s=60,
        alpha=0.8,
    )
    plt.title("Кластери брендів: Продажі vs Прибуток", fontsize=14)
    plt.xlabel("Продажі")
    plt.ylabel("Прибуток")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "3.png")
    plt.close()
    plt.figure(figsize=(9, 5))
    sns.countplot(x="effectiveness_class", data=df)
    plt.title("Розподіл брендів за класами ефективності", fontsize=14)
    plt.xlabel("Клас ефективності")
    plt.ylabel("Кількість брендів")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "4.png")
    plt.close()
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="marketing_spend",
        y="roi",
        hue="effectiveness_class",
        data=df,
        s=60,
        alpha=0.8,
    )
    plt.title("ROI vs Маркетинговий бюджет", fontsize=14)
    plt.xlabel("Маркетинговий бюджет")
    plt.ylabel("ROI")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "5.png")
    plt.close()

def main():
    with get_db_connection() as conn:
        if conn:
            create_tables(conn)
            populate_smart_data(conn)
            analyze_data(conn)
        else:
            print("Критична помилка: MySQL недоступна.")

if __name__ == "__main__":
    main()
