import mysql.connector
from mysql.connector import Error
import random
from contextlib import contextmanager
import pandas as pd
import matplotlib.pyplot as plt
import os
from sqlalchemy import create_engine

DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "1234Nana",
    "charset": "utf8mb4"
}

DB_NAME = "cosmetics_db"
NUM_RECORDS = 800

os.makedirs("img", exist_ok=True)

def plot_bar(series, title, ylabel, path, rotation=45):
    plt.figure(figsize=(10, 6))
    series.plot(kind="bar")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()

def plot_bar_df(df, title, ylabel, path, rotation=45):
    plt.figure(figsize=(10, 6))
    df.plot(kind="bar", ax=plt.gca())
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()

def plot_heatmap(pivot_df, title, path):
    plt.figure(figsize=(8, 6))
    data = pivot_df.values
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, aspect="auto")
    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_xticklabels(pivot_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(pivot_df.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center", color="white")
    plt.title(title)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()

@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = mysql.connector.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            charset=DB_CONFIG["charset"]
        )
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS {DB_NAME} "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        conn.database = DB_NAME
        yield conn
        conn.commit()
    except Error as e:
        print("Помилка підключення:", e)
        if conn:
            conn.rollback()
        raise
    finally:
        if conn and conn.is_connected():
            conn.close()

def create_table(conn):
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS brand_analytics;")
        cur.execute("""
            CREATE TABLE brand_analytics (
                id INT AUTO_INCREMENT PRIMARY KEY,
                brand VARCHAR(100),
                segment VARCHAR(50),
                product_line VARCHAR(100),
                business_model VARCHAR(50),
                region VARCHAR(100),
                customers INT,
                revenue DECIMAL(10,2),
                marketing_cost DECIMAL(10,2),
                operational_cost DECIMAL(10,2),
                profit DECIMAL(10,2),
                roi DECIMAL(10,2)
            );
        """)

def populate_data(conn):
    brands = {
        "Fenty Beauty": "Преміум",
        "Rare Beauty": "Преміум",
        "L'Oréal Paris": "Масмаркет",
        "Maybelline": "Масмаркет",
        "Sephora Collection": "Преміум",
        "NYX": "Масмаркет"
    }
    product_lines = ["Макіяж", "Догляд за шкірою", "Волосся", "Парфуми"]
    business_models = ["D2C", "Ритейл", "Маркетплейс", "Омніканальна"]
    regions = ["Європа", "Північна Америка", "Азія", "Онлайн-глобально"]

    with conn.cursor() as cur:
        for _ in range(NUM_RECORDS):
            brand = random.choice(list(brands.keys()))
            segment = brands[brand]
            line = random.choice(product_lines)
            model = random.choice(business_models)
            region = random.choice(regions)
            customers = random.randint(20, 500)
            avg_order = random.uniform(15, 120)
            revenue = round(customers * avg_order, 2)
            marketing = round(revenue * random.uniform(0.05, 0.25), 2)
            operations = round(revenue * random.uniform(0.20, 0.45), 2)
            profit = round(revenue - marketing - operations, 2)
            roi = round((profit / (marketing + 1)) * 100, 2)

            cur.execute("""
                INSERT INTO brand_analytics 
                (brand, segment, product_line, business_model, region, customers, 
                 revenue, marketing_cost, operational_cost, profit, roi)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (brand, segment, line, model, region, customers,
                  revenue, marketing, operations, profit, roi))

def get_statistics():
    engine = create_engine("mysql+pymysql://root:1234Nana@localhost:3306/cosmetics_db")
    df = pd.read_sql("SELECT * FROM brand_analytics", engine)

    t1 = df.groupby("brand")["profit"].sum().sort_values(ascending=False)
    plot_bar(t1, "Прибуток за брендами", "Прибуток, грн", "img/1.png", rotation=45)

    t2 = df.groupby("business_model")["roi"].mean().sort_values(ascending=False)
    plot_bar(t2, "Середній ROI за бізнес-моделями", "ROI, %", "img/2.png", rotation=0)

    t3 = df.groupby("segment")["revenue"].sum()
    plot_bar(t3, "Виручка за сегментами", "Виручка, грн", "img/3.png", rotation=0)

    t4 = df.groupby(["brand", "product_line"])["profit"].sum().sort_values(ascending=False).head(5)
    t4_index = [f"{b} — {pl}" for b, pl in t4.index]
    t4_series = pd.Series(t4.values, index=t4_index)
    plot_bar(t4_series, "ТОП-5 (бренд + продуктова лінія) за прибутком", "Прибуток, грн", "img/4.png", rotation=45)

    t5 = df.pivot_table(values="profit", index="business_model", columns="region", aggfunc="sum", fill_value=0)
    plot_heatmap(t5, "Прибуток: бізнес-моделі × регіони", "img/5.png")

    print("Скріншоти збережено у папці img")

def main():
    with get_db_connection() as conn:
        create_table(conn)
        populate_data(conn)
    get_statistics()

if __name__ == "__main__":
    main()
