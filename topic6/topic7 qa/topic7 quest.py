import warnings
warnings.filterwarnings("ignore")

import mysql.connector
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from contextlib import contextmanager
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

os.makedirs("img", exist_ok=True)

DB_CONFIG = {
    "database": "cosmetic_analytics",
    "user": "root",
    "password": "1234Nana",
    "host": "localhost",
    "port": 3306
}

NUM_RECORDS = 3000

@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        yield conn
        conn.commit()
    except:
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

def create_tables(conn):
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS dataset;")
        cur.execute("""
        CREATE TABLE dataset (
            brand_id INT AUTO_INCREMENT PRIMARY KEY,
            brand_name VARCHAR(50),
            category VARCHAR(20),
            channel VARCHAR(20),
            region VARCHAR(30),
            avg_price DOUBLE,
            marketing_spend DOUBLE,
            sales DOUBLE,
            profit DOUBLE,
            roi DOUBLE,
            social_engagement INT,
            loyalty_index INT,
            effectiveness_class VARCHAR(20)
        );
        """)

def generate_effectiveness(sales, roi, social_eng, loyalty):
    score = 0
    if roi > 1.5: score += 3
    elif roi > 1.1: score += 2
    elif roi > 0.8: score += 1
    if sales > 200: score += 2
    elif sales > 120: score += 1
    if social_eng > 80: score += 2
    elif social_eng > 60: score += 1
    if loyalty > 80: score += 2
    elif loyalty > 65: score += 1
    if score >= 8: eff = "High"
    elif score >= 4: eff = "Medium"
    else: eff = "Low"
    if eff == "High" and random.random() < 0.05: eff = "Medium"
    if eff == "Low" and random.random() < 0.05: eff = random.choice(["Medium", "High"])
    return eff

def populate_smart_data(conn):
    brands = ["Fenty","L'Oreal","Maybelline","Rare Beauty","Sephora","Huda Beauty","NYX","Dior","Chanel"]
    categories = {"Fenty":"premium","Rare Beauty":"premium","Huda Beauty":"premium","Dior":"luxury","Chanel":"luxury",
                  "L'Oreal":"mass","Maybelline":"mass","NYX":"mass","Sephora":"premium"}
    channels = ["online","offline","omnichannel"]
    regions = ["Europe","Asia","North America"]
    with conn.cursor() as cur:
        for _ in range(NUM_RECORDS):
            brand = random.choice(brands)
            category = categories[brand]
            channel = random.choice(channels)
            region = random.choice(regions)
            if category == "mass":
                avg_price = random.normalvariate(12,2)
                marketing = random.normalvariate(80,15)
                units = random.normalvariate(900,150)
                social = random.normalvariate(55,15)
                loyalty = random.normalvariate(60,10)
            elif category == "premium":
                avg_price = random.normalvariate(25,4)
                marketing = random.normalvariate(140,25)
                units = random.normalvariate(650,120)
                social = random.normalvariate(70,12)
                loyalty = random.normalvariate(75,8)
            else:
                avg_price = random.normalvariate(45,7)
                marketing = random.normalvariate(220,40)
                units = random.normalvariate(350,80)
                social = random.normalvariate(80,10)
                loyalty = random.normalvariate(85,7)
            mult = 1.1 if channel == "online" else 1.25 if channel == "omnichannel" else 1
            units *= mult
            sales = (avg_price * units) / 1000
            margin = random.uniform(0.18,0.28) if category=="mass" else random.uniform(0.25,0.35) if category=="premium" else random.uniform(0.3,0.4)
            profit = sales * margin - marketing * random.uniform(0.6,1.1)
            roi = profit / marketing if marketing != 0 else 0
            social = int(max(0,min(100,social)))
            loyalty = int(max(0,min(100,loyalty)))
            eff = generate_effectiveness(sales, roi, social, loyalty)
            cur.execute("""
            INSERT INTO dataset
            (brand_name,category,channel,region,avg_price,marketing_spend,sales,profit,roi,
             social_engagement,loyalty_index,effectiveness_class)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,(brand,category,channel,region,avg_price,marketing,sales,profit,roi,social,loyalty,eff))

def perform_prediction_analysis(conn):
    df = pd.read_sql("SELECT * FROM dataset", conn)
    le1 = LabelEncoder(); le2 = LabelEncoder(); le3 = LabelEncoder()
    df["category_encoded"] = le1.fit_transform(df["category"])
    df["channel_encoded"] = le2.fit_transform(df["channel"])
    df["region_encoded"] = le3.fit_transform(df["region"])
    X = df[["avg_price","marketing_spend","sales","profit","roi","social_engagement","loyalty_index",
            "category_encoded","channel_encoded","region_encoded"]]
    y = df["effectiveness_class"]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
    clf = RandomForestClassifier(n_estimators=150,random_state=42)
    clf.fit(X_train,y_train)
    pred = clf.predict(X_test)
    print(classification_report(y_test,pred))

    plt.figure(figsize=(6,4))
    sns.countplot(x="effectiveness_class",data=df,order=["Low","Medium","High"])
    plt.tight_layout()
    plt.savefig("img/1.png"); plt.close()

    cm = confusion_matrix(y_test,pred,labels=clf.classes_)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=clf.classes_,yticklabels=clf.classes_)
    plt.tight_layout()
    plt.savefig("img/2.png"); plt.close()

    names = ["Ціна","Маркетинг","Продажі","Прибуток","ROI","Залученість","Лояльність","Категорія","Канал","Регіон"]
    fi = pd.Series(clf.feature_importances_,index=names).sort_values()
    plt.figure(figsize=(7,6))
    fi.plot(kind="barh")
    plt.tight_layout()
    plt.savefig("img/3.png"); plt.close()

    plt.figure(figsize=(7,5))
    sns.boxplot(x="effectiveness_class",y="roi",data=df,order=["Low","Medium","High"])
    plt.tight_layout()
    plt.savefig("img/4.png"); plt.close()

    plt.figure(figsize=(7,5))
    sns.scatterplot(x="marketing_spend",y="sales",hue="effectiveness_class",data=df,palette="deep")
    plt.tight_layout()
    plt.savefig("img/5.png"); plt.close()

    corr = df[["avg_price","marketing_spend","sales","profit","roi","social_engagement","loyalty_index"]].corr()
    plt.figure(figsize=(7,5))
    sns.heatmap(corr,annot=True,cmap="coolwarm",fmt=".2f")
    plt.tight_layout()
    plt.savefig("img/6.png"); plt.close()

def main():
    with get_db_connection() as conn:
        create_tables(conn)
        populate_smart_data(conn)
        perform_prediction_analysis(conn)

if __name__ == "__main__":
    main()
