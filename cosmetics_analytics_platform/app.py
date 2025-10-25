import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from analytics.loader import load_data
from analytics.kpi import brand_level_kpis
from analytics.segmentation import segment_brands
from analytics.timeseries import forecast_sales
from analytics.survival import prepare_survival_data, run_cox_model
from analytics.translation import translate_columns
from analytics.insights import generate_all_insights, generate_summary_insight, generate_brand_insight

st.set_page_config(
    page_title="Аналітика косметичних брендів",
    layout="wide",
    page_icon="💄"
)

st.title("Аналітична платформа ефективності косметичних брендів")


df = load_data("data/brands_data.csv")
brands = df['brand'].unique()

st.header("KPI по брендах")
kpi_df = brand_level_kpis(df)
kpi_df_ua = translate_columns(kpi_df, context="kpi")
st.dataframe(kpi_df_ua)

st.header("Введення власних даних бренду")

with st.form("manual_input_form"):
    brand_name = st.text_input("Назва бренду", "Новий бренд")
    avg_sales = st.number_input("Середні продажі (тис. грн)", min_value=0.0, value=120.0)
    avg_profit = st.number_input("Середній прибуток (тис. грн)", min_value=0.0, value=40.0)
    avg_marketing = st.number_input("Рекламний бюджет (тис. грн)", min_value=0.0, value=25.0)
    avg_roi = st.number_input("ROI (прибуток/реклама)", min_value=0.0, value=1.2)
    avg_retention = st.slider("Утримання клієнтів (0–1)", 0.0, 1.0, 0.7)
    avg_innovation = st.slider("Інноваційність (0–10)", 0.0, 10.0, 8.5)
    avg_sustainability = st.slider("Сталий розвиток (0–5)", 0.0, 5.0, 4.2)
    submit = st.form_submit_button("Додати бренд та згенерувати інсайт")

if submit:
    manual_row = pd.DataFrame([{
        "brand": brand_name,
        "avg_sales": avg_sales,
        "avg_profit": avg_profit,
        "avg_marketing_spend": avg_marketing,
        "avg_roi": avg_roi,
        "avg_retention": avg_retention,
        "avg_innovation": avg_innovation,
        "avg_sustainability": avg_sustainability,
        "profit_per_marketing": avg_profit / avg_marketing if avg_marketing > 0 else 0
    }])

    kpi_df = pd.concat([kpi_df, manual_row], ignore_index=True)

st.subheader("AI Insight Generator")

summary_text = generate_summary_insight(kpi_df)
st.success(summary_text)

insight_df = generate_all_insights(kpi_df)
for _, row in insight_df.iterrows():
    st.markdown(f"**{row['brand']}** — {row['insight']}")

if submit:
    insight_text = generate_brand_insight(manual_row.iloc[0])
    st.markdown("### Новий бренд додано:")
    st.info(f"**{brand_name}** — {insight_text}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    sns.barplot(x=["Продажі", "Прибуток", "Реклама"],
                y=[avg_sales, avg_profit, avg_marketing],
                palette="mako", ax=ax[0])
    ax[0].set_title("Фінансові показники нового бренду")

    sns.barplot(x=["ROI", "Утримання", "Інновації", "Сталий розвиток"],
                y=[avg_roi, avg_retention, avg_innovation, avg_sustainability],
                palette="flare", ax=ax[1])
    ax[1].set_title("Профіль ефективності бренду")

    segmented = segment_brands(kpi_df.copy(), n_clusters=3)
    sns.scatterplot(
        data=segmented,
        x='avg_roi',
        y='avg_sales',
        hue='cluster',
        palette='viridis',
        s=120,
        ax=ax[2]
    )
    ax[2].scatter(avg_roi, avg_sales, color='red', s=200, label=brand_name, edgecolor='black')
    ax[2].set_title("Позиція бренду серед кластерів")
    ax[2].legend()
    st.pyplot(fig)



st.header("Сегментація бізнес-моделей брендів")
segmented = segment_brands(kpi_df.copy(), n_clusters=3)
segmented_ua = translate_columns(segmented, context="clusters")

col1, col2 = st.columns([1, 2])
with col1:
    st.dataframe(segmented_ua[['Бренд', 'Кластер']])
with col2:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        data=segmented,
        x='avg_roi',
        y='avg_sales',
        hue='cluster',
        palette='viridis',
        s=150
    )
    plt.title("Сегменти брендів за ROI та продажами")
    st.pyplot(plt)

st.header("Прогноз продажів бренду")
selected_brand = st.selectbox("Оберіть бренд:", kpi_df['brand'].unique())
forecast = forecast_sales(df, selected_brand, periods_ahead=3)
st.line_chart(forecast)

st.header("Survival-аналіз (стійкість бренду у часі)")
surv_df = prepare_survival_data(kpi_df)
cox_model = run_cox_model(surv_df)
cox_ua = translate_columns(cox_model.summary, context="cox")
st.dataframe(cox_ua)

st.header("Порівняння ефективності брендів")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(data=kpi_df, x='brand', y='avg_profit', ax=ax[0], color='orchid')
ax[0].set_title("Середній прибуток по брендах")
sns.barplot(data=kpi_df, x='brand', y='avg_roi', ax=ax[1], color='gold')
ax[1].set_title("Середній ROI по брендах")
st.pyplot(fig)

st.markdown(
    """
    <div style="text-align:center; font-size:14px; color:#444;">
        © 2025 <b>Cosmetics Analytics Platform</b><br>
        Навчальний проєкт з інтелектуального аналізу даних<br>
        Автор: <b>Коваленко В.В.</b>
    </div>
    """,
    unsafe_allow_html=True
)
