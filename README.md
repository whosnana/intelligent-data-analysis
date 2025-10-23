Зміст

Practical/ — ноутбуки/скрипти практичних робіт (ARIMA, K-Means, Random Forest, ACO/GA тощо).

topic questions for independent study/ — тексти й відповіді на питання для самостійної роботи.

topic3_control_questions/, topic4/topic1/ — контрольні питання та теорія по темах курсу. 
GitHub

Швидкий старт
1) Вимоги

Python ≥ 3.10 (рекомендовано 3.11–3.13)

pip/venv

2) Встановлення
# клон
git clone https://github.com/whosnana/intelligent-data-analysis.git
cd intelligent-data-analysis

# (опціонально) створити віртуальне середовище
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# базові пакети
python -m pip install --upgrade pip
pip install -r requirements.txt  # якщо файла ще нема — див. нижче “Мінімальні залежності”


Мінімальні залежності (якщо немає requirements.txt):

pip install numpy pandas matplotlib scikit-learn statsmodels scipy

Як запускати приклади

Приклади покривають три ключові напрями: часові ряди, кластеризація/класифікація, оптимізація.

1) Часові ряди (продажі/прибутковість брендів)
python Practical/prac_timeseries.py


Дає:

декомпозицію (тренд/сезонність/шум),

моделі Holt-Winters/ARIMA,

прогноз на 3–6 місяців,

графіки у plots/.

2) Кластеризація та класифікація брендів
python Practical/prac_ml_brands.py


Дає:

K-Means (3 кластери: преміум/мас-маркет/нішеві) + метрики Silhouette / DB / CH,

RandomForest-класифікацію (High/Medium/Low efficiency) з accuracy/F1,

важливість ознак (ROI, NPS, OnlineShare тощо).

3) Оптимізація (опціонально)
python Practical/prac_optimization.py


Приклади:

GA/DE — розклад маркетингового бюджету,

ACO — маршрути доставки (склад → магазини).

Якщо файли мають інші назви — запусти відповідний скрипт у Practical/ (імена залежать від твоїх робіт).

Структура даних

Скрипти працюють із синтетичними або CSV у форматі:

date,sales,roi,nps,online_share,marketing_spend,retention
2023-01-31,120,2.1,55,0.62,35.0,0.48
...


Дати — кінець місяця (частота ME).

Числові поля — десяткові.

Вхідний шлях до CSV можна змінити через аргументи CLI або константу у файлі.

Типові задачі курсу (на базі кейсу з косметикою)

Часові ряди: виявлення трендів/сезонності та прогноз продажів.

Кластеризація: сегментація брендів за ефективністю/поведінкою.

Класифікація: прогноз класу ефективності (High/Medium/Low).

Оптимізація: бюджет промо, маршрути логістики (ACO/GA).

Аналітика: кореляції «маркетинг → продажі», важливість ознак.
