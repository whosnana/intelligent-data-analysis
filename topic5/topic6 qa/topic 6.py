import mysql.connector
from mysql.connector import Error
import random
from faker import Faker
from contextlib import contextmanager
from datetime import date, timedelta, datetime

DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "1234Nana",
    "charset": "utf8mb4"
}

DB_NAME = "banquet_hall_db"

fake = Faker('uk_UA')
NUM_CLIENTS = 80
NUM_STAFF = 40
NUM_BANQUETS = 150
NUM_PURCHASES = 200

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
    except Error:
        if conn and conn.is_connected():
            conn.rollback()
        raise
    finally:
        if conn and conn.is_connected():
            conn.close()

def create_tables(conn):
    with conn.cursor() as cur:
        cur.execute("SET FOREIGN_KEY_CHECKS = 0;")
        tables = [
            "banquet_staff",
            "payments",
            "banquet_dishes",
            "purchases",
            "banquets",
            "dish_products",
            "dishes",
            "dish_categories",
            "products",
            "staff",
            "clients"
        ]
        for t in tables:
            cur.execute(f"DROP TABLE IF EXISTS {t};")
        cur.execute("SET FOREIGN_KEY_CHECKS = 1;")

        cur.execute("""
        CREATE TABLE dish_categories (
            category_id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100) NOT NULL UNIQUE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        cur.execute("""
        CREATE TABLE dishes (
            dish_id INT AUTO_INCREMENT PRIMARY KEY,
            category_id INT NOT NULL,
            name VARCHAR(150) NOT NULL,
            price DECIMAL(8,2) NOT NULL,
            is_active TINYINT(1) NOT NULL DEFAULT 1,
            FOREIGN KEY (category_id) REFERENCES dish_categories(category_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        cur.execute("""
        CREATE TABLE products (
            product_id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(150) NOT NULL UNIQUE,
            unit VARCHAR(20) NOT NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        cur.execute("""
        CREATE TABLE dish_products (
            dish_id INT NOT NULL,
            product_id INT NOT NULL,
            quantity DECIMAL(8,3) NOT NULL,
            PRIMARY KEY (dish_id, product_id),
            FOREIGN KEY (dish_id) REFERENCES dishes(dish_id) ON DELETE CASCADE,
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        cur.execute("""
        CREATE TABLE clients (
            client_id INT AUTO_INCREMENT PRIMARY KEY,
            full_name VARCHAR(150) NOT NULL,
            phone VARCHAR(30),
            email VARCHAR(100)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        cur.execute("""
        CREATE TABLE staff (
            staff_id INT AUTO_INCREMENT PRIMARY KEY,
            full_name VARCHAR(150) NOT NULL,
            position VARCHAR(100),
            hourly_rate DECIMAL(8,2)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        cur.execute("""
        CREATE TABLE banquets (
            banquet_id INT AUTO_INCREMENT PRIMARY KEY,
            client_id INT NOT NULL,
            event_date DATE NOT NULL,
            hall_name VARCHAR(100),
            guests_count INT NOT NULL,
            status VARCHAR(50),
            total_amount DECIMAL(10,2) DEFAULT 0,
            paid_amount DECIMAL(10,2) DEFAULT 0,
            created_at DATETIME NOT NULL,
            FOREIGN KEY (client_id) REFERENCES clients(client_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        cur.execute("""
        CREATE TABLE banquet_dishes (
            banquet_dish_id INT AUTO_INCREMENT PRIMARY KEY,
            banquet_id INT NOT NULL,
            dish_id INT NOT NULL,
            quantity INT NOT NULL,
            price_per_unit DECIMAL(8,2) NOT NULL,
            subtotal DECIMAL(10,2) NOT NULL,
            FOREIGN KEY (banquet_id) REFERENCES banquets(banquet_id) ON DELETE CASCADE,
            FOREIGN KEY (dish_id) REFERENCES dishes(dish_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        cur.execute("""
        CREATE TABLE payments (
            payment_id INT AUTO_INCREMENT PRIMARY KEY,
            banquet_id INT NOT NULL,
            payment_date DATE NOT NULL,
            amount DECIMAL(10,2) NOT NULL,
            method VARCHAR(30),
            FOREIGN KEY (banquet_id) REFERENCES banquets(banquet_id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        cur.execute("""
        CREATE TABLE purchases (
            purchase_id INT AUTO_INCREMENT PRIMARY KEY,
            product_id INT NOT NULL,
            purchase_date DATE NOT NULL,
            supplier VARCHAR(150),
            quantity DECIMAL(10,2) NOT NULL,
            unit_price DECIMAL(8,2) NOT NULL,
            total_cost DECIMAL(10,2) NOT NULL,
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        cur.execute("""
        CREATE TABLE banquet_staff (
            banquet_id INT NOT NULL,
            staff_id INT NOT NULL,
            role VARCHAR(100),
            hours_worked DECIMAL(5,2),
            PRIMARY KEY (banquet_id, staff_id),
            FOREIGN KEY (banquet_id) REFERENCES banquets(banquet_id) ON DELETE CASCADE,
            FOREIGN KEY (staff_id) REFERENCES staff(staff_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

def populate_data(conn):
    with conn.cursor() as cur:
        categories = ['Салати', 'Гарячі страви', 'Закуски', 'Десерти', 'Напої']
        for c in categories:
            cur.execute("INSERT INTO dish_categories (name) VALUES (%s)", (c,))

        dishes_by_category = {
            'Салати': ['Олів’є', 'Цезар', 'Грецький', 'Вінегрет'],
            'Гарячі страви': ['Курка по-королівськи', 'Свинина BBQ', 'Лосось', 'Стейк'],
            'Закуски': ['М’ясне асорті', 'Сирна тарілка', 'Овочеве асорті', 'Канапе'],
            'Десерти': ['Тірамісу', 'Чізкейк', 'Наполеон', 'Фрукти'],
            'Напої': ['Сік', 'Морс', 'Вода', 'Кава']
        }

        cur.execute("SELECT category_id, name FROM dish_categories;")
        cat_rows = cur.fetchall()
        cat_map = {name: cid for cid, name in cat_rows}

        for cname, items in dishes_by_category.items():
            category_id = cat_map[cname]
            for dish in items:
                price = round(random.uniform(80, 400), 2)
                cur.execute(
                    "INSERT INTO dishes (category_id, name, price) VALUES (%s, %s, %s)",
                    (category_id, dish, price)
                )

        products = [
            ('Куряче філе', 'кг'),
            ('Свинина', 'кг'),
            ('Яловичина', 'кг'),
            ('Лосось', 'кг'),
            ('Сир твердий', 'кг'),
            ('Сир м’який', 'кг'),
            ('Овочі свіжі', 'кг'),
            ('Фрукти свіжі', 'кг'),
            ('Картопля', 'кг'),
            ('Олія рослинна', 'л'),
            ('Сік фруктовий', 'л'),
            ('Вода мінеральна', 'л'),
            ('Кава мелена', 'кг'),
            ('Цукор', 'кг')
        ]
        for name, unit in products:
            cur.execute(
                "INSERT INTO products (name, unit) VALUES (%s, %s)",
                (name, unit)
            )

        cur.execute("SELECT dish_id, price FROM dishes;")
        dish_rows = cur.fetchall()
        dish_ids = [r[0] for r in dish_rows]
        dish_price_map = {dish_id: float(price) for dish_id, price in dish_rows}

        cur.execute("SELECT product_id FROM products;")
        product_ids = [r[0] for r in cur.fetchall()]

        for dish_id in dish_ids:
            used_products = random.sample(product_ids, random.randint(2, 5))
            for product_id in used_products:
                quantity = round(random.uniform(0.05, 1.0), 3)
                cur.execute(
                    "INSERT INTO dish_products (dish_id, product_id, quantity) VALUES (%s, %s, %s)",
                    (dish_id, product_id, quantity)
                )

        for _ in range(NUM_CLIENTS):
            full_name = fake.name()
            phone = fake.phone_number()
            email = fake.email()
            cur.execute(
                "INSERT INTO clients (full_name, phone, email) VALUES (%s, %s, %s)",
                (full_name, phone, email)
            )

        positions = ['Офіціант', 'Кухар', 'Бармен', 'Адміністратор', 'Прибиральник']
        for _ in range(NUM_STAFF):
            full_name = fake.name()
            position = random.choice(positions)
            hourly_rate = round(random.uniform(80, 200), 2)
            cur.execute(
                "INSERT INTO staff (full_name, position, hourly_rate) VALUES (%s, %s, %s)",
                (full_name, position, hourly_rate)
            )

        cur.execute("SELECT client_id FROM clients;")
        client_ids = [r[0] for r in cur.fetchall()]

        cur.execute("SELECT staff_id FROM staff;")
        staff_ids = [r[0] for r in cur.fetchall()]

        start_date = date.today() - timedelta(days=365)
        suppliers = ['ТОВ "Продукти плюс"', 'ФОП Іваненко', 'ТОВ "Свіжі овочі"', 'ФОП Петренко']

        for _ in range(NUM_PURCHASES):
            product_id = random.choice(product_ids)
            purchase_date = start_date + timedelta(days=random.randint(0, 365))
            supplier = random.choice(suppliers)
            quantity = round(random.uniform(5, 50), 2)
            unit_price = round(random.uniform(20, 300), 2)
            total_cost = round(quantity * unit_price, 2)
            cur.execute(
                """
                INSERT INTO purchases (product_id, purchase_date, supplier, quantity, unit_price, total_cost)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (product_id, purchase_date, supplier, quantity, unit_price, total_cost)
            )

        halls = ['Зал "Роял"', 'Зал "Панорама"', 'Зал "Класік"']
        statuses = ['Заплановано', 'Проведено', 'Скасовано']

        for _ in range(NUM_BANQUETS):
            client_id = random.choice(client_ids)
            event_date = start_date + timedelta(days=random.randint(0, 365))
            hall_name = random.choice(halls)
            guests_count = random.randint(20, 150)
            status = random.choices(statuses, weights=[3, 5, 1], k=1)[0]
            created_at = datetime.combine(event_date - timedelta(days=random.randint(5, 60)), datetime.min.time())

            banquet_dish_ids = random.sample(dish_ids, random.randint(3, 7))
            banquet_items = []
            total_amount = 0.0

            for dish_id in banquet_dish_ids:
                quantity = random.randint(int(guests_count * 0.7), int(guests_count * 1.2))
                price_per_unit = dish_price_map[dish_id]
                subtotal = round(quantity * price_per_unit, 2)
                total_amount += subtotal
                banquet_items.append((dish_id, quantity, price_per_unit, subtotal))

            total_amount = round(total_amount, 2)

            if status == 'Скасовано':
                paid_amount = 0.0
            elif status == 'Заплановано':
                paid_amount = round(total_amount * random.uniform(0.2, 0.6), 2)
            else:
                paid_amount = round(total_amount * random.uniform(0.7, 1.0), 2)

            cur.execute(
                """
                INSERT INTO banquets (client_id, event_date, hall_name, guests_count,
                                      status, total_amount, paid_amount, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (client_id, event_date, hall_name, guests_count,
                 status, total_amount, paid_amount, created_at)
            )
            banquet_id = cur.lastrowid

            for dish_id, quantity, price_per_unit, subtotal in banquet_items:
                cur.execute(
                    """
                    INSERT INTO banquet_dishes (banquet_id, dish_id, quantity,
                                                price_per_unit, subtotal)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (banquet_id, dish_id, quantity, price_per_unit, subtotal)
                )

            if paid_amount > 0:
                payments_count = random.randint(1, 3)
                remaining = paid_amount
                for i in range(payments_count):
                    if i == payments_count - 1:
                        amount = remaining
                    else:
                        amount = round(paid_amount * random.uniform(0.2, 0.5), 2)
                        if amount > remaining:
                            amount = remaining
                    if amount <= 0:
                        continue
                    remaining = round(remaining - amount, 2)
                    payment_date = event_date - timedelta(days=random.randint(1, 30))
                    method = random.choice(['Готівка', 'Картка', 'Безготівково'])
                    cur.execute(
                        """
                        INSERT INTO payments (banquet_id, payment_date, amount, method)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (banquet_id, payment_date, amount, method)
                    )

            staff_for_banquet = random.sample(staff_ids, random.randint(3, 7))
            roles_for_staff = ['Старший офіціант', 'Офіціант', 'Кухар', 'Бармен', 'Адміністратор']
            for staff_id in staff_for_banquet:
                role = random.choice(roles_for_staff)
                hours_worked = round(random.uniform(4, 10), 1)
                cur.execute(
                    """
                    INSERT INTO banquet_staff (banquet_id, staff_id, role, hours_worked)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (banquet_id, staff_id, role, hours_worked)
                )

def get_banquet_statistics(conn):
    with conn.cursor() as cur:
        print("\n--- ЗВЕДЕНА СТАТИСТИКА ПО БАНКЕТНОМУ ЗАЛУ ---")
        cur.execute("SELECT COUNT(*) FROM banquets;")
        total_banquets = cur.fetchone()[0]
        print(f"\n1. Загальна кількість банкетів: {total_banquets}")

        print("\n2. Виручка за банкетними залами:")
        cur.execute("""
            SELECT hall_name, SUM(total_amount) AS revenue
            FROM banquets
            GROUP BY hall_name
            ORDER BY revenue DESC;
        """)
        for hall_name, revenue in cur.fetchall():
            print(f"   - {hall_name}: {revenue:.2f} грн")

        print("\n3. Статистика по статусу оплати:")
        cur.execute("""
            SELECT
                CASE
                    WHEN paid_amount = 0 THEN 'Не оплачено'
                    WHEN paid_amount < total_amount THEN 'Частково оплачено'
                    ELSE 'Повністю оплачено'
                END AS pay_status,
                COUNT(*) AS cnt
            FROM banquets
            GROUP BY pay_status
            ORDER BY cnt DESC;
        """)
        for status, cnt in cur.fetchall():
            print(f"   - {status}: {cnt} банкетів")

        print("\n4. Топ-5 страв за кількістю порцій:")
        cur.execute("""
            SELECT d.name, SUM(bd.quantity) AS total_qty
            FROM banquet_dishes bd
            JOIN dishes d ON bd.dish_id = d.dish_id
            GROUP BY d.name
            ORDER BY total_qty DESC
            LIMIT 5;
        """)
        for dish_name, qty in cur.fetchall():
            print(f"   - {dish_name}: {qty} порцій")

        print("\n5. Середній чек банкету та на гостя:")
        cur.execute("""
            SELECT
                AVG(total_amount) AS avg_check,
                AVG(total_amount / NULLIF(guests_count, 0)) AS avg_per_guest
            FROM banquets;
        """)
        avg_check, avg_per_guest = cur.fetchone()
        print(f"   - Середній чек банкету: {avg_check:.2f} грн")
        print(f"   - Середній чек на гостя: {avg_per_guest:.2f} грн")

        print("\n6. Топ-5 продуктів за сумою закупівель:")
        cur.execute("""
            SELECT p.name, SUM(pr.total_cost) AS total_cost
            FROM purchases pr
            JOIN products p ON pr.product_id = p.product_id
            GROUP BY p.name
            ORDER BY total_cost DESC
            LIMIT 5;
        """)
        for prod_name, total_cost in cur.fetchall():
            print(f"   - {prod_name}: {total_cost:.2f} грн")

        print("\n7. Топ-5 співробітників за кількістю банкетів:")
        cur.execute("""
            SELECT s.full_name, COUNT(DISTINCT bs.banquet_id) AS banquets_count
            FROM banquet_staff bs
            JOIN staff s ON bs.staff_id = s.staff_id
            GROUP BY s.full_name
            ORDER BY banquets_count DESC
            LIMIT 5;
        """)
        for full_name, cnt in cur.fetchall():
            print(f"   - {full_name}: {cnt} банкетів")

def main():
    with get_db_connection() as conn:
        create_tables(conn)
        populate_data(conn)
        get_banquet_statistics(conn)

if __name__ == "__main__":
    main()
