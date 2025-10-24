# ===============================================
# Лабораторна робота з аналізу часових рядів
#Тема: Рекомендація інтенсивності косметичного засобу за жирністю та чутливістю шкіри
# Автор: Коваленко Владислава Володимирівна
# Мова: Python 3.13
# ===============================================
import numpy as np
import matplotlib.pyplot as plt
import warnings
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# 1) вхідні та вихідні змінні
# oiliness жирність шкіри, 0-10 (0-суха, 10-дуже жирна)
# sensitivity чутливість шкіри, 0-10 (0-нечутлива, 10-чутлива)
# product_intensity інтенсивність засобу, 0-10 (0-мяка, 10-активна
oiliness = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'oiliness')
sensitivity = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'sensitivity')
product_intensity = ctrl.Consequent(np.arange(0, 10.1, 0.1), 'product_intensity')

# жирність шкіри
oiliness['low']  = fuzz.trimf(oiliness.universe,  [0, 0, 4])
oiliness['mid']  = fuzz.trimf(oiliness.universe,  [2, 5, 8])
oiliness['high'] = fuzz.trimf(oiliness.universe,  [6, 10, 10])

# чутливість шкіри
sensitivity['low']  = fuzz.trimf(sensitivity.universe, [0, 0, 4])
sensitivity['mid']  = fuzz.trimf(sensitivity.universe, [2, 5, 8])
sensitivity['high'] = fuzz.trimf(sensitivity.universe, [6, 10, 10])

# інтенсивність засобу
product_intensity['very_gentle'] = fuzz.trimf(product_intensity.universe, [0, 0, 3])
product_intensity['gentle']      = fuzz.trimf(product_intensity.universe, [1, 3.5, 5])
product_intensity['balanced']    = fuzz.trimf(product_intensity.universe, [4, 5.5, 7])
product_intensity['active']      = fuzz.trimf(product_intensity.universe, [6, 8, 9.5])
product_intensity['very_active'] = fuzz.trimf(product_intensity.universe, [8, 10, 10])


rules = [
    # чутлива шкіра — м’які формули
    ctrl.Rule(sensitivity['high'] & oiliness['low'],  product_intensity['very_gentle']),
    ctrl.Rule(sensitivity['high'] & oiliness['mid'],  product_intensity['gentle']),
    ctrl.Rule(sensitivity['high'] & oiliness['high'], product_intensity['gentle']),

    # низька чутливість — можна інтенсивніше
    ctrl.Rule(sensitivity['low'] & oiliness['low'],   product_intensity['gentle']),
    ctrl.Rule(sensitivity['low'] & oiliness['mid'],   product_intensity['balanced']),
    ctrl.Rule(sensitivity['low'] & oiliness['high'],  product_intensity['very_active']),

    # середня чутливість — дивимось на жирність
    ctrl.Rule(sensitivity['mid'] & oiliness['low'],   product_intensity['very_gentle']),
    ctrl.Rule(sensitivity['mid'] & oiliness['mid'],   product_intensity['balanced']),
    ctrl.Rule(sensitivity['mid'] & oiliness['high'],  product_intensity['active']),
]

system = ctrl.ControlSystem(rules)
simulator = ctrl.ControlSystemSimulation(system)

# приклад 1: жирність=7.5 (висока), чутливість=3 (низька)
simulator.input['oiliness'] = 7.5
simulator.input['sensitivity'] = 3.0
simulator.compute()
result1 = simulator.output['product_intensity']

# приклад 2: жирність=3 (низька), чутливість=8 (висока)
simulator2 = ctrl.ControlSystemSimulation(system)
simulator2.input['oiliness'] = 3.0
simulator2.input['sensitivity'] = 8.0
simulator2.compute()
result2 = simulator2.output['product_intensity']

print("Приклад 1 (oiliness=7.5, sensitivity=3.0) → рекомендована інтенсивність:", round(result1, 2))
print("Приклад 2 (oiliness=3.0, sensitivity=8.0) → рекомендована інтенсивність:", round(result2, 2))

def plot_mf(var, title):
    plt.figure(figsize=(7, 4))
    for term_name, mf in var.terms.items():
        plt.plot(var.universe, mf.mf, label=term_name)
    plt.title(title)
    plt.xlabel(var.label)
    plt.ylabel('μ')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_mf(oiliness, "Функції належності: Жирність шкіри")
plot_mf(sensitivity, "Функції належності: Чутливість шкіри")
plot_mf(product_intensity, "Функції належності: Інтенсивність засобу")

O, S = np.meshgrid(np.linspace(0, 10, 51), np.linspace(0, 10, 51))
Z = np.zeros_like(O)

# щоб не створювати тисячі симуляторів, використаємо один і перезаписуватимемо значення.
sim_surface = ctrl.ControlSystemSimulation(system, flush_after_run=51*51 + 1)

for i in range(O.shape[0]):
    for j in range(O.shape[1]):
        sim_surface.input['oiliness'] = O[i, j]
        sim_surface.input['sensitivity'] = S[i, j]
        sim_surface.compute()
        Z[i, j] = sim_surface.output['product_intensity']

try:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(O, S, Z, linewidth=0, antialiased=True, alpha=0.85)
    ax.set_title("Поверхня відгуку: oiliness × sensitivity → product_intensity")
    ax.set_xlabel("oiliness (жирність)")
    ax.set_ylabel("sensitivity (чутливість)")
    ax.set_zlabel("product_intensity (інтенсивність)")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("3D візуалізація недоступна:", e)

def map_intensity_to_product(y):
    if y <= 2.5:
        return "very_gentle"
    if y <= 4.0:
        return "gentle"
    if y <= 6.5:
        return "balanced"
    if y <= 8.5:
        return "active"
    return "very_active"

print("Категорія (приклад 1):", map_intensity_to_product(result1))
print("Категорія (приклад 2):", map_intensity_to_product(result2))
