# ---------------------------------------------
# ЛР №2: Порівняння Мамдані та Сугено
#Тема:
# Автор: Коваленко Владислава Володимирівна
# Мова: Python 3.13
# ---------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import warnings
import skfuzzy as fuzz
from skfuzzy import control as ctrl
warnings.filterwarnings("ignore")

def demand_function(price, marketing):
    return (15 - 1.2 * price) * np.sin(marketing / 2) + 5 * np.cos(price / 3)

n = 41
price = np.linspace(1, 10, n)
marketing = np.linspace(0, 10, n)
P, M = np.meshgrid(price, marketing)
D_target = demand_function(P, M)
d_min, d_max = np.min(D_target), np.max(D_target)

def tri_mf(a, b, c):
    def mf(z):
        return np.maximum(np.minimum((z - a)/(b - a + 1e-12),(c - z)/(c - b + 1e-12)), 0.0)
    return mf

price_low  = tri_mf(1, 1, 4)
price_mid  = tri_mf(2, 5, 8)
price_high = tri_mf(6, 10, 10)
marketing_low  = tri_mf(0, 0, 4)
marketing_mid  = tri_mf(3, 5, 7)
marketing_high = tri_mf(6, 10, 10)

def sugeno_output(p, m):
    mu_p_low, mu_p_mid, mu_p_high = price_low(p), price_mid(p), price_high(p)
    mu_m_low, mu_m_mid, mu_m_high = marketing_low(m), marketing_mid(m), marketing_high(m)
    a1 = mu_p_mid
    a2 = min(mu_p_high, mu_m_high)
    a3 = min(mu_p_high, mu_m_low)
    a4 = min(mu_p_low,  mu_m_mid)
    a5 = min(mu_p_low,  mu_m_low)
    a6 = min(mu_p_low,  mu_m_high)
    z1 = 5
    z2 = 10 + 2*m - p
    z3 = 6 + p - m
    z4 = 12 - 1.5*p + 1.2*m
    z5 = 3
    z6 = 4
    alphas = np.array([a1, a2, a3, a4, a5, a6], dtype=float)
    zs = np.array([z1, z2, z3, z4, z5, z6], dtype=float)
    s = np.sum(alphas)
    if s <= 1e-12:
        return 0.0
    return float(np.dot(alphas, zs) / s)

D_sugeno = np.zeros_like(D_target)
for i in range(n):
    for j in range(n):
        D_sugeno[i, j] = sugeno_output(P[i, j], M[i, j])

price_universe = np.linspace(1, 10, 201)
marketing_universe = np.linspace(0, 10, 201)
demand_universe  = np.linspace(d_min, d_max, 401)

in_price = ctrl.Antecedent(price_universe, 'price')
in_marketing = ctrl.Antecedent(marketing_universe, 'marketing')
out_demand = ctrl.Consequent(demand_universe, 'demand', defuzzify_method='centroid')

in_price['low']  = fuzz.trimf(in_price.universe, [1, 1, 4])
in_price['mid']  = fuzz.trimf(in_price.universe, [2, 5, 8])
in_price['high'] = fuzz.trimf(in_price.universe, [6, 10, 10])
in_marketing['low']  = fuzz.trimf(in_marketing.universe, [0, 0, 4])
in_marketing['mid']  = fuzz.trimf(in_marketing.universe, [3, 5, 7])
in_marketing['high'] = fuzz.trimf(in_marketing.universe, [6, 10, 10])

span = d_max - d_min
out_demand['very_low']  = fuzz.trimf(out_demand.universe, [d_min, d_min, d_min + 0.2*span])
out_demand['low']       = fuzz.trimf(out_demand.universe, [d_min + 0.1*span, d_min + 0.25*span, d_min + 0.4*span])
out_demand['mid']       = fuzz.trimf(out_demand.universe, [d_min + 0.35*span, d_min + 0.5*span, d_min + 0.65*span])
out_demand['high']      = fuzz.trimf(out_demand.universe, [d_min + 0.6*span, d_min + 0.75*span, d_min + 0.9*span])
out_demand['very_high'] = fuzz.trimf(out_demand.universe, [d_min + 0.8*span, d_max, d_max])

rules = [
    ctrl.Rule(in_price['mid'], out_demand['mid']),
    ctrl.Rule(in_price['high'] & in_marketing['high'], out_demand['high']),
    ctrl.Rule(in_price['high'] & in_marketing['low'],  out_demand['low']),
    ctrl.Rule(in_price['low']  & in_marketing['mid'],  out_demand['high']),
    ctrl.Rule(in_price['low']  & in_marketing['low'],  out_demand['mid']),
    ctrl.Rule(in_price['low']  & in_marketing['high'], out_demand['very_high']),
]

mamdani_system = ctrl.ControlSystem(rules)

def mamdani_predict(p, m, system):
    try:
        sim = ctrl.ControlSystemSimulation(system)
        sim.input['price'] = p
        sim.input['marketing'] = m
        sim.compute()
        return sim.output.get('demand', np.nan)
    except Exception:
        return np.nan

D_mamdani = np.zeros_like(D_target)
for i in range(n):
    for j in range(n):
        D_mamdani[i, j] = mamdani_predict(P[i, j], M[i, j], mamdani_system)

def rmse(a, b):
    return np.sqrt(np.mean((a - b)**2))

rmse_sugeno  = rmse(D_sugeno,  D_target)
rmse_mamdani = rmse(D_mamdani, D_target)
print(f"RMSE (Sugeno vs Target):  {rmse_sugeno:.4f}")
print(f"RMSE (Mamdani vs Target): {rmse_mamdani:.4f}")

def plot_surface(X, Y, Z, title):
    fig = plt.figure(figsize=(7, 5))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax.set_title(title)
    ax.set_xlabel('Price')
    ax.set_ylabel('Marketing')
    ax.set_zlabel('Demand')
    plt.tight_layout()
    plt.show()

plot_surface(P, M, D_target,  "Еталонна поверхня: попит на косметику")
plot_surface(P, M, D_sugeno,  f"Sugeno модель (RMSE={rmse_sugeno:.3f})")
plot_surface(P, M, D_mamdani, f"Mamdani модель (RMSE={rmse_mamdani:.3f})")
