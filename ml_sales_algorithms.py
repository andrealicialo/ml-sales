"""
Ejemplos de Machine Learning para ventas
Incluye:
  1. Random Forest (supervisado)
  2. K-Means Clustering (no supervisado)
  3. Q-Learning (aprendizaje por refuerzo)
Conexión a SQLite (SQL) y gráficas con matplotlib.
"""

import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- Preparación de carpetas ---
os.makedirs("outputs", exist_ok=True)

# --- 1. Generar dataset de ventas y guardar en SQLite ---
np.random.seed(42)
dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=90)
stores = ["North", "South", "East", "West"]
products = ["A", "B", "C"]
rows = []
for d in dates:
    for s in stores:
        for p in products:
            base_price = {"A": 20, "B": 35, "C": 50}[p]
            promo = np.random.binomial(1, 0.2)
            season_factor = 1 + 0.3 * np.sin(2 * np.pi * d.dayofyear / 365)
            units = int(np.random.poisson(30 * season_factor * (1.3 if promo else 1)))
            revenue = units * base_price * (0.9 if promo else 1)
            rows.append((d.strftime("%Y-%m-%d"), s, p, base_price, promo, units, revenue))

df = pd.DataFrame(rows, columns=["date", "store", "product", "price", "promo", "units_sold", "revenue"])

conn = sqlite3.connect("sales.db")
df.to_sql("sales", conn, if_exists="replace", index=False)
conn.commit()
print("Base de datos SQLite creada: sales.db")

# --- 2. RANDOM FOREST (Supervisado) ---
query = """
SELECT date, store, SUM(revenue) AS total_revenue
FROM sales
GROUP BY date, store
"""
daily_sales = pd.read_sql_query(query, conn, parse_dates=["date"])

# Features simples: día de la semana
daily_sales["dayofweek"] = daily_sales["date"].dt.weekday
X = daily_sales[["dayofweek"]]
y = daily_sales["total_revenue"]

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

rf = RandomForestRegressor(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"[Random Forest] MSE: {mse:.2f}  R2: {r2_score(y_test, y_pred):.3f}")

plt.figure()
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicho")
plt.legend()
plt.title("Random Forest - Predicción de revenue por día")
plt.savefig("outputs/random_forest_revenue.png")
plt.close()

# --- 3. K-MEANS CLUSTERING (No supervisado) ---
query2 = """
SELECT store, AVG(revenue) AS avg_rev, AVG(units_sold) AS avg_units, AVG(promo) AS promo_rate
FROM sales
GROUP BY store
"""
store_stats = pd.read_sql_query(query2, conn)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(store_stats[["avg_rev", "avg_units", "promo_rate"]])

kmeans = KMeans(n_clusters=2, random_state=42)
store_stats["cluster"] = kmeans.fit_predict(X_scaled)
print("\n[K-Means] Segmentación de tiendas:")
print(store_stats)

plt.figure()
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=store_stats["cluster"])
for i, name in enumerate(store_stats["store"]):
    plt.text(X_scaled[i, 0], X_scaled[i, 1], name)
plt.title("K-Means - Segmentación de tiendas")
plt.savefig("outputs/kmeans_stores.png")
plt.close()

# --- 4. Q-LEARNING (Aprendizaje por refuerzo) ---
# Escenario: decidir aplicar o no descuento para maximizar ventas
states = ["Low_Demand", "High_Demand"]
actions = ["No_Discount", "Discount"]

Q = np.zeros((len(states), len(actions)))
alpha = 0.1
gamma = 0.9
episodes = 500

for _ in range(episodes):
    state = np.random.choice(states)
    state_idx = states.index(state)

    # Simulación de demanda
    if state == "High_Demand":
        reward_no = 50
        reward_disc = 60
    else:
        reward_no = 20
        reward_disc = 35

    # Elegir acción al azar (exploración)
    action_idx = np.random.choice(len(actions))

    # Obtener recompensa simulada
    reward = reward_disc if action_idx == 1 else reward_no

    # Q-Learning update
    Q[state_idx, action_idx] += alpha * (reward + gamma * np.max(Q[state_idx]) - Q[state_idx, action_idx])

print("\n[Q-Learning] Q-Table aprendida:")
for i, s in enumerate(states):
    print(f"{s}: {dict(zip(actions, Q[i]))}")

plt.figure()
plt.imshow(Q, cmap="Blues")
plt.xticks(range(len(actions)), actions)
plt.yticks(range(len(states)), states)
plt.colorbar(label="Valor Q")
plt.title("Q-Learning - Política de descuentos")
plt.savefig("outputs/qlearning_policy.png")
plt.close()

conn.close()
print("\nEjecución completada. Gráficas en ./outputs")
