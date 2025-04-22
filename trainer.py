# app.py

import pandas as pd
import numpy as np
import time
from fastapi import FastAPI
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os

# --- Cargar y procesar datos ---
print("📦 Cargando datos desde csv/final.csv ...")
df = pd.read_csv('csv/final.csv')

assert {'user_id', 'product_id', 'reordered'}.issubset(df.columns), "Faltan columnas necesarias"

# Usar una muestra para rendimiento (puedes ajustar el frac)
df_sample = df.sample(frac=0.01, random_state=42)

# Matriz de interacciones (suma de 'reordered')
interaction = df_sample.groupby(['user_id', 'product_id'])['reordered'].sum().unstack(fill_value=0)

# Normalizar: restar la media por usuario
interaction_centered = interaction.sub(interaction.mean(axis=1), axis=0)
user_means = interaction.mean(axis=1)

# Convertir a formato disperso
sparse_mat = csr_matrix(interaction_centered.fillna(0).values.astype(float))

# --- Entrenar modelo SVD ---
print("⚙️ Entrenando modelo SVD con normalización...")
start = time.time()
k = min(50, min(sparse_mat.shape) - 1)
U, s, Vt = svds(sparse_mat, k=k)
sigma = np.diag(s)
pred_mat = U @ sigma @ Vt

# Desnormalizar: agregar la media del usuario de vuelta
pred_mat = pred_mat + user_means.values.reshape(-1, 1)
fit_time = time.time() - start

# Reconstrucción en DataFrame
pred_df = pd.DataFrame(pred_mat, index=interaction.index, columns=interaction.columns)

# --- Evaluar el modelo ---
mask = interaction > 0
y_true = interaction.values[mask.values]
y_pred = pred_mat[mask.values]

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

print(f"✅ Modelo entrenado en {fit_time:.2f}s — RMSE: {rmse:.4f} — MAE: {mae:.4f}")

os.makedirs("model", exist_ok=True)

# Guardar los objetos necesarios
joblib.dump(pred_df, "model/pred_df.joblib")
joblib.dump(interaction, "model/interaction.joblib")

print("💾 Modelo guardado en 'model/'")

# --- Recomendador básico ---
def recommend_svd(user_id: int, n: int = 5):
    if user_id not in pred_df.index:
        return []

    seen = interaction.loc[user_id]
    seen = set(seen[seen > 0].index)

    scores = pred_df.loc[user_id]
    recs = scores.drop(labels=seen).sort_values(ascending=False)

    return recs.head(n).index.tolist()

# --- FastAPI ---
app = FastAPI(title="Recomendador SVD Normalizado")

@app.get("/")
def root():
    return {"message": "Sistema de recomendación activo con normalización"}

@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = 5):
    start = time.time()
    recs = recommend_svd(user_id, n)
    latency = time.time() - start
    return {
        "user_id": user_id,
        "n": n,
        "recommendations": recs,
        "inference_time_s": round(latency, 4)
    }
