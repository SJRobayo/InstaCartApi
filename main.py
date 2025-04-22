import os
import time
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Optional, List

MODEL_DIR = "model"
PRED_PATH = os.path.join(MODEL_DIR, "pred_df.joblib")
INTER_PATH = os.path.join(MODEL_DIR, "interaction.joblib")
DATA_PATH = "csv/definitivo.csv"

app = FastAPI(title="Recomendador SVD Normalizado")


# --- Función para entrenar y guardar el modelo ---
def train_and_save_model():
    print("📦 Cargando datos desde csv/small_sample.csv ...")
    df = pd.read_csv(DATA_PATH)

    assert {'user_id', 'product_id', 'reordered'}.issubset(df.columns), "Faltan columnas necesarias"

    interaction = df.groupby(['user_id', 'product_id'])['reordered'].sum().unstack(fill_value=0)

    interaction_centered = interaction.sub(interaction.mean(axis=1), axis=0)
    user_means = interaction.mean(axis=1)

    sparse_mat = csr_matrix(interaction_centered.fillna(0).values.astype(float))

    print("⚙️ Entrenando modelo SVD con normalización...")
    start = time.time()
    k = min(50, min(sparse_mat.shape) - 1)
    U, s, Vt = svds(sparse_mat, k=k)
    sigma = np.diag(s)
    pred_mat = U @ sigma @ Vt
    pred_mat = pred_mat + user_means.values.reshape(-1, 1)
    fit_time = time.time() - start

    pred_df = pd.DataFrame(pred_mat, index=interaction.index, columns=interaction.columns)

    # Evaluación
    mask = interaction > 0
    y_true = interaction.values[mask.values]
    y_pred = pred_mat[mask.values]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print(f"✅ Modelo entrenado en {fit_time:.2f}s — RMSE: {rmse:.4f} — MAE: {mae:.4f}")

    # Guardar modelo
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pred_df, PRED_PATH)
    joblib.dump(interaction, INTER_PATH)
    print("💾 Modelo guardado en carpeta 'model/'")


# --- Cargar modelo entrenado o entrenar si no existe ---
if not os.path.exists(PRED_PATH) or not os.path.exists(INTER_PATH):
    train_and_save_model()

print("📥 Cargando modelo desde disco...")
pred_df = joblib.load(PRED_PATH)
interaction = joblib.load(INTER_PATH)


# --- Recomendador básico ---
def recommend_svd(user_id: int, n: int = 5) -> Optional[List[int]]:
    if user_id not in pred_df.index:
        return None

    seen = interaction.loc[user_id]
    seen = set(seen[seen > 0].index)

    scores = pred_df.loc[user_id]
    recs = scores.drop(labels=seen).sort_values(ascending=False)

    return recs.head(n).index.tolist()


# --- FastAPI endpoints ---
@app.get("/")
def root():
    return {"message": "Sistema de recomendación activo con modelo cargado desde disco"}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = 5):
    start = time.time()
    recs = recommend_svd(user_id, n)
    latency = time.time() - start

    if recs is None:
        raise HTTPException(status_code=404, detail=f"El usuario {user_id} no existe en el modelo actual.")

    return {
        "user_id": user_id,
        "n": n,
        "recommendations": recs,
        "inference_time_s": round(latency, 4)
    }


@app.get("/users")
def get_all_users():
    """
    Devuelve una lista de user_id válidos presentes en el modelo actual.
    """
    return {"available_user_ids": pred_df.index.tolist()}
