import os
import time
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from typing import Optional, List
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from pydantic import BaseModel

# --- RUTAS DE ARCHIVOS ---
MODEL_DIR = "model"
PRED_PATH = os.path.join(MODEL_DIR, "pred_df.joblib")
INTER_PATH = os.path.join(MODEL_DIR, "interaction.joblib")
POP_MODEL_PATH = os.path.join(MODEL_DIR, "popular_model.pkl")
MBA_RULES_PATH = os.path.join(MODEL_DIR, "mba_rules.joblib")
DATA_PATH = "csv/df_final.csv"

app = FastAPI(title="Recomendador SVD (pasillos) + Popularidad + MBA")

# --- ENTRENAMIENTO DEL MODELO SVD (basado en pasillos) ---
def train_and_save_model():
    print("📦 Cargando datos...")
    df = pd.read_csv(DATA_PATH)
    assert {'user_id', 'aisle_id', 'reordered'}.issubset(df.columns), "Faltan columnas necesarias"

    interaction = df.groupby(['user_id', 'aisle_id'])['reordered'].sum().unstack(fill_value=0)
    interaction_centered = interaction.sub(interaction.mean(axis=1), axis=0)
    user_means = interaction.mean(axis=1)

    sparse_mat = csr_matrix(interaction_centered.fillna(0).values.astype(float))

    print("⚙️ Entrenando modelo SVD con normalización sobre **pasillos**...")
    start = time.time()
    k = min(50, min(sparse_mat.shape) - 1)
    U, s, Vt = svds(sparse_mat, k=k)
    sigma = np.diag(s)
    pred_mat = U @ sigma @ Vt + user_means.values.reshape(-1, 1)
    fit_time = time.time() - start

    pred_df = pd.DataFrame(pred_mat, index=interaction.index, columns=interaction.columns)

    mask = interaction > 0
    y_true = interaction.values[mask.values]
    y_pred = pred_mat[mask.values]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    y_true_binary = (y_true >= 1).astype(int)
    y_pred_binary = (y_pred >= 0.5).astype(int)
    accuracy = accuracy_score(y_true_binary, y_pred_binary)

    print(f"✅ Modelo SVD (pasillos) entrenado en {fit_time:.2f}s — RMSE: {rmse:.4f} — MAE: {mae:.4f} — Accuracy: {accuracy:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pred_df, PRED_PATH)
    joblib.dump(interaction, INTER_PATH)

# --- ENTRENAMIENTO MODELO DE POPULARIDAD ---
def train_popularity_model():
    print("📊 Entrenando modelo supervisado de popularidad...")
    df = pd.read_csv(DATA_PATH)

    product_counts = df.groupby('product_id')['reordered'].sum().reset_index()
    threshold = product_counts['reordered'].quantile(0.90)
    product_counts['popular'] = (product_counts['reordered'] >= threshold).astype(int)

    features = df.groupby('product_id').agg({
        'add_to_cart_order': 'mean',
        'order_hour_of_day': 'mean',
        'order_dow': 'mean',
        'days_since_prior_order': 'mean',
        'aisle_id': 'first',
        'department_id': 'first'
    }).reset_index()

    final_df = features.merge(product_counts[['product_id', 'popular']], on='product_id')
    X = final_df.drop(columns=['product_id', 'popular'])
    y = final_df['popular']

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    y_pred = clf.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print("🎉 Modelo de popularidad guardado.")
    print(f"📊 Métricas — Accuracy: {accuracy:.4f} — Precision: {precision:.4f} — Recall: {recall:.4f} — F1: {f1:.4f}")

    joblib.dump((clf, final_df[['product_id']]), POP_MODEL_PATH)

# --- ENTRENAMIENTO DE MARKET BASKET ANALYSIS ---
def train_mba_model():
    print("🧠 Entrenando modelo Market Basket Analysis...")
    df = pd.read_csv(DATA_PATH, usecols=['order_id', 'aisle_id'])
    grouped = df.groupby('order_id')['aisle_id'].apply(list).tolist()

    te = TransactionEncoder()
    te_ary = te.fit(grouped).transform(grouped)
    df_tf = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df_tf, min_support=0.08, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

    joblib.dump(rules, MBA_RULES_PATH)
    print(f"🔗 Reglas generadas: {len(rules)}")

    # Imprimir las 5 reglas con mayor lift
    top_rules = rules.sort_values(by="lift", ascending=False).head(5)
    print("\n📈 Top 5 reglas por lift:")
    for _, row in top_rules.iterrows():
        print(f"  Si {set(row['antecedents'])} => Entonces {set(row['consequents'])} "
              f"(Soporte: {row['support']:.3f}, Confianza: {row['confidence']:.3f}, Lift: {row['lift']:.3f})")

# --- CARGA DE MODELOS DESDE DISCO ---
if not os.path.exists(PRED_PATH) or not os.path.exists(INTER_PATH):
    train_and_save_model()
if not os.path.exists(POP_MODEL_PATH):
    train_popularity_model()
if not os.path.exists(MBA_RULES_PATH):
    train_mba_model()

pred_df = joblib.load(PRED_PATH)
interaction = joblib.load(INTER_PATH)
popular_model, product_ids_df = joblib.load(POP_MODEL_PATH)
mba_rules = joblib.load(MBA_RULES_PATH)

# --- FUNCIONES DE RECOMENDACIÓN ---
def recommend_svd(user_id: int, n: int = 5) -> Optional[List[int]]:
    if user_id not in pred_df.index:
        return None
    seen = interaction.loc[user_id]
    seen = set(seen[seen > 0].index)
    scores = pred_df.loc[user_id]
    recs = scores.drop(labels=seen).sort_values(ascending=False)
    return recs.head(n).index.tolist()

def recommend_popular_model(n: int = 10):
    df = pd.read_csv(DATA_PATH)
    features = df.groupby('product_id').agg({
        'add_to_cart_order': 'mean',
        'order_hour_of_day': 'mean',
        'order_dow': 'mean',
        'days_since_prior_order': 'mean',
        'aisle_id': 'first',
        'department_id': 'first'
    }).reset_index()
    X_all = features.drop(columns=['product_id'])
    probs = popular_model.predict_proba(X_all)[:, 1]
    product_ids = features['product_id']
    top_n = product_ids[np.argsort(-probs)][:n]
    return top_n.tolist()

def recommend_by_mba(input_aisles: List[int], top_n: int = 5) -> List[int]:
    recommended = set()

    for aisle in input_aisles:
        matched_rules = mba_rules[
            mba_rules['antecedents'].apply(lambda x: aisle in x)
        ]
        matched_rules = matched_rules.sort_values(by='lift', ascending=False)

        for _, row in matched_rules.iterrows():
            consequent = row['consequents']
            recommended.update(consequent)

    # Eliminar pasillos ya presentes en la cesta
    filtered = list(recommended - set(input_aisles))[:top_n]
    return filtered


# --- FASTAPI ENDPOINTS ---
@app.get("/")
def root():
    return {"message": "Sistema de recomendación activo con SVD (pasillos), popularidad y MBA"}

@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = 5):
    start = time.time()
    recs = recommend_svd(user_id, n)
    latency = time.time() - start
    if recs is None:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return {
        "user_id": user_id,
        "n": n,
        "recommendations": recs,
        "inference_time_s": round(latency, 4)
    }

@app.get("/popular_products_model")
def get_popular_model(n: int = 10):
    start = time.time()
    recs = recommend_popular_model(n)
    latency = time.time() - start
    return {
        "method": "supervised_popularity_model",
        "top_n": n,
        "recommendations": recs,
        "inference_time_s": round(latency, 4)
    }

@app.get("/users")
def get_all_users():
    return {"available_user_ids": pred_df.index.tolist()}

@app.get("/aisle/{aisle_id}/products")
def get_products_by_aisle(aisle_id: int, n: int = 10):
    try:
        df = pd.read_csv(DATA_PATH, usecols=['product_id', 'aisle_id'])
        productos = (
            df[df['aisle_id'] == aisle_id]['product_id']
            .dropna()
            .drop_duplicates()
            .head(n)
            .tolist()
        )
        if not productos:
            raise ValueError
        return {
            "aisle_id": aisle_id,
            "n": n,
            "product_ids": productos
        }
    except:
        raise HTTPException(status_code=404, detail="Pasillo no encontrado o sin productos disponibles.")

class BasketRequest(BaseModel):
    aisle_ids: List[int]
    n: Optional[int] = 5

@app.post("/basket/recommend")
def basket_recommend(request: BasketRequest):
    recommendations = recommend_by_mba(request.aisle_ids, request.n)
    if not recommendations:
        raise HTTPException(status_code=404, detail="No se encontraron recomendaciones asociadas.")
    return {
        "input_aisle_ids": request.aisle_ids,
        "recommendations": recommendations
    }
