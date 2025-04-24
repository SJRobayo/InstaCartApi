import os
import time
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Optional, List
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from pydantic import BaseModel


# --- RUTAS DE ARCHIVOS ---
MODEL_DIR = "model"
PRED_PATH = os.path.join(MODEL_DIR, "pred_df.joblib")
INTER_PATH = os.path.join(MODEL_DIR, "interaction.joblib")
POP_MODEL_PATH = os.path.join(MODEL_DIR, "popular_model.pkl")
MBA_PATH = os.path.join(MODEL_DIR, "mba_rules.joblib")
DATA_PATH = "csv/definitivo.csv"

app = FastAPI(title="Recomendador SVD + Popularidad + MBA")

# --- ENTRENAMIENTO DEL MODELO SVD ---
def train_and_save_model():
    print("📦 Cargando datos...")
    df = pd.read_csv(DATA_PATH)

    assert {'user_id', 'product_id', 'reordered'}.issubset(df.columns)

    interaction = df.groupby(['user_id', 'product_id'])['reordered'].sum().unstack(fill_value=0)
    interaction_centered = interaction.sub(interaction.mean(axis=1), axis=0)
    user_means = interaction.mean(axis=1)

    sparse_mat = csr_matrix(interaction_centered.fillna(0).values.astype(float))

    print("⚙️ Entrenando modelo SVD con normalización...")
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

    print(f"✅ Modelo SVD entrenado en {fit_time:.2f}s — RMSE: {rmse:.4f} — MAE: {mae:.4f}")

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

    joblib.dump((clf, final_df[['product_id']]), POP_MODEL_PATH)
    print("🎉 Modelo de popularidad guardado.")

# --- ENTRENAMIENTO MODELO MBA ---
def train_mba_model():
    print("🛒 Entrenando modelo de reglas de asociación (MBA)...")
    df = pd.read_csv(DATA_PATH)

    grouped = df.groupby('order_id')['product_id'].apply(list)
    te = TransactionEncoder()
    te_ary = te.fit(grouped).transform(grouped)
    df_tf = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df_tf, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

    rules['antecedents'] = rules['antecedents'].apply(lambda x: next(iter(x)) if len(x) == 1 else None)
    rules['consequents'] = rules['consequents'].apply(lambda x: next(iter(x)) if len(x) == 1 else None)
    rules = rules.dropna(subset=['antecedents', 'consequents'])

    mba_df = rules[['antecedents', 'consequents', 'confidence']]
    joblib.dump(mba_df, MBA_PATH)
    print(f"📁 Reglas MBA guardadas en {MBA_PATH}")

# --- CARGA DE MODELOS DESDE DISCO ---
if not os.path.exists(PRED_PATH) or not os.path.exists(INTER_PATH):
    train_and_save_model()

if not os.path.exists(POP_MODEL_PATH):
    train_popularity_model()

if not os.path.exists(MBA_PATH):
    train_mba_model()

pred_df = joblib.load(PRED_PATH)
interaction = joblib.load(INTER_PATH)
popular_model, product_ids_df = joblib.load(POP_MODEL_PATH)
mba_rules = joblib.load(MBA_PATH)

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
    X_all = product_ids_df.copy()
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

# --- ENDPOINTS FASTAPI ---
@app.get("/")
def root():
    return {"message": "Sistema de recomendación activo con SVD, popularidad y MBA"}

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

class MbaRequest(BaseModel):
    cart: List[int]
    top_k: int = 5
@app.post("/recommend_mba/")
def recommend_mba(request: MbaRequest):
    if mba_rules.empty:
        raise HTTPException(status_code=500, detail="Reglas MBA no entrenadas")

    recommended = set()
    for pid in request.cart:
        matches = mba_rules[mba_rules['antecedents'] == pid]
        matches = matches.sort_values(by="confidence", ascending=False)
        recommended.update(matches['consequents'].tolist())

    final_recs = [pid for pid in recommended if pid not in request.cart][:request.top_k]

    return {
        "cart": request.cart,
        "recommendations": final_recs,
        "rules_considered": len(recommended)
    }

