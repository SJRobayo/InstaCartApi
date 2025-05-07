import os
import time
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import kmeans, vq
from typing import Optional, List
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from pydantic import BaseModel
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
from math import sqrt

METRICS_DIR = "metrics"
METRICS_PATH = os.path.join(METRICS_DIR, "model_metrics.csv")


# --- RUTAS DE ARCHIVOS ---
MODEL_DIR = "model"
POP_MODEL_PATH = os.path.join(MODEL_DIR, "popular_model.pkl")
MBA_RULES_PATH = os.path.join(MODEL_DIR, "mba_rules.joblib")
RFM_CLUSTER_PATH = os.path.join(MODEL_DIR, "rfm_clusters.joblib")
DATA_PATH = "csv/df_final.csv"

app = FastAPI(title="Recomendador SVD segmentado (KMeans) + Popularidad + MBA")


def save_metrics(model_name: str, metrics: dict):
    os.makedirs(METRICS_DIR, exist_ok=True)
    file_exists = os.path.exists(METRICS_PATH)
    with open(METRICS_PATH, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["timestamp", "model_name"] + list(metrics.keys()))
        if not file_exists:
            writer.writeheader()
        row = {
            "model_name": model_name,
            **metrics
        }
        writer.writerow(row)
# --- RFM KMEANS SEGMENTATION ---
def calculate_rfm_kmeans_segments(df: pd.DataFrame, k_range=range(2, 10)):
    print("🧮 Calculando RFM con KMeans...")
    latest_order = df['order_number'].max()
    rfm = df.groupby('user_id').agg({
        'order_number': 'max',
        'order_id': 'nunique',
        'reordered': 'sum'
    }).rename(columns={
        'order_number': 'recency',
        'order_id': 'frequency',
        'reordered': 'monetary'
    })
    rfm['recency'] = latest_order - rfm['recency']
    scaler = StandardScaler()
    rfm_scaled = pd.DataFrame(scaler.fit_transform(rfm), index=rfm.index, columns=rfm.columns)
    fixed_k = 4
    print(f"✅ Número de clusters fijado manualmente: {fixed_k}")
    centroids, _ = kmeans(rfm_scaled.values, fixed_k)
    cluster_labels, _ = vq(rfm_scaled.values, centroids)
    rfm['cluster'] = cluster_labels
    joblib.dump(rfm[['cluster']], RFM_CLUSTER_PATH)
    return rfm[['cluster']]

def get_or_load_rfm_clusters():
    if os.path.exists(RFM_CLUSTER_PATH):
        return joblib.load(RFM_CLUSTER_PATH)
    else:
        df = pd.read_csv(DATA_PATH)
        return calculate_rfm_kmeans_segments(df)

# --- ENTRENAMIENTO DEL MODELO SVD SEGMENTADO ---
def train_and_save_model_segmented_kmeans():
    print("📦 Cargando datos para segmentación KMeans...")
    df = pd.read_csv(DATA_PATH)
    rfm_clusters = get_or_load_rfm_clusters()
    os.makedirs(MODEL_DIR, exist_ok=True)
    for cluster_id in sorted(rfm_clusters['cluster'].unique()):
        print(f"⚙️ Entrenando modelo para cluster: {cluster_id}")
        users_in_cluster = rfm_clusters[rfm_clusters['cluster'] == cluster_id].index
        df_cluster = df[df['user_id'].isin(users_in_cluster)]
        interaction = df_cluster.groupby(['user_id', 'aisle_id'])['reordered'].sum().unstack(fill_value=0)
        if interaction.empty:
            print(f"⚠️ Cluster {cluster_id} vacío, se omite.")
            continue
        interaction_centered = interaction.sub(interaction.mean(axis=1), axis=0)
        user_means = interaction.mean(axis=1)
        sparse_mat = csr_matrix(interaction_centered.fillna(0).values.astype(float))
        k = min(50, min(sparse_mat.shape) - 1)
        U, s, Vt = svds(sparse_mat, k=k)
        sigma = np.diag(s)
        pred_mat = U @ sigma @ Vt + user_means.values.reshape(-1, 1)
        pred_df = pd.DataFrame(pred_mat, index=interaction.index, columns=interaction.columns)
        joblib.dump(pred_df, os.path.join(MODEL_DIR, f"svd_cluster_{cluster_id}.joblib"))
        print(f"✅ Modelo guardado: svd_cluster_{cluster_id}.joblib")

        # Métricas del modelo SVD
        true_vals = interaction.values.flatten()
        pred_vals = pred_df.values.flatten()
        mse = mean_squared_error(true_vals, pred_vals)
        rmse = sqrt(mse)
        metrics = {
            "num_users": interaction.shape[0],
            "num_products": interaction.shape[1],
            "rmse": rmse
        }
        save_metrics(f"svd_cluster_{cluster_id}", metrics)
        print(f"📈 Métricas SVD cluster {cluster_id}: {metrics}")

# --- DETERMINA EL CLUSTER DEL USUARIO ---
def get_user_cluster(user_id: int) -> int:
    rfm_clusters = get_or_load_rfm_clusters()
    return rfm_clusters.loc[user_id]['cluster'] if user_id in rfm_clusters.index else 0

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

    # Evaluación y métricas
    y_pred = clf.predict(X)
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred)
    }
    save_metrics("popular_model", metrics)
    print(f"📈 Métricas modelo de popularidad: {metrics}")

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

    metrics = {
        "num_rules": len(rules),
        "avg_lift": rules['lift'].mean() if not rules.empty else 0,
        "avg_confidence": rules['confidence'].mean() if not rules.empty else 0
    }
    save_metrics("mba_model", metrics)
    print(f"📈 Métricas MBA: {metrics}")

# --- CARGA DE MODELOS ---
os.makedirs(MODEL_DIR, exist_ok=True)
model_files = [f"svd_cluster_{i}.joblib" for i in range(10)]
if not any(os.path.exists(os.path.join(MODEL_DIR, f)) for f in model_files):
    train_and_save_model_segmented_kmeans()
if not os.path.exists(POP_MODEL_PATH):
    train_popularity_model()
if not os.path.exists(MBA_RULES_PATH):
    train_mba_model()

popular_model, product_ids_df = joblib.load(POP_MODEL_PATH)
mba_rules = joblib.load(MBA_RULES_PATH)





# --- FUNCIONES DE RECOMENDACIÓN ---
def recommend_segmented_svd(user_id: int, n: int = 5) -> Optional[dict]:
    cluster_id = get_user_cluster(user_id)
    model_path = os.path.join(MODEL_DIR, f"svd_cluster_{cluster_id}.joblib")
    if not os.path.exists(model_path):
        return None
    pred_df = joblib.load(model_path)
    if user_id not in pred_df.index:
        return None
    seen = pred_df.loc[user_id]
    seen = set(seen[seen > 0].index)
    scores = pred_df.loc[user_id]
    recs = scores.drop(labels=seen).sort_values(ascending=False)
    return {
        "recommendations": recs.head(n).index.tolist(),
        "cluster": int(cluster_id)
    }


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
            recommended.update(row['consequents'])
    return list(recommended - set(input_aisles))[:top_n]

# --- FASTAPI ENDPOINTS ---
@app.get("/")
def root():
    return {"message": "Sistema de recomendación activo con segmentación RFM KMeans"}

@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = 5):
    start = time.time()
    result = recommend_segmented_svd(user_id, n)
    latency = time.time() - start
    if result is None:
        raise HTTPException(status_code=404, detail="Usuario no encontrado o sin modelo correspondiente")
    return {
        "user_id": user_id,
        "n": n,
        "recommendations": result["recommendations"],
        "cluster": result["cluster"],
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
    df = pd.read_csv(DATA_PATH)
    return {"available_user_ids": df['user_id'].unique().tolist()}

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

@app.post("/retrain/rfm")
def retrain_rfm():
    df = pd.read_csv(DATA_PATH)
    calculate_rfm_kmeans_segments(df)
    return {"status": "RFM clusters reentrenados y guardados"}
