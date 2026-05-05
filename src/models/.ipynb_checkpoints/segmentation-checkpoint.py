# src/models/segmentation.py
# Customer Segmentation — matches Customer_Segmentation.ipynb logic exactly
# Uses M5 RFM features from data/features/rfm_features_m5.parquet
# Algorithm: RobustScaler → PCA(2D) → KMeans (best_k via silhouette)
# Segment labels: Champions, Loyal Customers, Potential, At Risk, Hibernating
# Run: python src/models/segmentation.py

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import os
import logging

from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
RFM_PATH   = "data/features/rfm_features_m5.parquet"
OUT_PATH   = "data/features/rfm_with_segments.parquet"
MODEL_DIR  = "models"
REPORT_DIR = "reports"
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

MLFLOW_URI   = "sqlite:///mlflow.db"
RANDOM_STATE = 42


# ── Segment Label Function (matches notebook exactly) ─────────────────────────
def assign_segment(row, rfm_df):
    f     = row["frequency"]
    m     = row["monetary"]
    med_f = rfm_df["frequency"].median()
    med_m = rfm_df["monetary"].median()
    p75_f = rfm_df["frequency"].quantile(0.75)
    p75_m = rfm_df["monetary"].quantile(0.75)

    if f >= p75_f and m >= p75_m:
        return "Champions"
    elif f >= med_f and m >= med_m:
        return "Loyal Customers"
    elif f >= med_f and m < med_m:
        return "Potential"
    elif f < med_f and m >= med_m:
        return "At Risk"
    elif f < med_f and m < med_m:
        return "Hibernating"
    else:
        return "New Customers"


# ── Load & Preprocess ─────────────────────────────────────────────────────────
def load_and_preprocess(path: str):
    log.info(f"Loading M5 RFM from {path}")
    rfm = pd.read_parquet(path)
    log.info(f"Shape: {rfm.shape}")
    log.info(f"Columns: {list(rfm.columns)}")

    rfm["recency_log"]   = np.log1p(rfm["recency"])
    rfm["frequency_log"] = np.log1p(rfm["frequency"])
    rfm["monetary_log"]  = np.log1p(rfm["monetary"])

    features = ["recency_log", "frequency_log", "monetary_log"]
    X        = rfm[features].copy().fillna(0)

    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, f"{MODEL_DIR}/seg_scaler.pkl")

    pca   = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_.sum() * 100
    log.info(f"PCA Variance Explained: {explained:.1f}%")

    return rfm, X_scaled, X_pca, scaler, pca


# ── Find Best K ───────────────────────────────────────────────────────────────
def find_best_k(X_pca, k_range=range(3, 10)):
    log.info("Finding best K via silhouette score...")
    silhouettes = []
    for k in k_range:
        km  = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=30)
        lbl = km.fit_predict(X_pca)
        silhouettes.append(silhouette_score(X_pca, lbl))

    best_k   = list(k_range)[np.argmax(silhouettes)]
    best_sil = max(silhouettes)
    log.info(f"Best K: {best_k} | Silhouette: {best_sil:.3f}")

    plt.figure(figsize=(8, 4))
    plt.plot(list(k_range), silhouettes, "gs-", linewidth=2)
    plt.axvline(best_k, color="red", linestyle="--", label=f"Best k={best_k}")
    plt.title("Silhouette Score vs K")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.legend()
    plt.tight_layout()
    path = f"{REPORT_DIR}/seg_silhouette_k.png"
    plt.savefig(path, dpi=120)
    plt.close()
    log.info(f"Saved: {path}")

    return best_k, best_sil


# ── PCA Scatter Plot ──────────────────────────────────────────────────────────
def plot_pca_clusters(X_pca, labels, segment_names, title, fname):
    unique_segs = sorted(set(segment_names))
    colors      = plt.cm.tab10(np.linspace(0, 1, len(unique_segs)))
    color_map   = {s: colors[i] for i, s in enumerate(unique_segs)}

    plt.figure(figsize=(10, 7))
    for seg in unique_segs:
        mask = np.array(segment_names) == seg
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    s=15, alpha=0.6, color=color_map[seg], label=seg)

    plt.title(f"{title} — PCA 2D Projection")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(markerscale=3, fontsize=9, loc="best")
    plt.tight_layout()
    path = f"{REPORT_DIR}/{fname}"
    plt.savefig(path, dpi=120)
    plt.close()
    log.info(f"Saved: {path}")


# ── RFM Distribution Plot ─────────────────────────────────────────────────────
def plot_rfm_distribution(rfm):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, color in zip(axes,
                               ["recency", "frequency", "monetary"],
                               ["steelblue", "seagreen", "coral"]):
        ax.hist(rfm[col], bins=30, color=color, edgecolor="white", alpha=0.8)
        ax.set_title(f"{col.capitalize()} Distribution")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
    plt.suptitle("M5 RFM Feature Distributions", fontsize=13)
    plt.tight_layout()
    path = f"{REPORT_DIR}/seg_rfm_distributions.png"
    plt.savefig(path, dpi=120)
    plt.close()
    log.info(f"Saved: {path}")


# ── Main Pipeline ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  NEURALRETAIL — CUSTOMER SEGMENTATION PIPELINE")
    print("=" * 60)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("NeuralRetail_Segmentation")

    rfm, X_scaled, X_pca, scaler, pca = load_and_preprocess(RFM_PATH)
    plot_rfm_distribution(rfm)
    best_k, _ = find_best_k(X_pca)

    with mlflow.start_run(run_name=f"M5_KMeans_PCA_k{best_k}"):
        mlflow.log_param("n_clusters",  best_k)
        mlflow.log_param("scaler",      "RobustScaler")
        mlflow.log_param("algorithm",   "KMeans_PCA")
        mlflow.log_param("data_source", "M5_Forecasting")

        kmeans         = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=30)
        rfm["cluster"] = kmeans.fit_predict(X_pca)

        sil_score = silhouette_score(X_pca, rfm["cluster"])
        db_score  = davies_bouldin_score(X_pca, rfm["cluster"])

        mlflow.log_metric("silhouette_score", round(sil_score, 4))
        mlflow.log_metric("davies_bouldin",   round(db_score,  4))
        mlflow.sklearn.log_model(kmeans, "kmeans_model")
        joblib.dump(kmeans, f"{MODEL_DIR}/kmeans_seg.pkl")

        rfm["segment_name"] = rfm.apply(lambda r: assign_segment(r, rfm), axis=1)

        plot_pca_clusters(
            X_pca,
            rfm["cluster"].tolist(),
            rfm["segment_name"].tolist(),
            f"KMeans k={best_k}",
            "seg_kmeans_pca.png"
        )

        rfm["PCA1"] = X_pca[:, 0]
        rfm["PCA2"] = X_pca[:, 1]

        rfm.to_parquet(OUT_PATH, index=False)
        log.info(f"Saved: {OUT_PATH}")

        mlflow.log_artifact(OUT_PATH)
        mlflow.log_artifact(f"{REPORT_DIR}/seg_kmeans_pca.png")
        mlflow.log_artifact(f"{REPORT_DIR}/seg_silhouette_k.png")

        status = "PRODUCTION_READY" if sil_score >= 0.55 else "NEEDS_TUNING"
        mlflow.log_param("status", status)

    print("\n" + "=" * 60)
    print("  SEGMENTATION RESULTS")
    print("=" * 60)
    print(f"  Best K           : {best_k}")
    print(f"  Silhouette Score : {sil_score:.4f}  (Target >= 0.55)")
    print(f"  Davies-Bouldin   : {db_score:.4f}")
    print(f"  Status           : {status}")
    print(f"\n  Segment Distribution:")
    print(rfm["segment_name"].value_counts().to_string())
    print(f"\n  Output → {OUT_PATH}")
    print("=" * 60)
    if sil_score >= 0.55:
        print(f"\n  TARGET ACHIEVED! ✅ Silhouette: {sil_score:.4f}")
    else:
        print(f"\n  ⚠️  Below target — consider tuning k")
    print("=" * 60)


if __name__ == "__main__":
    main()