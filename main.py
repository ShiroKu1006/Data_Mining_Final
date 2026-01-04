from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA


DATA_DIR = Path("data")
FEAT_PATH = DATA_DIR / "features" / "account_features_v1.csv"
ALERT_PATH = DATA_DIR / "processed" / "alerts_clean.csv"

OUT_DIR = Path("cluster_outputs")
OUT_DIR.mkdir(exist_ok=True)

OUT_WITH_CLUSTER = OUT_DIR / "account_features_with_cluster.csv"
OUT_PROFILE = OUT_DIR / "cluster_profile_mean.csv"
OUT_CROSSTAB = OUT_DIR / "cluster_label_ratio.csv"
OUT_K_SELECT = OUT_DIR / "k_selection_inertia.csv"
OUT_PCA_PLOT = OUT_DIR / "pca_clusters_sample.png"

DEFAULT_FEATURES = [
    "txn_cnt",
    "txn_cnt_per_day",
    "mean_amt",
    "std_amt",
    "night_txn_ratio",
    "self_txn_ratio",
    "cross_bank_ratio",
    "foreign_currency_ratio",
    "channel_entropy",
    "system_txn_ratio",
]


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到檔案：{path}")
    return pd.read_csv(path, encoding="utf-8", low_memory=False)


def prepare_dataset(features_df: pd.DataFrame, alerts_df: pd.DataFrame) -> pd.DataFrame:
    alerts_df = alerts_df.copy()
    if "label" not in alerts_df.columns:
        alerts_df["label"] = 1
    alerts_df = alerts_df[["acct", "label"]].drop_duplicates(subset=["acct"])

    df = features_df.merge(alerts_df, left_on="from_acct", right_on="acct", how="left")
    df["label"] = df["label"].fillna(0).astype(int)
    if "acct" in df.columns:
        df = df.drop(columns=["acct"])
    return df


def pick_feature_columns(df: pd.DataFrame, wanted: list[str]) -> list[str]:
    cols = [c for c in wanted if c in df.columns]
    if not cols:
        raise ValueError("找不到任何可用的分群特徵欄位，請確認 account_features_v1.csv 是否包含特徵欄位。")
    return cols


def select_k_by_inertia(
    X_sample_scaled: np.ndarray,
    k_min: int = 2,
    k_max: int = 8,
    random_state: int = 42,
    batch_size: int = 8192,
) -> pd.DataFrame:
    rows = []
    for k in range(k_min, k_max + 1):
        mbk = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            batch_size=batch_size,
            n_init="auto",
            max_no_improvement=20,
        )
        mbk.fit(X_sample_scaled)
        rows.append({"k": k, "inertia": float(mbk.inertia_)})
    return pd.DataFrame(rows)


def fit_full_mbkmeans(
    X_scaled: np.ndarray,
    k: int,
    random_state: int = 42,
    batch_size: int = 8192,
) -> np.ndarray:
    mbk = MiniBatchKMeans(
        n_clusters=k,
        random_state=random_state,
        batch_size=batch_size,
        n_init="auto",
        max_no_improvement=20,
    )
    return mbk.fit_predict(X_scaled)


def cluster_profiles(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    return df.groupby("cluster")[feature_cols].mean().reset_index()


def cluster_label_ratio(df: pd.DataFrame) -> pd.DataFrame:
    ct = pd.crosstab(df["cluster"], df["label"], normalize="index")
    ct = ct.rename(columns={0: "label_0_ratio", 1: "label_1_ratio"}).reset_index()
    if "label_0_ratio" not in ct.columns:
        ct["label_0_ratio"] = 0.0
    if "label_1_ratio" not in ct.columns:
        ct["label_1_ratio"] = 0.0
    return ct[["cluster", "label_0_ratio", "label_1_ratio"]]


def plot_pca_on_sample(df_sample: pd.DataFrame, X_sample_scaled: np.ndarray, out_path: Path) -> None:
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_sample_scaled)
    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], c=df_sample["cluster"], s=10)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Projection of Clusters (Sample)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    feat_df = safe_read_csv(FEAT_PATH)
    alerts_df = safe_read_csv(ALERT_PATH)
    df = prepare_dataset(feat_df, alerts_df)

    feature_cols = pick_feature_columns(df, DEFAULT_FEATURES)

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    n = X_scaled.shape[0]
    rng = np.random.default_rng(42)

    sample_size_for_k = min(100_000, n)
    idx_k = rng.choice(n, size=sample_size_for_k, replace=False)
    X_k = X_scaled[idx_k]

    k_metrics = select_k_by_inertia(X_k, k_min=2, k_max=8, random_state=42)
    k_metrics.to_csv(OUT_K_SELECT, index=False, encoding="utf-8-sig")

    k_metrics["delta_inertia"] = k_metrics["inertia"].shift(1) - k_metrics["inertia"]
    k_metrics["delta_inertia"] = k_metrics["delta_inertia"].fillna(0.0)

    best_k = 4
    if len(k_metrics) >= 3:
        d = k_metrics["delta_inertia"].values
        best_k = int(k_metrics.loc[np.argmax(d[2:]) + 2, "k"])

    df["cluster"] = fit_full_mbkmeans(X_scaled, k=best_k, random_state=42)

    df.to_csv(OUT_WITH_CLUSTER, index=False, encoding="utf-8-sig")

    prof = cluster_profiles(df, feature_cols)
    prof.to_csv(OUT_PROFILE, index=False, encoding="utf-8-sig")

    ratio = cluster_label_ratio(df)
    ratio.to_csv(OUT_CROSSTAB, index=False, encoding="utf-8-sig")

    sample_size_plot = min(50_000, n)
    idx_p = rng.choice(n, size=sample_size_plot, replace=False)
    df_sample = df.iloc[idx_p].copy()
    X_p = X_scaled[idx_p]
    plot_pca_on_sample(df_sample, X_p, OUT_PCA_PLOT)


if __name__ == "__main__":
    main()
