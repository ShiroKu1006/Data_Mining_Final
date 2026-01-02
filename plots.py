from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("data")
PROC_DIR = DATA_DIR / "processed"
FEAT_DIR = DATA_DIR / "features"
OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)

def plot_txn_hour_distribution():
    df = pd.read_csv(PROC_DIR / "transactions_clean.csv")

    if "txn_time_sec" not in df.columns:
        raise ValueError("transactions_clean.csv 缺少 txn_time_sec")

    df["hour"] = (df["txn_time_sec"] // 3600).astype("Int64")

    cnt = df["hour"].value_counts().sort_index()

    plt.figure()
    cnt.plot(kind="bar")
    plt.xlabel("Hour of Day")
    plt.ylabel("Transaction Count")
    plt.title("Transaction Hour Distribution (After Cleaning)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "txn_hour_distribution.png")
    plt.close()

def plot_txn_amount_distribution():
    df = pd.read_csv(PROC_DIR / "transactions_clean.csv")

    amt = pd.to_numeric(df["txn_amt"], errors="coerce").dropna()

    plt.figure()
    plt.hist(np.log1p(amt), bins=50)
    plt.xlabel("log(1 + txn_amt)")
    plt.ylabel("Frequency")
    plt.title("Log-scaled Transaction Amount Distribution")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "txn_amount_log_distribution.png")
    plt.close()

def plot_feature_box_by_label(feature_name: str, filename: str):
    feat = pd.read_csv(FEAT_DIR / "account_features_v1.csv")
    alert = pd.read_csv(PROC_DIR / "alerts_clean.csv")[["acct", "label"]]

    df = feat.merge(alert, left_on="from_acct", right_on="acct", how="left")
    df["label"] = df["label"].fillna(0)

    g0 = df[df["label"] == 0][feature_name]
    g1 = df[df["label"] == 1][feature_name]

    plt.figure()
    plt.boxplot([g0, g1], tick_labels=["Normal (0)", "Alert (1)"], showfliers=False)
    plt.ylabel(feature_name)
    plt.title(f"{feature_name} by Account Label")
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename)
    plt.close()

def plot_channel_ratio_comparison():
    feat = pd.read_csv(FEAT_DIR / "account_features_v1.csv")
    alert = pd.read_csv(PROC_DIR / "alerts_clean.csv")[["acct", "label"]]

    df = feat.merge(alert, left_on="from_acct", right_on="acct", how="left")
    df["label"] = df["label"].fillna(0)
    channel_cols = [
        "atm_ratio", "counter_ratio", "mobile_bank_ratio",
        "web_bank_ratio", "epay_ratio", "system_txn_ratio"
    ]
    mean_by_label = df.groupby("label")[channel_cols].mean()
    mean_by_label.T.plot(kind="bar")
    plt.xlabel("Channel Type")
    plt.ylabel("Average Ratio")
    plt.title("Average Channel Usage Ratio by Label")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "channel_ratio_by_label.png")
    plt.close()

def main():
    plot_txn_hour_distribution()
    plot_txn_amount_distribution()
    plot_feature_box_by_label(
        feature_name="night_txn_ratio",
        filename="night_txn_ratio_boxplot.png"
    )
    plot_feature_box_by_label(
        feature_name="std_amt",
        filename="std_amt_boxplot.png"
    )
    plot_channel_ratio_comparison()

if __name__ == "__main__":
    main()
