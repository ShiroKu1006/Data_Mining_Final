from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = Path("data") / "初賽資料" / "acct_transaction.csv"
OUT_PATH = Path("plots")
OUT_PATH.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH, usecols=["txn_amt"], low_memory=False)

df["txn_amt"] = pd.to_numeric(df["txn_amt"], errors="coerce")
df = df.dropna(subset=["txn_amt"])
df = df[df["txn_amt"] > 0]

upper = df["txn_amt"].quantile(0.995)
df_plot = df[df["txn_amt"] <= upper]

plt.figure()
plt.hist(df_plot["txn_amt"], bins=100)
plt.xlabel("Transaction Amount (Raw)")
plt.ylabel("Frequency")
plt.title("Raw Transaction Amount Distribution")
plt.tight_layout()
plt.savefig(OUT_PATH / "raw_txn_amount_distribution.png")
plt.close()
