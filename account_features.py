from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

IN_PATH = Path("data") / "processed" / "transactions_clean.csv"
OUT_DIR = Path("data") / "features"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "account_features_v1.csv"

CHANNEL_MAP = {
    "01": "atm",
    "02": "counter",
    "03": "mobile_bank",
    "04": "web_bank",
    "05": "voice",
    "06": "eatm",
    "07": "epay",
    "99": "system_txn",
    "UNK": "unk_channel",
}

VALID_CHANNEL_KEYS = set(CHANNEL_MAP.keys())


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    b = b.replace(0, np.nan)
    return (a / b).fillna(0.0)


def entropy_from_counts(counts: pd.DataFrame) -> pd.Series:
    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    eps = 1e-12
    return -(probs * np.log(probs + eps)).sum(axis=1)


def normalize_channel_type(s: pd.Series) -> pd.Series:
    x = s.astype("string").str.strip().str.upper()
    x = x.fillna("UNK")
    x = x.replace({"": "UNK", "NAN": "UNK", "NONE": "UNK"})

    is_num = x.str.fullmatch(r"\d+")
    num = pd.to_numeric(x.where(is_num, np.nan), errors="coerce")

    num = num.astype("Int64")
    num_str = num.astype("string")

    mapped_num = num_str.map(lambda v: f"{int(v):02d}" if v is not pd.NA else pd.NA)
    x = x.where(~is_num, mapped_num)

    x = x.where(x.isin(VALID_CHANNEL_KEYS), "UNK")
    return x


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"找不到檔案：{IN_PATH}")

    df = pd.read_csv(IN_PATH, encoding="utf-8", low_memory=False)

    need_cols = ["from_acct", "txn_amt", "txn_date", "currency_type", "channel_type", "is_self_txn"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"缺少必要欄位：{c}")

    if "txn_ts" not in df.columns and "txn_time_sec" in df.columns and "txn_date" in df.columns:
        df["txn_ts"] = pd.to_numeric(df["txn_date"], errors="coerce") * 86400 + pd.to_numeric(df["txn_time_sec"], errors="coerce")

    df["txn_amt"] = pd.to_numeric(df["txn_amt"], errors="coerce")
    df["txn_date"] = pd.to_numeric(df["txn_date"], errors="coerce").astype("Int64")

    df["is_self_txn"] = df["is_self_txn"].astype("string").str.strip().str.upper().fillna("UNK")
    df.loc[~df["is_self_txn"].isin(["Y", "N", "UNK"]), "is_self_txn"] = "UNK"

    df["currency_type"] = df["currency_type"].astype("string").str.strip().str.upper().fillna("UNK")

    df["channel_type"] = normalize_channel_type(df["channel_type"])
    df["channel_group"] = df["channel_type"].map(CHANNEL_MAP).fillna("unk_channel")

    df["is_cross_bank"] = 0
    if "from_acct_type" in df.columns and "to_acct_type" in df.columns:
        fa = df["from_acct_type"].astype("string").str.strip().str.upper()
        ta = df["to_acct_type"].astype("string").str.strip().str.upper()
        df["is_cross_bank"] = (fa != ta).astype(int)

    if "txn_time_sec" in df.columns:
        tsec = pd.to_numeric(df["txn_time_sec"], errors="coerce")
        df["is_night"] = ((tsec < 6 * 3600) | (tsec >= 22 * 3600)).astype(int)
    else:
        df["is_night"] = 0

    df["is_foreign"] = (df["currency_type"] != "TWD").astype(int)

    base = df.groupby("from_acct", dropna=False).agg(
        txn_cnt=("txn_amt", "size"),
        active_days=("txn_date", pd.Series.nunique),
        total_amt=("txn_amt", "sum"),
        mean_amt=("txn_amt", "mean"),
        max_amt=("txn_amt", "max"),
        std_amt=("txn_amt", "std"),
        p95_amt=("txn_amt", lambda s: s.quantile(0.95) if s.notna().any() else np.nan),
        self_txn_cnt=("is_self_txn", lambda s: (s == "Y").sum()),
        cross_bank_cnt=("is_cross_bank", "sum"),
        night_cnt=("is_night", "sum"),
        foreign_cnt=("is_foreign", "sum"),
    ).reset_index()

    base["txn_cnt_per_day"] = safe_div(base["txn_cnt"], base["active_days"])
    base["self_txn_ratio"] = safe_div(base["self_txn_cnt"], base["txn_cnt"])
    base["cross_bank_ratio"] = safe_div(base["cross_bank_cnt"], base["txn_cnt"])
    base["night_txn_ratio"] = safe_div(base["night_cnt"], base["txn_cnt"])
    base["foreign_currency_ratio"] = safe_div(base["foreign_cnt"], base["txn_cnt"])

    ch_counts = (
        df.pivot_table(index="from_acct", columns="channel_group", values="txn_amt", aggfunc="size", fill_value=0)
        .reset_index()
    )

    for key in CHANNEL_MAP.values():
        if key not in ch_counts.columns:
            ch_counts[key] = 0

    ch_counts["txn_cnt_tmp"] = ch_counts[list(CHANNEL_MAP.values())].sum(axis=1)
    for key in CHANNEL_MAP.values():
        ch_counts[f"{key}_ratio"] = safe_div(ch_counts[key], ch_counts["txn_cnt_tmp"])

    ratio_cols = [f"{k}_ratio" for k in CHANNEL_MAP.values()]
    ent_counts = ch_counts.set_index("from_acct")[list(CHANNEL_MAP.values())]
    ch_entropy = entropy_from_counts(ent_counts).rename("channel_entropy").reset_index()

    ch_feat = ch_counts[["from_acct"] + ratio_cols].merge(ch_entropy, on="from_acct", how="left")

    gap_feat = pd.DataFrame({"from_acct": base["from_acct"]})
    if "txn_ts" in df.columns:
        tmp = df[["from_acct", "txn_ts", "txn_date"]].copy()
        tmp["txn_ts"] = pd.to_numeric(tmp["txn_ts"], errors="coerce")
        tmp = tmp.dropna(subset=["txn_ts"])
        tmp = tmp.sort_values(["from_acct", "txn_ts"])
        tmp["gap_sec"] = tmp.groupby("from_acct")["txn_ts"].diff()

        gap_agg = tmp.groupby("from_acct").agg(
            mean_txn_gap=("gap_sec", "mean"),
            min_txn_gap=("gap_sec", "min"),
            std_txn_gap=("gap_sec", "std"),
        ).reset_index()

        per_day = tmp.groupby(["from_acct", "txn_date"], dropna=False).size().reset_index(name="cnt_day")
        max_day = per_day.groupby("from_acct")["cnt_day"].max().reset_index(name="max_txn_per_day")

        gap_feat = gap_agg.merge(max_day, on="from_acct", how="outer")

    out = base.merge(ch_feat, on="from_acct", how="left").merge(gap_feat, on="from_acct", how="left")

    fill_zero_cols = [
        "std_amt",
        "p95_amt",
        "mean_txn_gap",
        "min_txn_gap",
        "std_txn_gap",
        "max_txn_per_day",
        "channel_entropy",
    ] + [c for c in out.columns if c.endswith("_ratio")]

    for c in fill_zero_cols:
        if c in out.columns:
            out[c] = out[c].fillna(0.0)

    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
