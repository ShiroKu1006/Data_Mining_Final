from __future__ import annotations

from pathlib import Path
import pandas as pd


BASE_DIR = Path("data") / "初賽資料"
OUT_DIR = Path("data") / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到檔案：{path}")
    return pd.read_csv(path, encoding="utf-8", low_memory=False)


def _strip_series(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def _hms_to_seconds(time_str: pd.Series) -> pd.Series:
    s = _strip_series(time_str).fillna("")
    parts = s.str.split(":", expand=True)
    sec = pd.Series([pd.NA] * len(s), dtype="Int64")

    if parts.shape[1] == 3:
        h = pd.to_numeric(parts[0], errors="coerce")
        m = pd.to_numeric(parts[1], errors="coerce")
        ss = pd.to_numeric(parts[2], errors="coerce")
        total = h * 3600 + m * 60 + ss
        sec = total.round().astype("Int64")
    else:
        num = pd.to_numeric(s, errors="coerce")
        sec = num.round().astype("Int64")

    sec = sec.where(sec.notna(), pd.NA)
    return sec


def clean_acct_transaction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in [x for x in ["from_acct", "to_acct", "is_self_txn", "currency_type", "channel_type", "txn_time"] if x in df.columns]:
        df[c] = _strip_series(df[c])

    for c in ["from_acct_type", "to_acct_type"]:
        if c in df.columns:
            df[c] = _strip_series(df[c]).str.upper()
            df[c] = df[c].fillna("UNK")
            df.loc[df[c].isin(["", "NAN", "<NA>"]), c] = "UNK"

    if "is_self_txn" in df.columns:
        df["is_self_txn"] = df["is_self_txn"].str.upper()
        df["is_self_txn"] = df["is_self_txn"].fillna("UNK")
        df.loc[~df["is_self_txn"].isin(["Y", "N", "UNK"]), "is_self_txn"] = "UNK"

    if "currency_type" in df.columns:
        df["currency_type"] = df["currency_type"].str.upper()
        df["currency_type"] = df["currency_type"].fillna("UNK")
        df.loc[df["currency_type"].isin(["", "NAN", "<NA>"]), "currency_type"] = "UNK"

    if "channel_type" in df.columns:
        df["channel_type"] = df["channel_type"].str.upper().fillna("UNK")
        df.loc[df["channel_type"].isin(["", "NAN", "<NA>"]), "channel_type"] = "UNK"

    if "txn_amt" in df.columns:
        df["txn_amt"] = pd.to_numeric(df["txn_amt"], errors="coerce")

    if "txn_date" in df.columns:
        df["txn_date"] = pd.to_numeric(df["txn_date"], errors="coerce").astype("Int64")

    if "txn_time" in df.columns:
        df["txn_time_sec"] = _hms_to_seconds(df["txn_time"])

    if "txn_date" in df.columns and "txn_time_sec" in df.columns:
        df["txn_ts"] = df["txn_date"].astype("Int64") * 86400 + df["txn_time_sec"].astype("Int64")

    df = df.drop_duplicates()
    return df


def clean_acct_alert(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "acct" in df.columns:
        df["acct"] = _strip_series(df["acct"])

    if "event_date" in df.columns:
        df["event_date"] = pd.to_numeric(df["event_date"], errors="coerce").astype("Int64")

    df["label"] = 1

    if "acct" in df.columns and "event_date" in df.columns:
        df = df.drop_duplicates(subset=["acct", "event_date"])
    else:
        df = df.drop_duplicates()

    return df


def clean_acct_test(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "acct" in df.columns:
        df["acct"] = _strip_series(df["acct"])
        df = df.drop_duplicates(subset=["acct"])
    else:
        df = df.drop_duplicates()
    return df


def main() -> None:
    txn_df = _read_csv(BASE_DIR / "acct_transaction.csv")
    txn_clean = clean_acct_transaction(txn_df)
    txn_clean.to_csv(OUT_DIR / "transactions_clean.csv", index=False, encoding="utf-8-sig")

    alert_df = _read_csv(BASE_DIR / "acct_alert.csv")
    alert_clean = clean_acct_alert(alert_df)
    alert_clean.to_csv(OUT_DIR / "alerts_clean.csv", index=False, encoding="utf-8-sig")

    test_path = BASE_DIR / "acct_test.csv"
    if test_path.exists():
        test_df = _read_csv(test_path)
        test_clean = clean_acct_test(test_df)
        test_clean.to_csv(OUT_DIR / "acct_test_clean.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
