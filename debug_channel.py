import pandas as pd
from pathlib import Path

# 設定你的檔案路徑
FILE_PATH = Path("data") / "processed" / "transactions_clean.csv"

def check_channel():
    if not FILE_PATH.exists():
        print(f"找不到檔案: {FILE_PATH}")
        return

    # 讀取資料
    df = pd.read_csv(FILE_PATH, low_memory=False)
    
    # 印出 channel_type 的前 10 筆不重複值
    unique_vals = df['channel_type'].unique()
    print("Python 看到的 channel_type:", unique_vals[:10])
    
    # 檢查是否包含小數點
    has_dot = any("." in str(x) for x in unique_vals)
    if has_dot:
        print("\n⚠️ 偵測到小數點！這就是問題所在 (例如 '1.0' != '1')")
    else:
        print("\n✅ 資料看起來是乾淨的整數文字")

if __name__ == "__main__":
    check_channel()