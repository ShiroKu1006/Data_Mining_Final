import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

# ---------------------------------------------------------
# 設定：解決圖表中文字型可能出現的亂碼問題 (選用)
# Windows 用戶通常設為 'Microsoft JhengHei' (微軟正黑體)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
# ---------------------------------------------------------

def main():
    # ---------------------------------------------------------
    # 1. 資料讀取
    # ---------------------------------------------------------
    # 建議使用相對路徑，確保你的終端機是在專案根目錄執行
    file_path = os.path.join('data', 'features', 'account_features_v1.csv')

    if not os.path.exists(file_path):
        print(f"❌ 錯誤：找不到檔案 {file_path}")
        print("💡 請確認你的終端機路徑是否在專案根目錄，且檔案路徑正確。")
        return

    df = pd.read_csv(file_path)
    print(f"✅ 成功讀取資料，共 {df.shape[0]} 筆帳戶，{df.shape[1]} 個欄位。")

    # ---------------------------------------------------------
    # 2. 特徵選取 (Feature Selection)
    # ---------------------------------------------------------
    selected_features = [
        'mobile_bank_ratio', 'channel_entropy', 'txn_cnt', 
        'total_amt', 'std_amt', 'max_amt', 
        'active_days', 'night_cnt', 
        'min_txn_gap', 'std_txn_gap', 'max_txn_per_day'
    ]

    # 檢查欄位是否存在，避免報錯
    missing_cols = [col for col in selected_features if col not in df.columns]
    if missing_cols:
        print(f"❌ 資料缺少以下欄位，無法執行：{missing_cols}")
        return

    X = df[selected_features].copy()

    # ---------------------------------------------------------
    # 3. 資料前處理 (Preprocessing)
    # ---------------------------------------------------------
    print("\n🔄 正在進行資料前處理...")

    # 填補空值
    X = X.fillna(0)

    # Log Transform
    log_cols = [
        'total_amt', 'max_amt', 'std_amt',
        'min_txn_gap', 'std_txn_gap'
    ]

    # 檢查 log_cols 是否都在 X 裡
    valid_log_cols = [c for c in log_cols if c in X.columns]

    for col in valid_log_cols:
        # 確保數值非負
        mask = X[col] < 0
        if mask.any():
            print(f"⚠️ 警告：{col} 含有負值，將被視為 0 處理。")
            X.loc[mask, col] = 0
        X[col] = np.log1p(X[col])

    # StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------------------------------------------------------
    # 3.5 手肘法 (Elbow Method) - 互動式
    # ---------------------------------------------------------
    # print("\n📉 正在計算手肘法 (這可能需要一點時間)...")

    # k_range = range(2, 16)
    # inertias = []

    # for k in k_range:
    #     kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    #     kmeans_test.fit(X_scaled)
    #     inertias.append(kmeans_test.inertia_)

    # # 繪圖
    # plt.figure(figsize=(10, 6))
    # plt.plot(k_range, inertias, marker='o', linestyle='-', color='b')
    # plt.title('Elbow Method (Inertia vs K)')
    # plt.xlabel('Number of Clusters (K)')
    # plt.ylabel('Inertia')
    # plt.grid(True, alpha=0.3)
    # plt.xticks(k_range)

    # print("\n👀 手肘法圖表已開啟，請查看視窗。")
    # print("💡 關閉圖表視窗後，程式會繼續執行...")

    # plt.show() # 程式會在這裡暫停，直到你關閉圖表視窗

    # --------------------------------------------------------
    # 4. K-Means 分群 (互動式決定 K)
    # ---------------------------------------------------------
    while True:
        user_input = input("\n👉 請輸入你想使用的 K 值 (預設 6，直接按 Enter 使用預設值): ").strip()
        if user_input == "":
            K = 6
            break
        elif user_input.isdigit() and int(user_input) > 1:
            K = int(user_input)
            break
        else:
            print("❌ 輸入無效，請輸入大於 1 的整數。")

    print(f"\n🚀 正在執行 K-Means 分群 (K={K})...")

    kmeans = KMeans(n_clusters=K, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df['cluster_group'] = cluster_labels

    print("\n📊 各群組帳戶數量統計：")
    print(df['cluster_group'].value_counts().sort_index())

    # ---------------------------------------------------------
    # 5. 結果視覺化 (PCA 降維) - 加強版
    # ---------------------------------------------------------
    print("\n🎨 正在繪製 PCA 分群圖...")

    # 執行 PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 取得解釋變異量 (Explained Variance Ratio)
    # 這告訴我們 PC1 和 PC2 分別解釋了多少原本資料的變異
    explained_variance = pca.explained_variance_ratio_
    pc1_var = explained_variance[0] * 100
    pc2_var = explained_variance[1] * 100

    # -----------------------------------------------------
    # 💡 教授加碼：分析 PC1 和 PC2 到底是由哪些特徵組成的？
    # -----------------------------------------------------
    components = pd.DataFrame(pca.components_, columns=selected_features, index=['PC1', 'PC2'])
    print("\n🔍 PCA 主成分分析解密 (找出影響 X/Y 軸最大的特徵):")

    for i, pc in enumerate(['PC1', 'PC2']):
        print(f"\n--- {pc} (解釋力: {explained_variance[i]*100:.1f}%) 主要受以下特徵影響 ---")

        # 找出影響力絕對值最大的前 3 個特徵
        top_features = components.iloc[i].abs().sort_values(ascending=False).head(3)
        for feature, weight in top_features.items():
            # 顯示原始權重 (正值代表正相關，負值代表負相關)
            raw_weight = components.loc[pc, feature]
            direction = "正向" if raw_weight > 0 else "負向"
            print(f"   * {feature}: {raw_weight:.4f} ({direction})")

    print("-" * 50)

    # -----------------------------------------------------
    # 開始畫圖
    # -----------------------------------------------------
    # plt.figure(figsize=(10, 8))
    # sns.scatterplot(
    #     x=X_pca[:, 0],
    #     y=X_pca[:, 1],
    #     hue=cluster_labels,
    #     palette='viridis',
    #     alpha=0.6,
    #     s=20 # 點稍微大一點比較清楚
    # )

    # # 設定標題與軸名稱 (加入變異量說明)
    # plt.title(f'K-Means Clustering (K={K}) - PCA Projection\nTotal Variance Explained: {pc1_var + pc2_var:.1f}%')
    # plt.xlabel(f'PC1 (Dim 1) - {pc1_var:.1f}% Variance')
    # plt.ylabel(f'PC2 (Dim 2) - {pc2_var:.1f}% Variance')
    # plt.legend(title='Cluster Group')
    # plt.grid(True, alpha=0.3)

    # print("👀 PCA 圖表已開啟，請查看視窗。")
    # print("💡 提示：X 軸數值越大，代表該點在 PC1 的特徵表現越強（請對照上方的解密資訊）。")

    # plt.show()

    # ---------------------------------------------------------
    # 6. 輸出結果
    # ---------------------------------------------------------
    # 確保輸出路徑存在
    output_dir = os.path.dirname(file_path) # 存回原本讀取檔案的目錄
    output_file = os.path.join(output_dir, 'account_features_with_cluster.csv')

    df.to_csv(output_file, index=False)
    print(f"\n✅ 分群結果已儲存為: {output_file}")
    print("你可以開啟此 CSV 檔進行後續分析。")

if __name__ == "__main__":
    main()