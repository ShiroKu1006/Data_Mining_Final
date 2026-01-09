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
        'txn_cnt', 'active_days', 'mean_amt', 'max_amt', 'std_amt',
        'txn_cnt_per_day', 'self_txn_ratio', 'night_txn_ratio', 'cross_bank_ratio',
        'foreign_currency_ratio', 'channel_entropy',
        'mean_txn_gap', 'min_txn_gap', 'std_txn_gap', 'max_txn_per_day'
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
        'mean_amt', 'max_amt', 'std_amt',
        'mean_txn_gap', 'min_txn_gap', 'std_txn_gap'
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
    # 4. K-Means 第一層分群 (主分群)
    # ---------------------------------------------------------
    while True:
        user_input = input("\n👉 [第一層] 請輸入主 K 值 (預設 6): ").strip()
        if user_input == "":
            K = 6
            break
        elif user_input.isdigit() and int(user_input) > 1:
            K = int(user_input)
            break
        else:
            print("❌ 輸入無效，請輸入大於 1 的整數。")

    print(f"\n🚀 正在執行 K-Means 主分群 (K={K})...")

    kmeans = KMeans(n_clusters=K, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(X_scaled)
    # 先存成整數，稍後會轉成字串
    df['cluster_group'] = cluster_labels

    print("\n📊 主分群結果統計：")
    print(df['cluster_group'].value_counts().sort_index())

    # 找出 Cluster 0 的資料
    # mask_c0 = (df['cluster_group'] == 0)
    # X_c0 = X_scaled[mask_c0]

    # print(f"正在針對 Cluster 0 ({len(X_c0)} 筆) 進行手肘法分析...")

    # inertias = []
    # k_candidates = range(2, 11)  # 測試分 2~10 群

    # for k in k_candidates:
    #     # 這裡只跑 Cluster 0 的資料，速度會很快
    #     temp_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    #     temp_kmeans.fit(X_c0)
    #     inertias.append(temp_kmeans.inertia_)

    # # 畫圖
    # plt.figure(figsize=(8, 4))
    # plt.plot(k_candidates, inertias, 'bo-')
    # plt.title('Elbow Method for Cluster 0 Only')
    # plt.xlabel('Sub-Cluster K')
    # plt.ylabel('Inertia')
    # plt.grid(True)
    # plt.show()

    # ---------------------------------------------------------
    # 4.5 [新增] 針對 Cluster 0 的二次分群 (Sub-clustering)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("🔬 準備針對 Cluster 0 進行二次分群 (Sub-clustering)")
    print("="*50)

    # 找出 Cluster 0 的資料索引
    mask_c0 = (df['cluster_group'] == 0)
    count_c0 = mask_c0.sum()

    if count_c0 > 0:
        print(f"偵測到 Cluster 0 共有 {count_c0} 筆資料。")
        
        while True:
            sub_input = input(f"👉 [第二層] 請輸入 Cluster 0 要拆成幾群 (預設 5): ").strip()
            if sub_input == "":
                sub_K = 5
                break
            elif sub_input.isdigit() and int(sub_input) > 1:
                sub_K = int(sub_input)
                break
            else:
                print("❌ 輸入無效，請輸入大於 1 的整數。")

        print(f"🚀 正在對 Cluster 0 執行二次分群 (sub_K={sub_K})...")
        
        # 取得 Cluster 0 的特徵子集 (使用已經標準化過的數據)
        X_c0 = X_scaled[mask_c0]
        
        # 進行二次分群
        sub_kmeans = KMeans(n_clusters=sub_K, random_state=42, n_init=20)
        sub_labels = sub_kmeans.fit_predict(X_c0)
        
        # 更新 DataFrame 的分群標籤
        # 為了區別，我們將所有標籤轉為字串 (String)
        df['cluster_group'] = df['cluster_group'].astype(str)
        
        # 將 Cluster 0 的標籤改為 "0-0", "0-1", "0-2"... 格式
        # 這樣在後續分析時，我們一看就知道這些人來自原本的 Cluster 0
        new_labels = [f"0-{l}" for l in sub_labels]
        df.loc[mask_c0, 'cluster_group'] = new_labels
        
        print("\n✅ 二次分群完成！")
        print("📊 最終各群組帳戶數量統計：")
        # 依照標籤名稱排序顯示
        print(df['cluster_group'].value_counts().sort_index())
    else:
        print("⚠️ 警告：目前分群結果中沒有 Cluster 0，跳過二次分群。")
    
    print("-" * 50)

    # ---------------------------------------------------------
    # 5. 結果視覺化 (PCA 降維)
    # ---------------------------------------------------------
    print("\n🎨 正在繪製 PCA 分群圖 (含二次分群結果)...")
    
    # 執行 PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # 取得解釋變異量
    explained_variance = pca.explained_variance_ratio_
    pc1_var = explained_variance[0] * 100
    pc2_var = explained_variance[1] * 100

    # -----------------------------------------------------
    # PCA 解密 (略過重複顯示，僅保留畫圖)
    # -----------------------------------------------------

    # 開始畫圖
    plt.figure(figsize=(12, 10)) # 圖稍微加大一點
    
    # 使用更新後的 'cluster_group' 進行著色
    # 為了讓圖例不要亂跳，我們手動做一下排序
    unique_clusters = sorted(df['cluster_group'].unique())
    
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=df['cluster_group'], # 這裡改用 DataFrame 的欄位
        hue_order=unique_clusters, # 指定排序
        palette='viridis', # 或改用 'tab20' 因為群組變多了
        alpha=0.6,
        s=15
    )
    
    plt.title(f'Hierarchical K-Means (Main K={K}, Sub K={sub_K if count_c0>0 else 0})\nTotal Variance: {pc1_var + pc2_var:.1f}%')
    plt.xlabel(f'PC1 (Dim 1) - {pc1_var:.1f}% Variance')
    plt.ylabel(f'PC2 (Dim 2) - {pc2_var:.1f}% Variance')
    
    plt.legend(title='Cluster Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout() # 避免圖例被切掉
    
    print("👀 PCA 圖表已開啟，請查看視窗。")
    print("💡 提示：標籤 '0-x' 代表這是從原本 Cluster 0 拆分出來的子群。")
    plt.show()

    # ---------------------------------------------------------
    # 6. 輸出結果
    # ---------------------------------------------------------
    output_dir = os.path.dirname(file_path) 
    output_file = os.path.join(output_dir, 'account_features_with_cluster.csv')
    
    df.to_csv(output_file, index=False)
    print(f"\n✅ 最終分群結果已儲存為: {output_file}")
    print("你可以接著執行 alert_analysis.py 來查看新分群的警示命中率！")

if __name__ == "__main__":
    main()