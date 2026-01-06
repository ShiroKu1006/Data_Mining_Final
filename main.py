import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==========================================
# 0. 資料準備與「貼標籤」 (新增部分)
# ==========================================
# 讀取特徵檔案
df = pd.read_csv('data/features/account_features_v1.csv')
print(f"載入主資料完成，總樣本數: {len(df):,}")

# 讀取警示與預測帳戶名單
try:
    df_alert = pd.read_csv('data/初賽資料/acct_alert.csv')
    df_predict = pd.read_csv('data/初賽資料/acct_predict.csv')
    
    # 建立一個集合以便快速查找
    alert_set = set(df_alert['acct'])
    predict_set = set(df_predict['acct'])
    
    # 定義貼標籤函式 (優先級: Alert > Predict > Normal)
    def categorize_account(acct):
        if acct in alert_set:
            return 'Alert'
        elif acct in predict_set:
            return 'Predict'
        else:
            return 'Normal'
            
    # 應用到主資料框 (假設你的主鍵是 from_acct)
    print("正在標記帳戶類型 (Alert/Predict/Normal)...")
    df['account_type'] = df['from_acct'].apply(categorize_account)
    
    print("\n帳戶類型統計：")
    print(df['account_type'].value_counts())
    
except Exception as e:
    print(f"⚠️ 讀取警示/預測檔失敗或欄位對不上，請檢查檔案路徑。錯誤: {e}")
    # 如果讀失敗，就全設為 Normal 以免程式掛掉
    df['account_type'] = 'Normal'

# ==========================================
# 1. 特徵工程 (維持原樣)
# ==========================================
features = [
    # 基本交易規模與活躍度
    'txn_cnt', 'active_days', 'txn_cnt_per_day',
    # 交易金額分佈特徵
    'mean_amt', 'std_amt', 'p95_amt',
    # 交易類型與行為特徵
    'self_txn_ratio', 'cross_bank_ratio', 'night_txn_ratio', 'foreign_currency_ratio',
    # 時間行為特徵
    'min_txn_gap', 'std_txn_gap', 'max_txn_per_day',
    # 交易通路使用比例特徵
    'atm_ratio', 'counter_ratio', 'mobile_bank_ratio', 'web_bank_ratio',
    'voice_ratio', 'eatm_ratio', 'epay_ratio', 'system_txn_ratio', 'unk_channel_ratio',
    # 通路集中度特徵
    'channel_entropy'
]

df_model = df[features].fillna(0)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_model)

# 設定 K 的範圍，例如從 2 到 15
k_range = range(2, 16)
sse = []

for k in k_range:
    # 使用 MiniBatchKMeans 加快大數據的運算速度
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=4096, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    sse.append(kmeans.inertia_) # Inertia 就是 SSE (群內誤差平方和)
    print(f"已計算 K={k}, SSE={kmeans.inertia_:.2f}")

# 繪製手肘圖
plt.figure(figsize=(10, 6))
plt.plot(k_range, sse, marker='o', linestyle='--')
plt.title('Elbow Method Analysis')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('SSE (Inertia)')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# ==========================================
# 2. 執行分群 (使用你決定的 K=7)
# ==========================================
best_k = 9
print(f"\n>>> 使用 K={best_k} 進行 Mini-Batch K-Means 分群...")

final_model = MiniBatchKMeans(
    n_clusters=best_k, 
    batch_size=8192,  # 加大批次，減少震盪 (原本 4096)
    n_init=20,        # 多試幾次起點，確保找到最佳解 (原本 10)
    max_no_improvement=20, # (進階) 如果連續 20 次沒變好就提早停
    random_state=42
)
labels = final_model.fit_predict(df_scaled)
df['cluster'] = labels

# ==========================================

# 3. 視覺化結果 (PCA + Heatmap)

# ==========================================



# (A) PCA 散佈圖

pca = PCA(n_components=2)
components = pca.fit_transform(df_scaled)
plt.figure(figsize=(10, 6))
scatter = plt.scatter(components[:, 0], components[:, 1], c=labels, cmap='tab10', alpha=0.5, s=10)
plt.colorbar(scatter, label='Cluster ID')
plt.title(f'Cluster Visualization (K={best_k})')
plt.show()



# (B) 特徵熱力圖

cluster_summary = pd.DataFrame(df_scaled, columns=features)
cluster_summary['cluster'] = labels
cluster_means = cluster_summary.groupby('cluster').mean()



plt.figure(figsize=(14, 8))
sns.heatmap(cluster_means, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
plt.title(f'Cluster Feature Heatmap (K={best_k})')
plt.xticks(rotation=45, ha='right')
plt.show()



# ==========================================
# 4. 統計分析結果 (供AI分析使用)
# ==========================================
print("\n" + "="*80)
print("Cluster Analysis Statistics")
print("="*80)

# 各群大小
print("\n1. Cluster Size Distribution:")
print("-"*80)
cluster_counts = df['cluster'].value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    pct = count / len(df) * 100
    print(f"Cluster {cluster_id}: {count:,} accounts ({pct:.2f}%)")

# 使用原始數值（未標準化）進行統計
df_original = df[features + ['cluster']].copy()

print("\n2. Cluster Feature Means (Original Scale):")
print("-"*80)
cluster_means_original = df_original.groupby('cluster')[features].mean()
print(cluster_means_original.round(4).to_string())

print("\n3. Cluster Feature Medians (Original Scale):")
print("-"*80)
cluster_medians_original = df_original.groupby('cluster')[features].median()
print(cluster_medians_original.round(4).to_string())

print("\n4. Cluster Feature Std (Original Scale):")
print("-"*80)
cluster_std_original = df_original.groupby('cluster')[features].std()
print(cluster_std_original.round(4).to_string())

# PCA解釋變異數
print("\n5. PCA Explained Variance:")
print("-"*80)
print(f"PC1: {pca.explained_variance_ratio_[0]:.4f} ({pca.explained_variance_ratio_[0]*100:.2f}%)")
print(f"PC2: {pca.explained_variance_ratio_[1]:.4f} ({pca.explained_variance_ratio_[1]*100:.2f}%)")
print(f"Total: {sum(pca.explained_variance_ratio_):.4f} ({sum(pca.explained_variance_ratio_)*100:.2f}%)")

# Standardized特徵均值（Heatmap顯示的數值）
print("\n6. Cluster Feature Means (Standardized - as shown in Heatmap):")
print("-"*80)
print(cluster_means.round(4).to_string())

# ==========================================
# 5. 關鍵分析：警示帳戶都在哪一群？ (新增部分)
# ==========================================
print("\n" + "="*80)
print("🔥🔥🔥 警示帳戶落點分析 (Risk Analysis) 🔥🔥🔥")
print("="*80)

# 1. 製作交叉表 (每個 Cluster 有多少 Alert, Normal, Predict)
cross_tab = pd.crosstab(df['cluster'], df['account_type'])

# 2. 計算「警示帳戶濃度 (Alert Rate)」
# 這代表：在該群裡，有多少比例是警示帳戶？(數值越高越危險)
if 'Alert' in cross_tab.columns:
    cross_tab['Total'] = cross_tab.sum(axis=1)
    cross_tab['Alert_Rate(%)'] = (cross_tab['Alert'] / cross_tab['Total'] * 100).round(2)
    
    # 依照危險程度排序顯示
    risk_report = cross_tab.sort_values('Alert_Rate(%)', ascending=False)
    print("\n依「警示帳戶濃度」排序的風險群聚表：")
    print(risk_report)
    
    # 3. 畫出風險圖
    plt.figure(figsize=(12, 6))
    
    # 雙軸圖：左軸是人數長條圖，右軸是風險率折線圖
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # 依照 Cluster ID 排序方便看
    chart_data = cross_tab.sort_index()
    
    # 畫長條圖 (該群總人數)
    chart_data[['Normal', 'Predict', 'Alert']].plot(kind='bar', stacked=True, ax=ax1, colormap='Pastel1')
    
    # 畫折線圖 (警示帳戶濃度)
    ax2.plot(chart_data.index, chart_data['Alert_Rate(%)'], color='red', marker='o', linewidth=2, label='Alert Rate (%)')
    
    ax1.set_ylabel('Number of Accounts')
    ax2.set_ylabel('Alert Rate (%) (Red Line)', color='red')
    ax1.set_title('Cluster Composition & Risk Level Analysis')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.show()
    
    # 4. 自動判讀結論
    riskiest_cluster = risk_report.index[0]
    highest_rate = risk_report.iloc[0]['Alert_Rate(%)']
    print(f"\n>>> 結論：最危險的群聚是 【Cluster {riskiest_cluster}】")
    print(f"    它的警示帳戶佔比高達 {highest_rate}%。")
    print(f"    請特別檢查這一群的特徵 (參考前面的 Heatmap)，這就是詐欺犯的長相！")
    
    # 5. 檢查 Predict 帳戶
    if 'Predict' in cross_tab.columns:
        predict_in_risk_cluster = cross_tab.loc[riskiest_cluster, 'Predict']
        print(f"\n>>> 預測建議：")
        print(f"    在最危險的 Cluster {riskiest_cluster} 中，發現了 {predict_in_risk_cluster} 個「預測帳戶(Predict)」。")
        print(f"    這些帳戶非常可能也是異常帳戶，建議優先通報或審查！")

else:
    print("資料中沒有發現 'Alert' 標籤，無法計算風險率。")

print("="*80)