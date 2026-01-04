"""
主題1：交易活躍度 - HDBSCAN 分群（簡化版，可解釋）

目標：得到 10-20 個可解釋的群集

Colab 環境設置（先執行）：
!pip install hdbscan -q
!apt-get install -y fonts-noto-cjk > /dev/null 2>&1
from google.colab import files
uploaded = files.upload()  # 上傳 account_features_v1.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import hdbscan

# Colab 中文字體設定
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

# 讀取資料（Colab 版本：檔案在同目錄）
df_all = pd.read_csv('account_features_v1.csv')
features = ['txn_cnt', 'active_days', 'txn_cnt_per_day', 'max_txn_per_day']

# 過濾：只保留交易數 >= 3 的帳戶
threshold = 3
df = df_all[df_all['txn_cnt'] >= threshold].copy()

print("="*80)
print("主題1：HDBSCAN 分群分析（Colab 版本）")
print("="*80)
print(f"總帳戶數: {len(df_all):,}")
print(f"過濾條件: txn_cnt >= {threshold}")
print(f"保留帳戶數: {len(df):,} ({len(df)/len(df_all)*100:.2f}%)")
print(f"排除帳戶數: {len(df_all)-len(df):,} ({(len(df_all)-len(df))/len(df_all)*100:.2f}%)\n")

# 1. 標準化
print("步驟1：特徵標準化")
print("-"*80)
X = df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("✓ 已使用 StandardScaler 標準化")
print(f"  標準化後形狀: {X_scaled.shape}\n")

# 2. HDBSCAN 分群（使用較大的 min_cluster_size 得到可解釋的群集數）
print("步驟2：執行 HDBSCAN 分群")
print("-"*80)

# 直接使用最佳參數：min_cluster_size=2000（得到 10-20 個群集）
min_size_chosen = 2000
print(f"使用 min_cluster_size={min_size_chosen}（目標：10-20 個可解釋的群集）")
print("⚠ 使用單核心處理以避免記憶體不足（預計 2-3 分鐘）\n")

print("執行分群（請耐心等待）...")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=min_size_chosen,
    min_samples=None,
    cluster_selection_method='eom',
    core_dist_n_jobs=-1
)
labels = clusterer.fit_predict(X_scaled)
print("✓ 分群完成\n")

# 將標籤加入資料
df['cluster'] = labels

# 3. 統計與分析
print("步驟3：群集統計與特徵分析")
print("-"*80)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f"群集數量: {n_clusters}")
print(f"噪音點: {n_noise:,} ({n_noise/len(df)*100:.2f}%)\n")
# 各群集統計（按大小排序，顯示前 20 個）
cluster_stats = df['cluster'].value_counts().sort_values(ascending=False)
print("各群集帳戶數（前 20 個最大的）:")
for i, (cluster_id, count) in enumerate(cluster_stats.head(20).items()):
    if cluster_id == -1:
        print(f"  噪音點: {count:,} ({count/len(df)*100:.2f}%)")
    else:
        print(f"  群集 {cluster_id}: {count:,} ({count/len(df)*100:.2f}%)")
if len(cluster_stats) > 20:
    print(f"  ... 其他 {len(cluster_stats)-20} 個群集\n")
else:
    print()

# 各群集特徵平均值
print("各群集特徵平均值:")
cluster_means = df.groupby('cluster')[features].mean()
print(cluster_means.round(2))

print("\n各群集特徵中位數:")
cluster_medians = df.groupby('cluster')[features].median()
print(cluster_medians.round(2))

# 視覺化
print("\n正在生成分群視覺化...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# 左圖：分群結果
ax = axes[0]
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', 
                     s=2, alpha=0.5)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title(f'HDBSCAN 分群結果 (min_cluster_size={min_size_chosen})')
plt.colorbar(scatter, ax=ax, label='群集')

# 中圖：噪音點 vs 非噪音點
ax = axes[1]
is_noise = labels == -1
ax.scatter(X_pca[~is_noise, 0], X_pca[~is_noise, 1], c='blue', s=2, alpha=0.5, label='群集內')
ax.scatter(X_pca[is_noise, 0], X_pca[is_noise, 1], c='red', s=2, alpha=0.5, label='噪音點')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('噪音點分布')
ax.legend()

# 右圖：群集穩定性（顏色深淺代表 outlier score）
ax = axes[2]
outlier_scores = clusterer.outlier_scores_
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=outlier_scores, cmap='viridis', 
                     s=2, alpha=0.5)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('Outlier Score（越高越異常）')
plt.colorbar(scatter, ax=ax, label='Score')

plt.tight_layout()
plt.savefig('hdbscan_clusters_filtered.png', dpi=300, bbox_inches='tight')
print(f"✓ 分群視覺化已儲存至: hdbscan_clusters_filtered.png")

# 儲存分群結果（包含被排除的帳戶，標記為 cluster = -999）
df_all['cluster'] = -999  # 低活躍度帳戶標記為 -999
df_all.loc[df.index, 'cluster'] = df['cluster']

output_file = 'theme1_cluster_results_hdbscan.csv'
# 保存所有原始欄位 + cluster
df_all.to_csv(output_file, index=False)
print(f"\n✓ 分群結果已儲存至: {output_file}")
print(f"  說明: cluster=-999 表示低活躍度帳戶（txn_cnt < {threshold}）")
print(f"        cluster=-1 表示噪音點（高活躍但異常）")
print(f"        cluster>=0 表示正常群集")

print("\n" + "="*80)
print("分析完成！")
print("="*80)
print("\n分群品質評估：")
print(f"1. 群集數量: {len([c for c in cluster_stats.index if c >= 0])}")
print(f"2. 噪音比例: {cluster_stats.get(-1, 0)/len(df)*100:.2f}%")
print(f"3. 最大群集: {cluster_stats[cluster_stats.index >= 0].max():,} 個帳戶")
print(f"4. 最小群集: {cluster_stats[cluster_stats.index >= 0].min():,} 個帳戶")

print("\nHDBSCAN 優勢：")
print("✓ 自動選擇最佳密度層次（不需調 eps）")
print("✓ 可找到不同密度的群集")
print("✓ 提供 outlier score 評估異常程度")
print("✓ 更穩定、更適合真實世界資料")

# ===== 下載結果（最後執行） =====
# from google.colab import files
# files.download('theme1_cluster_results_hdbscan.csv')
# files.download('hdbscan_clusters_filtered.png')
