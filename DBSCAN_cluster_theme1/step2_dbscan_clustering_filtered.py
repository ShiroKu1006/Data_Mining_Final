"""
主題1：交易活躍度 - DBSCAN 分群（過濾版）

改進：只對交易數 >= 3 的帳戶分群
原因：57.58% 帳戶只有 1 筆交易，特徵幾乎相同，無法有效分群
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 讀取資料
df_all = pd.read_csv('../data/features/account_features_v1.csv')
features = ['txn_cnt', 'active_days', 'txn_cnt_per_day', 'max_txn_per_day']

# 過濾：只保留交易數 >= 3 的帳戶
threshold = 3
df = df_all[df_all['txn_cnt'] >= threshold].copy()

print("="*80)
print("主題1：DBSCAN 分群分析（過濾版）")
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

# 2. 快速測試幾組參數（不用 k-distance 了）
print("步驟2：測試不同參數組合")
print("-"*80)
print("提示：對於 35 萬筆資料，建議 eps=0.5-1.0, min_samples=30-100\n")

# 使用單核避免記憶體問題
eps_values = [0.5, 0.7, 1.0]
min_samples_values = [30, 50, 100]

results = []

for eps in eps_values:
    for min_samp in min_samples_values:
        print(f"  測試 eps={eps}, min_samples={min_samp}...", end='', flush=True)
        # 使用 brute force 算法（較慢但省記憶體）
        dbscan = DBSCAN(eps=eps, min_samples=min_samp, algorithm='brute')
        labels = dbscan.fit_predict(X_scaled)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        results.append({
            'eps': eps,
            'min_samples': min_samp,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_pct': n_noise / len(labels) * 100
        })
        
        print(f" {n_clusters} 群集, 噪音 {n_noise/len(labels)*100:.1f}%")

# 結果摘要
results_df = pd.DataFrame(results)
print("\n參數組合結果摘要:")
print(results_df.to_string(index=False))

# 3. 選擇噪音點比例合理的參數（約 10-20%）
print("\n步驟3：選擇最佳參數並詳細分析")
print("-"*80)

# 自動選擇噪音點 10-20% 之間的組合
reasonable = results_df[(results_df['noise_pct'] >= 5) & (results_df['noise_pct'] <= 25)]
if len(reasonable) > 0:
    # 選擇群集數最多的（分群最細緻）
    best = reasonable.loc[reasonable['n_clusters'].idxmax()]
    eps_chosen = best['eps']
    min_samples_chosen = int(best['min_samples'])
    print(f"自動選擇: eps={eps_chosen}, min_samples={min_samples_chosen}")
    print(f"  理由: 噪音比例 {best['noise_pct']:.1f}%（合理範圍），{int(best['n_clusters'])} 個群集")
else:
    # 備選：使用中間值
    eps_chosen = 0.7
    min_samples_chosen = 50
    print(f"使用預設: eps={eps_chosen}, min_samples={min_samples_chosen}")

print("\n執行最終分群...")
dbscan = DBSCAN(eps=eps_chosen, min_samples=min_samples_chosen, algorithm='brute')
labels = dbscan.fit_predict(X_scaled)

# 將標籤加入資料
df['cluster'] = labels

# 統計各群集
print("\n群集統計:")
cluster_stats = df['cluster'].value_counts().sort_index()
for cluster_id, count in cluster_stats.items():
    if cluster_id == -1:
        print(f"  噪音點: {count:,} ({count/len(df)*100:.2f}%)")
    else:
        print(f"  群集 {cluster_id}: {count:,} ({count/len(df)*100:.2f}%)")

# 各群集特徵平均值
print("\n各群集特徵平均值:")
cluster_means = df.groupby('cluster')[features].mean()
print(cluster_means.round(2))

print("\n各群集特徵中位數:")
cluster_medians = df.groupby('cluster')[features].median()
print(cluster_medians.round(2))

# 視覺化
print("\n正在生成分群視覺化...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左圖：分群結果
ax = axes[0]
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', 
                     s=2, alpha=0.5)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title(f'DBSCAN 分群結果 (eps={eps_chosen}, min_samples={min_samples_chosen})')
plt.colorbar(scatter, ax=ax, label='群集')

# 右圖：噪音點 vs 非噪音點
ax = axes[1]
is_noise = labels == -1
ax.scatter(X_pca[~is_noise, 0], X_pca[~is_noise, 1], c='blue', s=2, alpha=0.5, label='群集內')
ax.scatter(X_pca[is_noise, 0], X_pca[is_noise, 1], c='red', s=2, alpha=0.5, label='噪音點')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('噪音點分布')
ax.legend()

plt.tight_layout()
plt.savefig('dbscan_clusters_filtered.png', dpi=300, bbox_inches='tight')
print(f"✓ 分群視覺化已儲存至: dbscan_clusters_filtered.png")

# 儲存分群結果（包含被排除的帳戶，標記為 cluster = -999）
df_all['cluster'] = -999  # 低活躍度帳戶標記為 -999
df_all.loc[df.index, 'cluster'] = df['cluster']

output_file = 'theme1_cluster_results_filtered.csv'
df_all[['cid'] + features + ['cluster']].to_csv(output_file, index=False)
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
