"""
主題1：交易活躍度 - DBSCAN 分群

步驟：
1. 標準化特徵（使用 StandardScaler）
2. 使用 k-distance 圖找合適的 eps
3. 執行 DBSCAN
4. 分析分群結果
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 讀取資料
df = pd.read_csv('../data/features/account_features_v1.csv')
features = ['txn_cnt', 'active_days', 'txn_cnt_per_day', 'max_txn_per_day']

print("="*80)
print("主題1：DBSCAN 分群分析")
print("="*80)
print(f"總帳戶數: {len(df):,}")
print("使用 ball_tree 算法加速（結果與預設相同）\n")

# 1. 標準化
print("步驟1：特徵標準化")
print("-"*80)
X = df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("✓ 已使用 StandardScaler 標準化")
print(f"  標準化後形狀: {X_scaled.shape}")
print(f"  標準化後均值: {X_scaled.mean(axis=0)}")
print(f"  標準化後標準差: {X_scaled.std(axis=0)}\n")

# 2. 使用 k-distance 圖找 eps
print("步驟2：尋找合適的 eps 參數")
print("-"*80)
print("使用 k-distance 圖方法（k=min_samples）")

# 對於大數據集（80萬筆），使用較大的 min_samples
min_samples = 50
print(f"使用 min_samples={min_samples}（適合大數據集）")

# 為加速，k-distance 只用 5% 抽樣計算（足以找到合適的 eps）
sample_size = min(50000, len(X_scaled))  # 最多 5 萬筆
sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
X_sample = X_scaled[sample_indices]

print(f"使用 {len(X_sample):,} 筆樣本計算 k-distance（加速）")

neighbors = NearestNeighbors(n_neighbors=min_samples, algorithm='ball_tree')
neighbors.fit(X_sample)
distances, indices = neighbors.kneighbors(X_sample)

# 取第 k 個鄰居的距離並排序
k_distances = distances[:, -1]
k_distances_sorted = np.sort(k_distances)[::-1]

# 繪製 k-distance 圖
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(len(k_distances_sorted)), k_distances_sorted)
ax.set_xlabel('資料點索引（排序後）')
ax.set_ylabel(f'{min_samples}-th 最近鄰距離')
ax.set_title(f'K-Distance 圖 (k={min_samples})')
ax.grid(True, alpha=0.3)

# 標註一些關鍵百分位
percentiles = [90, 95, 98, 99]
for p in percentiles:
    idx = int(len(k_distances_sorted) * (100-p) / 100)
    dist = k_distances_sorted[idx]
    ax.axhline(y=dist, color='red', linestyle='--', alpha=0.5)
    ax.text(len(k_distances_sorted)*0.7, dist, f'{p}%: {dist:.2f}', fontsize=9)

plt.tight_layout()
plt.savefig('k_distance_plot.png', dpi=300, bbox_inches='tight')
print(f"✓ K-distance 圖已儲存至: k_distance_plot.png")

# 建議的 eps 值（取 95 百分位附近的"肘點"）
suggested_eps = np.percentile(k_distances_sorted, 5)  # 95百分位
print(f"\n建議的 eps 值:")
print(f"  90% 百分位: {np.percentile(k_distances_sorted, 10):.3f}")
print(f"  95% 百分位: {np.percentile(k_distances_sorted, 5):.3f}")
print(f"  98% 百分位: {np.percentile(k_distances_sorted, 2):.3f}")
print(f"  99% 百分位: {np.percentile(k_distances_sorted, 1):.3f}")

# 3. 執行 DBSCAN
print("\n步驟3：執行 DBSCAN（使用全部資料）")
print("-"*80)

# 使用較大的 min_samples 適合大數據集
eps_values = [0.5, 0.7, 1.0]
min_samples_values = [30, 50, 100]  # 調整為適合大數據集的值

print("開始測試不同參數組合...")
results = []

for eps in eps_values:
    for min_samp in min_samples_values:
        print(f"  測試 eps={eps}, min_samples={min_samp}...", end='', flush=True)
        # 使用更快的算法，對全部資料分群
        dbscan = DBSCAN(eps=eps, min_samples=min_samp, algorithm='ball_tree', n_jobs=-1)
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
        
        print(f" 完成！{n_clusters} 個群集, {n_noise:,} 噪音點 ({n_noise/len(labels)*100:.2f}%)")

# 轉換成 DataFrame 方便查看
results_df = pd.DataFrame(results)
print("\n參數組合結果摘要:")
print(results_df.to_string(index=False))

# 4. 選擇一組參數進行詳細分析
print("\n步驟4：詳細分析（使用 eps=0.7, min_samples=50）")
print("-"*80)

eps_chosen = 0.7
min_samples_chosen = 50

print("執行最終分群...")
dbscan = DBSCAN(eps=eps_chosen, min_samples=min_samples_chosen, algorithm='ball_tree', n_jobs=-1)
labels = dbscan.fit_predict(X_scaled)

# 將標籤加入原始資料
df['cluster'] = labels

# 統計各群集
print("\n群集統計:")
cluster_stats = df['cluster'].value_counts().sort_index()
for cluster_id, count in cluster_stats.items():
    if cluster_id == -1:
        print(f"  噪音點: {count:,} ({count/len(df)*100:.2f}%)")
    else:
        print(f"  群集 {cluster_id}: {count:,} ({count/len(df)*100:.2f}%)")

# 分析各群集的特徵平均值
print("\n各群集特徵平均值:")
cluster_means = df.groupby('cluster')[features].mean()
print(cluster_means.round(2))

# 視覺化：前兩個主成分的分群結果
from sklearn.decomposition import PCA

print("\n正在生成分群視覺化...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左圖：分群結果
ax = axes[0]
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', 
                     s=1, alpha=0.5)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title(f'DBSCAN 分群結果 (eps={eps_chosen}, min_samples={min_samples_chosen})')
plt.colorbar(scatter, ax=ax, label='群集')

# 右圖：噪音點 vs 非噪音點
ax = axes[1]
is_noise = labels == -1
ax.scatter(X_pca[~is_noise, 0], X_pca[~is_noise, 1], c='blue', s=1, alpha=0.5, label='群集內')
ax.scatter(X_pca[is_noise, 0], X_pca[is_noise, 1], c='red', s=1, alpha=0.5, label='噪音點')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('噪音點分布')
ax.legend()

plt.tight_layout()
plt.savefig('dbscan_clusters_pca.png', dpi=300, bbox_inches='tight')
print(f"✓ 分群視覺化已儲存至: dbscan_clusters_pca.png")

# 儲存分群結果
output_file = 'theme1_cluster_results.csv'
df[['cid'] + features + ['cluster']].to_csv(output_file, index=False)
print(f"\n✓ 分群結果已儲存至: {output_file}")

print("\n" + "="*80)
print("分析完成！")
print("="*80)
print("\n觀察重點：")
print("1. 噪音點比例是否合理？（太高可能需要調整參數）")
print("2. 群集數量是否有意義？")
print("3. 各群集的特徵平均值是否有明顯差異？")
print("\n如果結果不理想，我們可以：")
print("- 調整 eps 和 min_samples 參數")
print("- 過濾低活躍度帳戶後再分群")
print("- 使用 RobustScaler 處理極端值")
