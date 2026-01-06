"""
Theme 1: Transaction Activity - DBSCAN Clustering (Colab Version)

Instructions:
1. Upload account_features_v1.csv to Colab
2. Install fonts (execute first cell)
3. Run this script

Improvement: Only cluster accounts with txn_cnt >= 3
"""

# ===== Colab Environment Setup (Execute this first) =====
# !apt-get install -y fonts-noto-cjk > /dev/null 2>&1
# from google.colab import files
# uploaded = files.upload()  # Upload account_features_v1.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# Colab font settings
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Read data (Colab version: file in same directory)
df_all = pd.read_csv('account_features_v1.csv')
features = ['txn_cnt', 'active_days', 'txn_cnt_per_day', 'max_txn_per_day']

# Filter: Only keep accounts with txn_cnt >= 3
threshold = 3
df = df_all[df_all['txn_cnt'] >= threshold].copy()

print("="*80)
print("Theme 1: DBSCAN Clustering Analysis (Colab Version)")
print("="*80)
print(f"Total accounts: {len(df_all):,}")
print(f"Filter condition: txn_cnt >= {threshold}")
print(f"Retained accounts: {len(df):,} ({len(df)/len(df_all)*100:.2f}%)")
print(f"Excluded accounts: {len(df_all)-len(df):,} ({(len(df_all)-len(df))/len(df_all)*100:.2f}%)\n")

# 1. Standardization
print("Step 1: Feature Standardization")
print("-"*80)
X = df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("✓ Standardized using StandardScaler")
print(f"  Standardized shape: {X_scaled.shape}\n")

# 2. Quick test of parameter combinations
print("Step 2: Testing Different Parameter Combinations")
print("-"*80)
print("Using ball_tree algorithm (Colab has sufficient memory)\n")

# Colab can use ball_tree and multi-core
eps_values = [0.5, 0.7, 1.0]
min_samples_values = [30, 50, 100]

results = []

for eps in eps_values:
    for min_samp in min_samples_values:
        print(f"  Testing eps={eps}, min_samples={min_samp}...", end='', flush=True)
        # Colab version: using ball_tree + multi-core
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
        
        print(f" {n_clusters} clusters, noise {n_noise/len(labels)*100:.1f}%")

# Results summary
results_df = pd.DataFrame(results)
print("\nParameter Combination Results Summary:")
print(results_df.to_string(index=False))

# 3. Select parameters with reasonable noise ratio (approx. 10-20%)
print("\nStep 3: Select Best Parameters and Detailed Analysis")
print("-"*80)

# Automatically select combinations with noise between 5-25%
reasonable = results_df[(results_df['noise_pct'] >= 5) & (results_df['noise_pct'] <= 25)]
if len(reasonable) > 0:
    # Select the one with most clusters (finest clustering)
    best = reasonable.loc[reasonable['n_clusters'].idxmax()]
    eps_chosen = best['eps']
    min_samples_chosen = int(best['min_samples'])
    print(f"Auto-selected: eps={eps_chosen}, min_samples={min_samples_chosen}")
    print(f"  Reason: Noise ratio {best['noise_pct']:.1f}% (reasonable range), {int(best['n_clusters'])} clusters")
else:
    # Fallback: use middle values
    eps_chosen = 0.7
    min_samples_chosen = 50
    print(f"Using default: eps={eps_chosen}, min_samples={min_samples_chosen}")

print("\nExecuting final clustering...")
dbscan = DBSCAN(eps=eps_chosen, min_samples=min_samples_chosen, algorithm='ball_tree', n_jobs=-1)
labels = dbscan.fit_predict(X_scaled)

# Add labels to data
df['cluster'] = labels

# Cluster statistics
print("\nCluster Statistics:")
cluster_stats = df['cluster'].value_counts().sort_index()
for cluster_id, count in cluster_stats.items():
    if cluster_id == -1:
        print(f"  Noise points: {count:,} ({count/len(df)*100:.2f}%)")
    else:
        print(f"  Cluster {cluster_id}: {count:,} ({count/len(df)*100:.2f}%)")

# Feature means by cluster
print("\nCluster Feature Means:")
cluster_means = df.groupby('cluster')[features].mean()
print(cluster_means.round(2))

print("\nCluster Feature Medians:")
cluster_medians = df.groupby('cluster')[features].median()
print(cluster_medians.round(2))

# Visualization
print("\nGenerating clustering visualization...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left plot: Clustering results
ax = axes[0]
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', 
                     s=2, alpha=0.5)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title(f'DBSCAN Clustering Results (eps={eps_chosen}, min_samples={min_samples_chosen})')
plt.colorbar(scatter, ax=ax, label='Cluster')

# Right plot: Noise vs Non-noise points
ax = axes[1]
is_noise = labels == -1
ax.scatter(X_pca[~is_noise, 0], X_pca[~is_noise, 1], c='blue', s=2, alpha=0.5, label='In Cluster')
ax.scatter(X_pca[is_noise, 0], X_pca[is_noise, 1], c='red', s=2, alpha=0.5, label='Noise Points')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('Noise Point Distribution')
ax.legend()

plt.tight_layout()
plt.savefig('dbscan_clusters_filtered.png', dpi=300, bbox_inches='tight')
print(f"✓ Clustering visualization saved to: dbscan_clusters_filtered.png")

# Save clustering results (including excluded accounts, marked as cluster = -999)
df_all['cluster'] = -999  # Low activity accounts marked as -999
df_all.loc[df.index, 'cluster'] = df['cluster']

output_file = 'theme1_cluster_results_filtered.csv'
df_all[['cid'] + features + ['cluster']].to_csv(output_file, index=False)
print(f"\n✓ Clustering results saved to: {output_file}")
print(f"  Note: cluster=-999 indicates low activity accounts (txn_cnt < {threshold})")
print(f"        cluster=-1 indicates noise points (high activity but anomalous)")
print(f"        cluster>=0 indicates normal clusters")

print("\n" + "="*80)
print("Analysis Complete!")
print("="*80)
print("\nClustering Quality Assessment:")
print(f"1. Number of clusters: {len([c for c in cluster_stats.index if c >= 0])}")
print(f"2. Noise ratio: {cluster_stats.get(-1, 0)/len(df)*100:.2f}%")
print(f"3. Largest cluster: {cluster_stats[cluster_stats.index >= 0].max():,} accounts")
print(f"4. Smallest cluster: {cluster_stats[cluster_stats.index >= 0].min():,} accounts")

# ===== Download results (execute last) =====
# from google.colab import files
# files.download('theme1_cluster_results_filtered.csv')
# files.download('dbscan_clusters_filtered.png')
