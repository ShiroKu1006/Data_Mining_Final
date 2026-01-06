import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import hdbscan

# Colab font settings
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Read data (Colab version: file in same directory)
df_all = pd.read_csv('account_features_v1.csv')
# Theme 2: Fund Scale & Volatility features
features = ['mean_amt', 'std_amt', 'p95_amt']

# ===== Sampling Strategy: 80,000 accounts (consistent results) =====
print("="*80)
print("Sampling Strategy: Ensuring all alert and predict accounts are included")
print("="*80)

# Read alert and predict account lists
try:
    df_alert = pd.read_csv('data/初賽資料/acct_alert.csv')
    alert_accounts = set(df_alert['acct'].unique())
    print(f"✓ Alert accounts loaded: {len(alert_accounts):,}")
except:
    alert_accounts = set()
    print("⚠ acct_alert.csv not found, skipping")

try:
    df_predict = pd.read_csv('data/初賽資料/acct_predict.csv')
    predict_accounts = set(df_predict['acct'].unique())
    print(f"✓ Predict accounts loaded: {len(predict_accounts):,}")
except:
    predict_accounts = set()
    print("⚠ acct_predict.csv not found, skipping")

# Combine must-include accounts
must_include = alert_accounts | predict_accounts
print(f"✓ Total must-include accounts: {len(must_include):,}")

# Separate must-include and other accounts
df_must = df_all[df_all['from_acct'].isin(must_include)].copy()
df_other = df_all[~df_all['from_acct'].isin(must_include)].copy()

# Calculate how many additional samples needed
sample_size = 80000
n_must = len(df_must)
n_additional = sample_size - n_must

print(f"\nSampling configuration:")
print(f"  Target sample size: {sample_size:,}")
print(f"  Must-include accounts: {n_must:,}")
print(f"  Additional random samples needed: {n_additional:,}")
print(f"  Available other accounts: {len(df_other):,}")

# Random sampling with fixed seed (for reproducibility)
if n_additional > 0:
    if n_additional <= len(df_other):
        df_sampled = df_other.sample(n=n_additional, random_state=42)
        df_all = pd.concat([df_must, df_sampled], ignore_index=True)
        print(f"✓ Sampling complete: {len(df_all):,} accounts selected\n")
    else:
        # If not enough other accounts, use all
        df_all = pd.concat([df_must, df_other], ignore_index=True)
        print(f"⚠ Not enough accounts to sample {n_additional:,}, using all {len(df_all):,} accounts\n")
else:
    df_all = df_must
    print(f"✓ Using only must-include accounts: {len(df_all):,}\n")

# Filter: Only keep accounts with mean_amt > 0 (accounts with actual transactions)
threshold_amt = 0
df = df_all[df_all['mean_amt'] > threshold_amt].copy()

print("="*80)
print("Theme 2: HDBSCAN Clustering Analysis - Fund Scale & Volatility (Colab Version)")
print("="*80)
print(f"Total accounts: {len(df_all):,}")
print(f"Filter condition: mean_amt > {threshold_amt}")
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

# 2. HDBSCAN clustering (using larger min_cluster_size for interpretable clusters)
print("Step 2: Execute HDBSCAN Clustering")
print("-"*80)

# Directly use optimal parameter: min_cluster_size=1000 (to get 10-20 clusters)
min_size_chosen = 1000
print(f"Using min_cluster_size={min_size_chosen} (target: 10-20 interpretable clusters)")
print("⚠ Using single core to avoid memory issues (estimated 2-3 minutes)\n")

print("Executing clustering (please wait patiently)...")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=min_size_chosen,
    min_samples=None,
    cluster_selection_method='eom',
    core_dist_n_jobs=1
)
labels = clusterer.fit_predict(X_scaled)
print("✓ Clustering complete\n")

# Add labels to data
df['cluster'] = labels

# 3. Statistics and analysis
print("Step 3: Cluster Statistics and Feature Analysis")
print("-"*80)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f"Number of clusters: {n_clusters}")
print(f"Noise points: {n_noise:,} ({n_noise/len(df)*100:.2f}%)\n")

# Cluster statistics (sorted by size, showing top 20)
cluster_stats = df['cluster'].value_counts().sort_values(ascending=False)
print("Account counts by cluster (top 20 largest):")
for i, (cluster_id, count) in enumerate(cluster_stats.head(20).items()):
    if cluster_id == -1:
        print(f"  Noise points: {count:,} ({count/len(df)*100:.2f}%)")
    else:
        print(f"  Cluster {cluster_id}: {count:,} ({count/len(df)*100:.2f}%)")
if len(cluster_stats) > 20:
    print(f"  ... {len(cluster_stats)-20} other clusters\n")
else:
    print()

# Feature means by cluster
print("Cluster feature means:")
cluster_means = df.groupby('cluster')[features].mean()
print(cluster_means.round(2))

print("\nCluster feature medians:")
cluster_medians = df.groupby('cluster')[features].median()
print(cluster_medians.round(2))

# Visualization
print("\nGenerating clustering visualization...")
# Use original features for interpretable axes
viz_features = ['mean_amt', 'std_amt', 'p95_amt']
X_viz = df[viz_features].values

# Apply sqrt transformation to std_amt to spread out small values
# This helps visualize the dense low-volatility region
X_viz_transformed = X_viz.copy()
X_viz_transformed[:, 1] = np.sqrt(X_viz[:, 1])  # sqrt transform std_amt
print(f"Applied sqrt transformation to std_amt for better visualization")
print(f"  Original std_amt range: {X_viz[:, 1].min():.1f} - {X_viz[:, 1].max():.1f}")
print(f"  Transformed range: {X_viz_transformed[:, 1].min():.1f} - {X_viz_transformed[:, 1].max():.1f}\n")

# Create account type markers for coloring
print("Marking alert and predict accounts for visualization...")
account_type = np.zeros(len(df))  # 0 = normal
df_accounts = df['from_acct'].values
for i, acct in enumerate(df_accounts):
    if acct in alert_accounts:
        account_type[i] = 2  # 2 = alert
    elif acct in predict_accounts:
        account_type[i] = 1  # 1 = predict

n_normal = np.sum(account_type == 0)
n_predict = np.sum(account_type == 1)
n_alert = np.sum(account_type == 2)
print(f"  Normal accounts: {n_normal:,} (gray)")
print(f"  Predict accounts: {n_predict:,} (orange)")
print(f"  Alert accounts: {n_alert:,} (red)\n")

fig = plt.figure(figsize=(24, 12))

# Top row: 2D plots (using first 2 features)
ax = fig.add_subplot(2, 3, 1)
# Plot in layers: normal first, then predict, then alert (so alert is on top)
mask_normal = account_type == 0
mask_predict = account_type == 1
mask_alert = account_type == 2
# First: show all points with cluster colors
scatter = ax.scatter(X_viz_transformed[:, 0], X_viz_transformed[:, 1], c=labels, cmap='tab10', s=2, alpha=0.4)
# Then: mark alert and predict accounts with special markers
ax.scatter(X_viz_transformed[mask_predict, 0], X_viz_transformed[mask_predict, 1], c='orange', marker='o', s=40, alpha=0.9, label='Predict', edgecolors='black', linewidths=0.8)
ax.scatter(X_viz_transformed[mask_alert, 0], X_viz_transformed[mask_alert, 1], c='red', marker='*', s=80, alpha=1.0, label='Alert', edgecolors='darkred', linewidths=0.5)
ax.set_xlabel('Mean Transaction Amount')
ax.set_ylabel('√Std Amount (sqrt transformed)')
ax.set_title(f'2D: HDBSCAN Clusters + Alert/Predict (min_cluster_size={min_size_chosen})')
ax.legend(loc='upper right')
plt.colorbar(scatter, ax=ax, label='Cluster')

ax = fig.add_subplot(2, 3, 2)
is_noise = labels == -1
ax.scatter(X_viz_transformed[~is_noise, 0], X_viz_transformed[~is_noise, 1], c='lightgray', s=2, alpha=0.3, label='In Cluster')
ax.scatter(X_viz_transformed[is_noise & mask_normal, 0], X_viz_transformed[is_noise & mask_normal, 1], c='blue', s=4, alpha=0.5, label='Noise (Normal)')
ax.scatter(X_viz_transformed[is_noise & mask_predict, 0], X_viz_transformed[is_noise & mask_predict, 1], c='orange', s=12, alpha=0.8, label='Noise (Predict)', edgecolors='darkorange', linewidths=0.5)
ax.scatter(X_viz_transformed[is_noise & mask_alert, 0], X_viz_transformed[is_noise & mask_alert, 1], c='red', s=15, alpha=0.9, label='Noise (Alert)', edgecolors='darkred', linewidths=0.5)
ax.set_xlabel('Mean Transaction Amount')
ax.set_ylabel('√Std Amount (sqrt transformed)')
ax.set_title('2D: Noise Point Distribution')
ax.legend(loc='upper right')

ax = fig.add_subplot(2, 3, 3)
outlier_scores = clusterer.outlier_scores_
ax.scatter(X_viz_transformed[mask_normal, 0], X_viz_transformed[mask_normal, 1], c=outlier_scores[mask_normal], cmap='viridis', s=2, alpha=0.4, vmin=0, vmax=1)
ax.scatter(X_viz_transformed[mask_predict, 0], X_viz_transformed[mask_predict, 1], c='orange', s=10, alpha=0.8, label='Predict', edgecolors='darkorange', linewidths=0.5)
scatter = ax.scatter(X_viz_transformed[mask_alert, 0], X_viz_transformed[mask_alert, 1], c='red', s=15, alpha=0.9, label='Alert', edgecolors='darkred', linewidths=0.5)
ax.set_xlabel('Mean Transaction Amount')
ax.set_ylabel('√Std Amount (sqrt transformed)')
ax.set_title('2D: Outlier Score')
ax.legend(loc='upper right')

# Bottom row: 3D plots (using all 3 features)
from mpl_toolkits.mplot3d import Axes3D

ax = fig.add_subplot(2, 3, 4, projection='3d')
# First: show all points with cluster colors
scatter = ax.scatter(X_viz_transformed[:, 0], X_viz_transformed[:, 1], X_viz_transformed[:, 2], c=labels, cmap='tab10', s=1, alpha=0.5)
# Then: mark alert and predict accounts
ax.scatter(X_viz_transformed[mask_predict, 0], X_viz_transformed[mask_predict, 1], X_viz_transformed[mask_predict, 2], c='orange', marker='o', s=40, alpha=0.9, label='Predict', edgecolors='black', linewidths=0.8)
ax.scatter(X_viz_transformed[mask_alert, 0], X_viz_transformed[mask_alert, 1], X_viz_transformed[mask_alert, 2], c='red', marker='*', s=80, alpha=1.0, label='Alert', edgecolors='darkred', linewidths=0.5)
ax.set_xlabel('Mean Amount')
ax.set_ylabel('√Std Amount')
ax.set_zlabel('P95 Amount')
ax.set_title(f'3D: HDBSCAN Clusters + Alert/Predict')
ax.legend(loc='upper right')

ax = fig.add_subplot(2, 3, 5, projection='3d')
ax.scatter(X_viz_transformed[~is_noise, 0], X_viz_transformed[~is_noise, 1], X_viz_transformed[~is_noise, 2], c='lightgray', s=1, alpha=0.3, label='In Cluster')
ax.scatter(X_viz_transformed[is_noise & mask_normal, 0], X_viz_transformed[is_noise & mask_normal, 1], X_viz_transformed[is_noise & mask_normal, 2], c='blue', s=3, alpha=0.5, label='Noise (Normal)')
ax.scatter(X_viz_transformed[is_noise & mask_predict, 0], X_viz_transformed[is_noise & mask_predict, 1], X_viz_transformed[is_noise & mask_predict, 2], c='orange', s=10, alpha=0.8, label='Noise (Predict)', edgecolors='darkorange', linewidths=0.5)
ax.scatter(X_viz_transformed[is_noise & mask_alert, 0], X_viz_transformed[is_noise & mask_alert, 1], X_viz_transformed[is_noise & mask_alert, 2], c='red', s=15, alpha=0.9, label='Noise (Alert)', edgecolors='darkred', linewidths=0.5)
ax.set_xlabel('Mean Amount')
ax.set_ylabel('√Std Amount')
ax.set_zlabel('P95 Amount')
ax.set_title('3D: Noise Point Distribution')
ax.legend(loc='upper right')

ax = fig.add_subplot(2, 3, 6, projection='3d')
ax.scatter(X_viz_transformed[mask_normal, 0], X_viz_transformed[mask_normal, 1], X_viz_transformed[mask_normal, 2], c=outlier_scores[mask_normal], cmap='viridis', s=1, alpha=0.4, vmin=0, vmax=1)
ax.scatter(X_viz_transformed[mask_predict, 0], X_viz_transformed[mask_predict, 1], X_viz_transformed[mask_predict, 2], c='orange', s=10, alpha=0.8, label='Predict', edgecolors='darkorange', linewidths=0.5)
ax.scatter(X_viz_transformed[mask_alert, 0], X_viz_transformed[mask_alert, 1], X_viz_transformed[mask_alert, 2], c='red', s=15, alpha=0.9, label='Alert', edgecolors='darkred', linewidths=0.5)
ax.set_xlabel('Mean Amount')
ax.set_ylabel('√Std Amount')
ax.set_zlabel('P95 Amount')
ax.set_title('3D: Outlier Score')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('hdbscan_clusters_theme2.png', dpi=300, bbox_inches='tight')
print(f"✓ Clustering visualization saved to: hdbscan_clusters_theme2.png")

# Save clustering results (including excluded accounts, marked as cluster = -999)
df_all['cluster_theme2'] = -999  # Accounts with no valid amount data marked as -999
df_all.loc[df.index, 'cluster_theme2'] = df['cluster']

output_file = 'theme2_cluster_results_hdbscan.csv'
# Save all original columns + cluster_theme2
df_all.to_csv(output_file, index=False)
print(f"\n✓ Clustering results saved to: {output_file}")
print(f"  Note: cluster_theme2=-999 indicates accounts with invalid amount data (mean_amt <= {threshold_amt})")
print(f"        cluster_theme2=-1 indicates noise points (extreme fund behavior)")
print(f"        cluster_theme2>=0 indicates normal clusters")

print("\n" + "="*80)
print("Analysis Complete!")
print("="*80)
print("\nClustering Quality Assessment:")
print(f"1. Number of clusters: {len([c for c in cluster_stats.index if c >= 0])}")
print(f"2. Noise ratio: {cluster_stats.get(-1, 0)/len(df)*100:.2f}%")
print(f"3. Largest cluster: {cluster_stats[cluster_stats.index >= 0].max():,} accounts")
print(f"4. Smallest cluster: {cluster_stats[cluster_stats.index >= 0].min():,} accounts")

print("\nTheme 2 Interpretation Guide:")
print("- Low mean_amt + Low std_amt: Small-scale stable accounts")
print("- High mean_amt + Low std_amt: Large-scale stable accounts")
print("- Low mean_amt + High std_amt: Small but volatile (unusual)")
print("- High mean_amt + High std_amt: Large and volatile (high risk)")
print("- High p95_amt: Accounts with occasional large transactions")

