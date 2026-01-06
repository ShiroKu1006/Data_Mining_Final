"""
主題1：交易活躍度 - 特徵分布探索

目的：了解 4 個特徵在所有帳戶中的分布情況
不涉及警示/非警示比較，純粹看資料特性
"""
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 讀取特徵資料
df = pd.read_csv('../data/features/account_features_v1.csv')

# 主題1的4個特徵
features = ['txn_cnt', 'active_days', 'txn_cnt_per_day', 'max_txn_per_day']

print("="*80)
print("主題1：交易活躍度特徵分布分析")
print("="*80)
print(f"總帳戶數: {len(df):,}\n")

# 基本統計
print("="*80)
print("特徵基本統計")
print("="*80)
for feat in features:
    print(f"\n【{feat}】")
    s = df[feat]
    print(f"  最小值: {s.min():.2f}")
    print(f"  最大值: {s.max():.2f}")
    print(f"  平均值: {s.mean():.2f}")
    print(f"  中位數: {s.median():.2f}")
    print(f"  標準差: {s.std():.2f}")
    print(f"  25%: {s.quantile(0.25):.2f}")
    print(f"  75%: {s.quantile(0.75):.2f}")
    print(f"  95%: {s.quantile(0.95):.2f}")
    print(f"  99%: {s.quantile(0.99):.2f}")

# 分布分析
print("\n" + "="*80)
print("分布特性分析")
print("="*80)

for feat in features:
    zeros = (df[feat] == 0).sum()
    ones = (df[feat] == 1).sum()
    print(f"\n【{feat}】")
    if zeros > 0:
        print(f"  值為 0 的帳戶: {zeros:,} ({zeros/len(df)*100:.2f}%)")
    if ones > 0:
        print(f"  值為 1 的帳戶: {ones:,} ({ones/len(df)*100:.2f}%)")
    
    # 檢查極端值 (99百分位以上)
    p99 = df[feat].quantile(0.99)
    extreme = (df[feat] > p99).sum()
    print(f"  超過99百分位的帳戶: {extreme:,} ({extreme/len(df)*100:.2f}%)")

# 特徵相關性
print("\n" + "="*80)
print("特徵相關性矩陣")
print("="*80)
corr = df[features].corr()
print(corr.round(3))

# 視覺化
print("\n正在生成分布圖...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('主題1：交易活躍度特徵分布', fontsize=16, y=0.995)

for idx, feat in enumerate(features):
    ax = axes[idx // 2, idx % 2]
    
    # 使用對數尺度繪製（因為可能有長尾）
    data = df[feat]
    
    # 過濾掉0值後繪製
    data_nonzero = data[data > 0]
    
    ax.hist(data_nonzero, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel(feat)
    ax.set_ylabel('帳戶數量')
    ax.set_title(f'{feat} 分布')
    ax.grid(True, alpha=0.3)
    
    # 標註統計值
    mean_val = data.mean()
    median_val = data.median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'平均: {mean_val:.1f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'中位數: {median_val:.1f}')
    ax.legend()

plt.tight_layout()
plt.savefig('topic1_feature_distribution.png', dpi=300, bbox_inches='tight')
print(f"✓ 分布圖已儲存至: topic1_feature_distribution.png")

# 盒鬚圖
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('主題1：特徵盒鬚圖（顯示異常值）', fontsize=14)

for idx, feat in enumerate(features):
    ax = axes[idx]
    # 只顯示非零值
    data_nonzero = df[df[feat] > 0][feat]
    ax.boxplot(data_nonzero, vert=True)
    ax.set_ylabel(feat)
    ax.set_title(feat)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('topic1_boxplot.png', dpi=300, bbox_inches='tight')
print(f"✓ 盒鬚圖已儲存至: topic1_boxplot.png")

print("\n" + "="*80)
print("分析完成！")
print("="*80)
print("\n請查看生成的圖表，然後我們再討論下一步")
