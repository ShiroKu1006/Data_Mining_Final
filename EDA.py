import pandas as pd
import numpy as np
from scipy import stats

# ==========================================
# 讀取資料
# ==========================================
print("載入資料中...")
df = pd.read_csv('data/features/account_features_v1.csv')
print(f"✓ 資料載入完成，總樣本數: {len(df):,}")
print(f"✓ 總欄位數: {len(df.columns)}\n")

# 自動識別所有數值型特徵（排除帳號欄位）
exclude_cols = ['from_acct']  # 排除非特徵欄位
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
features = [col for col in numeric_cols if col not in exclude_cols]

print(f"偵測到 {len(features)} 個數值特徵:")
for i, feat in enumerate(features, 1):
    print(f"  {i:2d}. {feat}")
print()

# ==========================================
# 執行完整的探索性資料分析
# ==========================================
print("執行探索性資料分析 (EDA)...")
print("="*80)

eda_results = []

for feature in features:
    print(f"分析特徵: {feature}")
    
    # 基本統計量
    data = df[feature]
    
    # 計算各項指標
    result = {
        'feature': feature,
        'count': len(data),
        'missing': data.isna().sum(),
        'missing_pct': data.isna().sum() / len(data) * 100,
        'mean': data.mean(),
        'std': data.std(),
        'min': data.min(),
        'p1': data.quantile(0.01),
        'p5': data.quantile(0.05),
        'p10': data.quantile(0.10),
        'q1': data.quantile(0.25),
        'median': data.median(),
        'q3': data.quantile(0.75),
        'p90': data.quantile(0.90),
        'p95': data.quantile(0.95),
        'p99': data.quantile(0.99),
        'max': data.max(),
        'range': data.max() - data.min(),
        'iqr': data.quantile(0.75) - data.quantile(0.25),
        'skewness': data.skew(),
        'kurtosis': data.kurtosis(),
        'zeros': (data == 0).sum(),
        'zeros_pct': (data == 0).sum() / len(data) * 100,
        'unique_values': data.nunique()
    }
    
    eda_results.append(result)

# 轉換成DataFrame
eda_df = pd.DataFrame(eda_results)

# ==========================================
# 輸出結果
# ==========================================
output_file = 'data/features/eda_statistics.csv'
eda_df.to_csv(output_file, index=False)
print("\n" + "="*80)
print(f"✓ EDA結果已儲存至: {output_file}")
print("="*80)

# 同時在console印出結果供快速查看
print("\n完整統計摘要 (前幾個特徵):")
print("-"*80)
# 轉置以便閱讀
print(eda_df.set_index('feature').T.to_string())

# ==========================================
# 額外分析：識別潛在問題
# ==========================================
print("\n" + "="*80)
print("資料品質檢查")
print("="*80)

# 1. 缺失值檢查
print("\n1. 缺失值情況:")
missing_features = eda_df[eda_df['missing'] > 0][['feature', 'missing', 'missing_pct']]
if len(missing_features) > 0:
    print(missing_features.to_string(index=False))
else:
    print("✓ 無缺失值")

# 2. 零值檢查
print("\n2. 零值比例超過50%的特徵:")
high_zeros = eda_df[eda_df['zeros_pct'] > 50][['feature', 'zeros', 'zeros_pct']]
if len(high_zeros) > 0:
    for _, row in high_zeros.iterrows():
        print(f"  {row['feature']}: {row['zeros']:,} zeros ({row['zeros_pct']:.2f}%)")
else:
    print("✓ 無特徵零值比例超過50%")

# 3. 偏態檢查
print("\n3. 高度偏態特徵 (|skewness| > 2):")
high_skew = eda_df[eda_df['skewness'].abs() > 2][['feature', 'skewness']]
if len(high_skew) > 0:
    for _, row in high_skew.iterrows():
        direction = "右偏" if row['skewness'] > 0 else "左偏"
        print(f"  {row['feature']}: {row['skewness']:.2f} ({direction})")
else:
    print("✓ 無高度偏態特徵")

# 4. 異常值比例估計 (使用IQR方法)
print("\n4. 潛在異常值比例 (IQR方法):")
for feature in features:
    data = df[feature].dropna()
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = ((data < lower_bound) | (data > upper_bound)).sum()
    outlier_pct = outliers / len(data) * 100
    if outlier_pct > 5:
        print(f"  {feature}: {outliers:,} outliers ({outlier_pct:.2f}%)")

# 5. 特徵變異性檢查
print("\n5. 低變異性特徵 (std/mean < 0.1，排除比例特徵):")
low_var_features = []
for _, row in eda_df.iterrows():
    if row['mean'] != 0 and 'ratio' not in row['feature']:
        cv = row['std'] / abs(row['mean'])  # Coefficient of Variation
        if cv < 0.1:
            low_var_features.append((row['feature'], cv))

if low_var_features:
    for feature, cv in low_var_features:
        print(f"  {feature}: CV={cv:.4f}")
else:
    print("✓ 所有特徵皆有適當變異性")

print("\n" + "="*80)
print("EDA分析完成！")
print("="*80)
