import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# ---------------------------------------------------------
# 設定：解決圖表中文字型可能出現的亂碼問題
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
# ---------------------------------------------------------

def main():
    print("🚀 開始執行警示與預測帳戶分析模組...")

    # ==============================================================================
    # 1. 資料準備 (讀取分群、警示名單、待預測名單)
    # ==============================================================================
    
    # --- 設定路徑 ---
    cluster_result_path = os.path.join('data', 'features', 'account_features_with_cluster.csv')
    raw_data_dir = os.path.join('data') 
    alert_list_path = os.path.join(raw_data_dir, 'acct_alert.csv')
    predict_list_path = os.path.join(raw_data_dir, 'acct_predict.csv')

    # --- A. 讀取分群結果 ---
    if not os.path.exists(cluster_result_path):
        print(f"❌ 錯誤：找不到分群結果檔案 {cluster_result_path}")
        return
    
    df = pd.read_csv(cluster_result_path)
    print(f"✅ 成功讀取分群結果，共 {df.shape[0]} 筆帳戶。")

    # --- B. 讀取警示帳戶名單 ---
    alert_accts = set()
    # 嘗試讀取 (包含容錯路徑)
    possible_alert_paths = [alert_list_path, os.path.join('data', '初賽資料', 'acct_alert.csv')]
    for path in possible_alert_paths:
        if os.path.exists(path):
            df_alert = pd.read_csv(path)
            if 'acct' in df_alert.columns: df_alert = df_alert.rename(columns={'acct': 'from_acct'})
            if 'from_acct' in df_alert.columns:
                alert_accts = set(df_alert['from_acct'].unique())
                print(f"✅ 成功讀取警示帳戶 ({path})，共 {len(alert_accts)} 筆。")
                break
    
    # --- C. 讀取待預測帳戶名單 ---
    pred_accts = set()
    possible_pred_paths = [predict_list_path, os.path.join('data', '初賽資料', 'acct_predict.csv')]
    for path in possible_pred_paths:
        if os.path.exists(path):
            df_pred = pd.read_csv(path)
            if 'acct' in df_pred.columns: df_pred = df_pred.rename(columns={'acct': 'from_acct'})
            if 'from_acct' in df_pred.columns:
                pred_accts = set(df_pred['from_acct'].unique())
                print(f"✅ 成功讀取待預測帳戶 ({path})，共 {len(pred_accts)} 筆。")
                break

    # ==============================================================================
    # 2. 資料合併與標記
    # ==============================================================================
    print("\n🔄 正在進行帳戶分類標記...")

    def get_account_type(acct):
        if acct in alert_accts:
            return 'Alert'   # 紅色：已知警示
        elif acct in pred_accts:
            return 'Predict' # 橘色：待預測
        else:
            return 'Normal'  # 藍色：一般

    if 'from_acct' in df.columns:
        df['acct_type'] = df['from_acct'].apply(get_account_type)
    else:
        print("❌ 資料中缺少 'from_acct' 欄位，無法標記。")
        return

    print("📊 整體帳戶類型統計：")
    print(df['acct_type'].value_counts())

    # ==============================================================================
    # 2.5 計算並輸出各群統計
    # ==============================================================================
    print("\n" + "="*60)
    print("📊 各群組【警示帳戶】與【待預測帳戶】分佈統計")
    print("="*60)

    clusters = sorted(df['cluster_group'].unique())
    for k in clusters:
        subset = df[df['cluster_group'] == k]
        total = len(subset)
        n_alert = len(subset[subset['acct_type'] == 'Alert'])
        p_alert = (n_alert / total * 100) if total > 0 else 0
        n_pred = len(subset[subset['acct_type'] == 'Predict'])
        p_pred = (n_pred / total * 100) if total > 0 else 0
        
        risk_tag = "🔴 高風險群!" if p_alert > 5.0 else ""
        print(f"Cluster {k} (總數: {total}) {risk_tag}")
        print(f"   ❌ 警示帳戶: {n_alert} 筆 ({p_alert:.2f}%)")
        print(f"   📐 待預測戶: {n_pred} 筆 ({p_pred:.2f}%)")
        print("-" * 30)

    # ==============================================================================
    # 3. 重建 PCA 座標
    # ==============================================================================
    print("\n🔄 正在重建 PCA 座標以進行繪圖...")
    selected_features = [
        'mobile_bank_ratio', 'channel_entropy', 'txn_cnt', 
        'total_amt', 'std_amt', 'max_amt', 
        'active_days', 'night_cnt', 
        'min_txn_gap', 'std_txn_gap', 'max_txn_per_day'
    ]
    X = df[selected_features].copy().fillna(0)
    
    log_cols = ['total_amt', 'max_amt', 'std_amt', 'min_txn_gap', 'std_txn_gap']
    for col in log_cols:
        if col in X.columns:
            X.loc[X[col] < 0, col] = 0
            X[col] = np.log1p(X[col])
            
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    total_var = sum(pca.explained_variance_ratio_) * 100

    # 準備繪圖資料
    plot_data = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Cluster': df['cluster_group'],
        'Type': df['acct_type']
    })

    # ==============================================================================
    # 4. 視覺化繪圖 - 第一階段：僅顯示「警示帳戶」(預測戶暫時視為背景)
    # ==============================================================================
    print("\n🎨 [1/2] 正在繪製第一階段圖表 (僅標記警示帳戶)...")
    
    plt.figure(figsize=(14, 11))

    # Layer 1: 背景 (包含 Normal 和 Predict，都先當作 Cluster 背景顯示)
    # 我們排除 'Alert' 類型，剩下的就是背景
    background_data = plot_data[plot_data['Type'] != 'Alert']
    sns.scatterplot(
        data=background_data,
        x='PCA1', y='PCA2',
        hue='Cluster',
        palette='viridis',
        alpha=0.15,
        s=15,
        legend='full'
    )

    # Layer 2: 警示 (Alert) - 紅色叉叉
    alert_data = plot_data[plot_data['Type'] == 'Alert']
    if not alert_data.empty:
        plt.scatter(
            alert_data['PCA1'],
            alert_data['PCA2'],
            color='red',
            s=60,
            marker='X',
            label='Alert Account',
            edgecolor='white',
            linewidth=0.5,
            zorder=5
        )
    
    plt.title(f'Cluster Analysis (Phase 1): Known Alerts Only (Var: {total_var:.1f}%)', fontsize=16)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Legend', loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    print("👀 第一張圖 (警示資料) 已顯示，請關閉視窗以繼續...")
    plt.show()

    # ==============================================================================
    # 5. 視覺化繪圖 - 第二階段：顯示「待預測資料」(且在警示上方)
    # ==============================================================================
    print("\n🎨 [2/2] 正在繪製第二階段圖表 (加入待預測資料)...")
    
    plt.figure(figsize=(14, 11))

    # Layer 1: 背景 (只剩 Normal)
    normal_data = plot_data[plot_data['Type'] == 'Normal']
    sns.scatterplot(
        data=normal_data,
        x='PCA1', y='PCA2',
        hue='Cluster',
        palette='viridis',
        alpha=0.15,
        s=15,
        legend='full'
    )

    # Layer 2: 警示 (Alert) - 紅色叉叉
    # 注意：這裡將 zorder 設為 4 (比待預測的 5 低)，讓它在下面
    alert_data = plot_data[plot_data['Type'] == 'Alert']
    if not alert_data.empty:
        plt.scatter(
            alert_data['PCA1'],
            alert_data['PCA2'],
            color='red',
            s=60,
            marker='X',
            label='Alert Account',
            edgecolor='white',
            linewidth=0.5,
            zorder=4  # <--- 在待預測資料下面
        )

    # Layer 3: 待預測 (Predict) - 橘色三角形
    # 注意：這裡將 zorder 設為 5 (最高)，讓它在最上面
    pred_data = plot_data[plot_data['Type'] == 'Predict']
    if not pred_data.empty:
        plt.scatter(
            pred_data['PCA1'],
            pred_data['PCA2'],
            color='orange',
            s=50,             # 稍微大一點
            marker='^',       # 三角形
            label='Prediction List',
            edgecolor='black',
            linewidth=0.5,
            alpha=0.9,
            zorder=5          # <--- 在最上面!
        )

    plt.title(f'Cluster Analysis (Phase 2): Alerts + Predictions (Var: {total_var:.1f}%)', fontsize=16)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    
    plt.legend(title='Legend', loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 儲存最終圖片
    output_dir = os.path.dirname(cluster_result_path)
    output_img = os.path.join(output_dir, 'pca_cluster_prediction_map_final.png')
    plt.savefig(output_img, dpi=300)
    print(f"✅ 最終圖片已儲存至: {output_img}")
    
    print("👀 第二張圖 (含預測資料) 已顯示。")
    plt.show()

if __name__ == "__main__":
    main()