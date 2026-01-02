# `clean_data.py`
## acct_transaction.csv 清洗規則
### 1) txn_time 保留原樣
- 原始 txn_time 格式為 HH:MM:SS（例：05:05:00）
- 清洗後仍保留 txn_time（字串），不會因轉型而變成空值
### 2) 新增可計算欄位
- txn_time_sec：將 HH:MM:SS 轉為一天內的秒數
  - 例：05:05:00 → 18300
- txn_ts：以「第幾天」與「秒數」合成排序用時間戳
  - txn_ts = txn_date * 86400 + txn_time_sec

用途：
- 便於排序、計算相鄰交易時間差、建立時間行為特徵

### 3) 類別欄位正規化
- 去前後空白
- 統一大寫（適用欄位：is_self_txn, currency_type, channel_type, from_acct_type, to_acct_type）
- 空值或空字串 → UNK
- is_self_txn 只允許 Y/N/UNK，其餘值 → UNK

### 4) 數值欄位轉型
- txn_amt → float（不可解析 → NaN）
- txn_date → Int64（不可解析 → NaN）

### 5) 去重
- 移除完全重複列（drop_duplicates）

## acct_alert.csv 清洗規則
- acct：去空白
- event_date：轉 Int64
- 新增 label=1
- 以 (acct, event_date) 去重（若欄位存在）
---
# 特徵說明
## 1. 基本交易規模與活躍度特徵

| Feature | 說明 |
|------|------|
| txn_cnt | 帳戶總交易筆數 |
| active_days | 有交易發生的不同天數 |
| txn_cnt_per_day | 平均每日交易筆數 |

### 行為解釋
此類特徵反映帳戶的**整體活躍程度與使用頻率**。

- 一般正常帳戶通常呈現穩定且分散的交易分佈  
- 警示帳戶可能出現：
  - 短時間內大量交易（高 txn_cnt、低 active_days）
  - 異常密集的每日交易行為（高 txn_cnt_per_day）

## 2. 金額統計特徵（Transaction Amount Statistics）

| Feature | 說明 |
|------|------|
| total_amt | 交易總金額 |
| mean_amt | 平均交易金額 |
| max_amt | 單筆最大交易金額 |
| std_amt | 交易金額標準差 |
| p95_amt | 交易金額第 95 百分位數 |

### 行為解釋
此類特徵用於描述帳戶的**資金規模與金額波動情形**。

- 正常帳戶通常金額分佈集中、波動較小  
- 警示帳戶常見特徵包含：
  - 極端大額交易（高 max_amt / p95_amt）
  - 金額不穩定（高 std_amt），可能代表異常資金流動

---

## 3. 交易關係與性質特徵

| Feature | 說明 |
|------|------|
| self_txn_cnt | 自轉帳交易筆數 |
| cross_bank_cnt | 跨行交易筆數 |
| foreign_cnt | 外幣交易筆數 |
| self_txn_ratio | 自轉帳比例 |
| cross_bank_ratio | 跨行交易比例 |
| foreign_currency_ratio | 外幣交易比例 |

### 行為解釋
此類特徵描述帳戶的**資金流向與交易關係結構**。

- 自轉帳比例過高，可能代表資金在帳戶間反覆移動
- 跨行交易頻繁，可能與資金分流或隱匿行為相關
- 外幣交易比例異常，通常非一般個人帳戶常態行為

這些特徵在風控領域中常被視為**高辨識度行為指標**。

---

## 4. 時間行為特徵（Temporal Behavior）

| Feature | 說明 |
|------|------|
| night_cnt | 夜間交易筆數 |
| night_txn_ratio | 夜間交易比例 |
| mean_txn_gap | 平均交易時間間隔（秒） |
| min_txn_gap | 最短交易時間間隔 |
| std_txn_gap | 交易間隔標準差 |
| max_txn_per_day | 單日最大交易筆數 |

### 行為解釋
時間行為特徵反映帳戶的**交易節奏與時間分佈特性**。

- 正常帳戶多集中於白天、交易間隔較規律
- 警示帳戶可能出現：
  - 夜間交易比例偏高
  - 短時間連續交易（低 min_txn_gap）
  - 不穩定的交易節奏（高 std_txn_gap）

---

## 5. 交易通路使用比例特徵

| Feature | 說明 |
|------|------|
| atm_ratio | ATM 交易比例 |
| counter_ratio | 臨櫃交易比例 |
| mobile_bank_ratio | 行動銀行交易比例 |
| web_bank_ratio | 網路銀行交易比例 |
| voice_ratio | 語音交易比例 |
| eatm_ratio | eATM 交易比例 |
| epay_ratio | 電子支付交易比例 |
| system_txn_ratio | 系統排程交易比例 |
| unk_channel_ratio | 未知通路比例 |

### 行為解釋
此類特徵描述帳戶的**操作管道偏好**。

- 正常帳戶通路分佈通常較多元
- 警示帳戶可能高度集中於特定通路，例如：
  - 僅透過電子支付或系統排程進行交易
  - 大量未知（UNK）通路交易，顯示非典型使用方式

---

## 6. 通路集中度特徵

| Feature | 說明 |
|------|------|
| channel_entropy | 交易通路使用熵 |

### 行為解釋
通路熵用於衡量帳戶交易通路的**集中或分散程度**。

- 高 entropy：多通路均衡使用，較接近一般行為
- 低 entropy：高度集中於單一通路，可能為自動化或異常行為

---

## 7. 特徵與警示帳戶分類的關聯總結

整體而言，本研究使用之帳戶層級特徵可從以下面向刻畫帳戶行為：

1. 交易活躍度（頻率與天數）
2. 金額規模與波動性
3. 資金流向與交易關係
4. 交易時間節奏
5. 操作通路結構與集中度

這些特徵能有效將「交易層級資料」轉換為可用於機器學習模型的結構化行為指標，支援後續之警示帳戶分類任務。

---

## 8. 標籤說明（Label Definition）

- label = 1：警示帳戶（出現在 `acct_alert.csv`）
- label = 0：非警示帳戶

分類模型將以上述帳戶行為特徵作為輸入，以預測帳戶是否屬於警示帳戶。