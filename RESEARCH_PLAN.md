# 研究計畫：ICS 異常偵測 — 基於 ICS-Flow 資料集的深度學習方法

**時程**: 7-8 週
**目標**: 使用現代深度學習方法改進 ICS 網路流量異常偵測，超越論文基線 (RF: F1=0.995 detection, F1=0.52~0.98 identification)

---

## 第 1 週：資料探索與環境建置

### 目標
- 深入理解 ICS-Flow 資料集結構與特性
- 建立可復現的實驗環境

### 任務
- [ ] 環境建置 (Python, PyTorch, DGL/PyG, scikit-learn)
- [ ] 下載 ICS-Flow 資料集 (Kaggle)
- [ ] 探索性資料分析 (EDA)
  - 各攻擊類型的分佈統計
  - 特徵相關性分析
  - 時序分布分析 (正常流量 vs 攻擊流量的時間模式)
  - 缺失值與異常值處理
- [ ] 資料前處理 pipeline
  - 特徵標準化 (Min-Max / Z-score)
  - 時間窗口切分 (用於序列模型)
  - IT / NST 標記策略的差異分析
- [ ] Process state variables 的初步分析

### 交付物
- `notebooks/01_eda.ipynb` — 完整的 EDA 報告
- `src/data/loader.py` + `preprocess.py` — 資料處理模組

---

## 第 2 週：基線模型複現與分析

### 目標
- 複現論文中的 DT / RF / ANN 基線
- 建立評估框架作為後續實驗的比較基準

### 任務
- [ ] 複現論文的前處理流程
  - MRMR 特徵選擇 (23 features)
  - Train/Val/Test split (50/20/30)
- [ ] 實作基線模型
  - Decision Tree
  - Random Forest
  - ANN (MLP)
- [ ] 建立統一的評估框架
  - Binary detection (Normal vs Attack)
  - Multi-class identification (Normal, DDoS, IP-Scan, Port-Scan, MitM, Replay)
  - Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix
- [ ] 分析基線模型的錯誤模式
  - IP-Scan 為何 F1 低？
  - Replay vs MitM 混淆的原因

### 交付物
- `notebooks/02_baseline.ipynb`
- `src/models/baseline.py`
- `src/utils/metrics.py`
- 基線結果表格

---

## 第 3 週：Anomaly Transformer

### 目標
- 實作 Anomaly Transformer 用於 ICS 時序異常偵測
- 驗證 Association Discrepancy 在 ICS 領域的效果

### 背景
論文指出「analyzing the predecessor and successor is a promising technique」。Anomaly Transformer (ICLR 2022) 利用 series-association 和 prior-association 的差異來偵測異常，適合發現時序中的異常模式。

### 任務
- [ ] 將 flow records 轉換為時間序列格式
  - 以固定時間窗口 (e.g., 10s, 30s, 60s) 聚合流量特徵
  - 每個窗口產生一個多變量時間序列樣本
- [ ] 實作 Anomaly Transformer
  - Anomaly-Attention mechanism
  - Association Discrepancy 計算
  - Minimax 訓練策略
- [ ] 無監督訓練 (僅使用正常流量)
- [ ] 實驗不同時間窗口大小的影響
- [ ] 與基線比較

### 交付物
- `notebooks/03_transformer.ipynb`
- `src/models/anomaly_transformer.py`
- 實驗結果 (detection + identification)

---

## 第 4 週：Graph Neural Network

### 目標
- 將 ICS 網路通訊建模為動態圖，利用 GNN 捕捉結構異常

### 背景
ICS 的通訊模式具有明確的拓撲結構 (PLC-1, PLC-2, HMI-1/2/3, Attacker)。GNN 可以學習正常的通訊模式圖結構，攻擊會改變圖的拓撲或邊的特徵。

### 任務
- [ ] 圖結構建模
  - 節點 = ICS 元件 (依 IP/MAC 地址)
  - 邊 = 時間窗口內的通訊流量
  - 邊特徵 = 聚合的 flow features
  - 節點特徵 = 元件的通訊統計
- [ ] 實作 GNN 模型
  - 參考 GDN (Graph Deviation Network, AAAI 2021)
  - Graph Attention Network (GAT) 用於學習注意力權重
  - 基於預測誤差的異常偵測
- [ ] 動態圖建構 (不同時間窗口的圖序列)
- [ ] 實驗與分析
  - GNN 是否能改善 IP-Scan 偵測?
  - 圖結構在攻擊期間如何變化?

### 交付物
- `notebooks/04_gnn.ipynb`
- `src/models/gnn_detector.py`
- 圖結構視覺化

---

## 第 5 週：Contrastive Learning + LSTM/TCN

### 目標
- 使用對比學習建構正常行為的表示空間
- 實作 LSTM/TCN 序列模型

### 任務 A：Contrastive Learning
- [ ] 實作 self-supervised contrastive framework
  - 資料增強策略 (jittering, scaling, permutation)
  - Contrastive loss (NT-Xent / SupCon)
  - 特徵提取 + 異常分數計算 (距離 normal cluster 的距離)
- [ ] 無監督 + 半監督設定的比較

### 任務 B：LSTM / TCN + Attention
- [ ] LSTM-based 序列異常偵測
  - 使用滑動窗口產生序列樣本
  - Attention 機制聚焦關鍵時間步
- [ ] Temporal Convolutional Network (TCN)
  - 因果卷積 + 膨脹卷積
  - 適合捕捉不同時間尺度的模式
- [ ] 與 Transformer 方法比較序列建模能力

### 交付物
- `notebooks/05_contrastive.ipynb`
- `notebooks/06_temporal.ipynb`
- `src/models/contrastive.py`
- `src/models/temporal.py`

---

## 第 6 週：多模態融合 (Network Flow + Process Variables)

### 目標
- 結合網路流量與物理過程變數進行異常偵測
- 這是論文明確提出但未實作的未來方向

### 背景
> "network attacks not only impact network traffic but can also modify physical processes. As the ICS-Flow dataset includes both types of data, a potential direction for future research would be to integrate network monitoring with physical process monitoring."

### 任務
- [ ] Process state variables 分析
  - 水位、閥門狀態、輸送帶等變數的時序分析
  - 攻擊對物理過程的影響分析
- [ ] 多模態融合架構設計
  - Network encoder (Transformer / GNN)
  - Process encoder (LSTM / TCN)
  - Cross-Attention 融合機制
  - Joint anomaly scoring
- [ ] 實作與訓練
  - 時間對齊 (網路流量 vs 過程變數)
  - 融合策略比較 (early fusion, late fusion, cross-attention)
- [ ] 分析融合是否改善偵測效果
  - 特別關注 MitM (false data injection) 是否更容易偵測

### 交付物
- `notebooks/07_multimodal.ipynb`
- `src/models/multimodal.py` (如需要)
- 融合 vs 單模態的比較分析

---

## 第 7 週：綜合比較與消融實驗

### 目標
- 系統性比較所有方法
- 消融實驗驗證各元件的貢獻

### 任務
- [ ] 統一實驗設定下的全面比較
  - 同樣的 train/val/test split
  - 同樣的前處理流程
  - IT vs NST 標記策略的影響
- [ ] 消融實驗
  - 特徵子集的影響 (flow / general / TCP)
  - 時間窗口大小的影響
  - 模型元件消融 (e.g., attention 機制, 圖結構)
- [ ] 計算效率分析
  - 訓練時間 / 推論時間
  - 模型參數量
  - 適用於即時偵測的可行性
- [ ] 錯誤分析
  - 各方法的 confusion matrix 比較
  - 不同攻擊類型的最佳偵測方法
  - False positive 分析

### 交付物
- 完整的比較結果表格
- 視覺化圖表 (bar chart, radar chart, confusion matrices)
- `results/` 目錄下的實驗結果

---

## 第 8 週：論文撰寫與整理

### 目標
- 撰寫研究報告或投稿論文
- 整理可復現的程式碼

### 任務
- [ ] 撰寫研究報告
  - Introduction & Motivation
  - Related Work (近 2 年的 ICS 異常偵測方法)
  - Methodology (各方法的詳細描述)
  - Experiments & Results
  - Discussion & Analysis
  - Conclusion & Future Work
- [ ] 程式碼整理
  - 確保所有實驗可復現
  - 添加必要的註解與文件
  - 更新 README 與 requirements
- [ ] GitHub repo 整理
  - 清理不必要的檔案
  - 確保 .gitignore 完整
  - 撰寫 CONTRIBUTING.md (如需要)

### 交付物
- 研究報告 (PDF)
- 整理完成的 GitHub repo
- 實驗結果摘要

---

## 預期成果

| 方法 | 類型 | 預期改善方向 |
|------|------|-------------|
| Anomaly Transformer | 無監督 / 時序 | 減少 false positive，改善序列層級偵測 |
| GNN | 半監督 / 結構 | 改善 IP-Scan 偵測 (利用圖拓撲變化) |
| Contrastive Learning | 無監督 | 不需標記資料，泛化能力更強 |
| LSTM/TCN + Attention | 監督 / 序列 | 改善 Replay vs MitM 區分 |
| 多模態融合 | 監督 | 結合物理過程資訊，全面提升偵測效果 |

## 風險與備案

1. **資料量不足**: 45K flow records 對深度學習可能偏少
   - 備案：使用資料增強、遷移學習、或結合其他 ICS 資料集 (SWaT, WUSTL-IIoT)
2. **GPU 資源**: Transformer / GNN 需要 GPU
   - 備案：使用 Google Colab Pro 或縮小模型規模
3. **時間壓力**: 8 週時間緊湊
   - 優先級：Transformer > LSTM/TCN > GNN > Contrastive > 多模態
   - 如果時間不夠，可以只做 Transformer + 一個其他方法的比較
