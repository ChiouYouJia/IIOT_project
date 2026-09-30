# 研究計畫：ICS 異常偵測 — 基於 ICS-Flow 資料集的深度學習方法

**時程**: 8 週
**成員**: 3 人
**目標**: 使用現代深度學習方法改進 ICS 網路流量異常偵測，超越論文基線 (RF: F1=0.995 detection, F1=0.52~0.98 identification)

---

## 成員分工

### 總覽

| 成員 | 負責方法 | 核心任務 |
|------|---------|---------|
| **成員 A** | Anomaly Transformer + 多模態融合 | 時序異常偵測 + 網路/物理過程融合 |
| **成員 B** | Federated Learning + Contrastive Learning | 聯邦式學習框架 + 對比學習 |
| **成員 C** | GNN + LSTM/TCN | 圖結構偵測 + 序列模型 |

### 共同任務 (全員)
- **第 1 週**: 一起完成 EDA、資料前處理 pipeline、環境建置
- **第 2 週**: 一起複現基線模型 (DT/RF/ANN) 並建立統一評估框架
- **第 7 週**: 各自整理實驗結果，共同進行綜合比較
- **第 8 週**: 共同撰寫報告，各自撰寫負責方法的章節

### 成員 A：Transformer + 多模態融合

| 週次 | 任務 |
|------|------|
| 第 3 週 | 實作 Anomaly Transformer，時序異常偵測實驗 |
| 第 4 週 | 調參 + 不同時間窗口實驗 |
| 第 5 週 | Process state variables 分析，設計多模態融合架構 |
| 第 6 週 | 實作 Cross-Attention 融合，比較 early/late/cross-attention |
| 第 7 週 | 消融實驗 + 整理結果 |
| 第 8 週 | 撰寫 Transformer + 多模態章節 |

**交付物**: `anomaly_transformer.py`, `multimodal.py`, `03_transformer.ipynb`, `07_multimodal.ipynb`

### 成員 B：Federated Learning + Contrastive Learning

| 週次 | 任務 |
|------|------|
| 第 3 週 | 設計 Non-IID 資料分割策略，搭建 Flower 框架 |
| 第 4 週 | 實作 FedAvg/FedProx/FedBN，IID vs Non-IID 實驗 |
| 第 5 週 | 加入差分隱私 (DP-SGD)，隱私-效能 trade-off 分析 |
| 第 6 週 | 進階場景 (惡意 client 防禦, 個人化 FL) + Contrastive Learning 實作 |
| 第 7 週 | 聯邦 vs 集中式比較 + 消融實驗 |
| 第 8 週 | 撰寫 Federated Learning + Contrastive Learning 章節 |

**交付物**: `federated.py`, `contrastive.py`, `08_federated.ipynb`, `05_contrastive.ipynb`

### 成員 C：GNN + LSTM/TCN

| 週次 | 任務 |
|------|------|
| 第 3 週 | 圖結構建模 (節點/邊定義)，實作 GAT/GDN |
| 第 4 週 | 動態圖建構 + GNN 異常偵測實驗 |
| 第 5 週 | 實作 LSTM + Attention，滑動窗口序列模型 |
| 第 6 週 | 實作 TCN (因果卷積 + 膨脹卷積)，與 Transformer 比較 |
| 第 7 週 | 消融實驗 + 圖結構視覺化 |
| 第 8 週 | 撰寫 GNN + LSTM/TCN 章節 |

**交付物**: `gnn_detector.py`, `temporal.py`, `04_gnn.ipynb`, `06_temporal.ipynb`

### 協作時程圖

```
        Week 1    Week 2    Week 3    Week 4    Week 5    Week 6    Week 7    Week 8
       ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
全  員  │  EDA    │ 基線複現 │         │         │         │         │綜合比較 │論文撰寫 │
       ├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
成員 A │         │         │Transformer│ 調參   │Process分析│多模態融合│消融實驗 │  撰寫   │
       ├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
成員 B │         │         │FL 框架  │FL 策略  │ DP 隱私 │Byzantine│消融實驗 │  撰寫   │
       │         │         │         │         │         │+Contrast│         │         │
       ├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
成員 C │         │         │GNN 建模 │GNN 實驗 │LSTM+Attn│  TCN    │消融實驗 │  撰寫   │
       └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
```

---

## 第 1 週：資料探索與環境建置

### 目標
- 深入理解 ICS-Flow 資料集結構與特性
- 建立可復現的實驗環境

### 任務
- [ ] 環境建置 (Python, PyTorch, DGL/PyG, Flower/FedML, scikit-learn)
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

## 第 4 週：Graph Neural Network + Contrastive Learning / LSTM

### 目標
- 將 ICS 網路通訊建模為動態圖，利用 GNN 捕捉結構異常
- 實作對比學習與序列模型

### 任務 A：Graph Neural Network
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

### 任務 B：Contrastive Learning + LSTM/TCN
- [ ] 實作 self-supervised contrastive framework
  - 資料增強策略 (jittering, scaling, permutation)
  - Contrastive loss (NT-Xent / SupCon)
- [ ] LSTM / TCN + Attention 序列模型
  - 使用滑動窗口產生序列樣本
  - 因果卷積 + 膨脹卷積

### 交付物
- `notebooks/04_gnn.ipynb`
- `notebooks/05_contrastive.ipynb` + `notebooks/06_temporal.ipynb`
- `src/models/gnn_detector.py`, `contrastive.py`, `temporal.py`

---

## 第 5 週：Federated Learning (聯邦式學習)

### 目標
- 模擬分散式 ICS 場域，在不共享原始資料的前提下協同訓練異常偵測模型
- 探索聯邦學習在 ICS 安全領域的適用性與挑戰

### 背景
論文指出 ICS 資料集面臨的核心挑戰之一是「some datasets are highly anonymized and cannot be shared due to confidentiality concerns」。在真實世界中，不同工廠/子站的 ICS 資料因商業機密和安全法規無法集中訓練。Federated Learning 允許各場域在本地訓練模型，只共享模型參數，是解決 ICS 資料孤島問題的關鍵技術。

### 任務

#### A. 資料分割 — 模擬多場域
- [ ] 將 ICS-Flow 資料集切分為 N 個 client (模擬 N 個工廠/子站)
  - **IID 分割**: 各 client 均勻分配所有攻擊類型
  - **Non-IID 分割 (攻擊異質性)**: 不同 client 面臨不同攻擊
    - Client 1: 主要遭受 DDoS
    - Client 2: 主要遭受 MitM + Replay
    - Client 3: 主要遭受 Reconnaissance
    - Client 4: 全部為正常流量 (模擬未曾被攻擊的場域)
  - **Non-IID 分割 (數量異質性)**: 各 client 資料量不同 (模擬大工廠 vs 小型子站)
  - **Non-IID 分割 (時間異質性)**: 各 client 有不同時間段的資料

#### B. 聯邦學習框架實作
- [ ] 使用 **Flower (flwr)** 框架實作聯邦訓練
- [ ] 實作多種聯邦聚合策略
  - **FedAvg**: 基礎加權平均聚合 (McMahan et al., 2017)
  - **FedProx**: 加入 proximal term 處理 Non-IID 問題 (Li et al., 2020)
  - **FedBN**: 保留各 client 的 Batch Normalization 層處理 feature shift
  - **FedNova**: 正規化各 client 的本地更新步數差異
- [ ] 本地模型選擇
  - 將第 2-4 週最佳的模型 (e.g., Transformer, LSTM) 作為各 client 的本地模型
  - 比較不同本地模型架構下的聯邦效果

#### C. 差分隱私 (Differential Privacy)
- [ ] 在聯邦學習中加入 DP-SGD
  - 梯度裁剪 (gradient clipping) + 高斯噪聲 (Gaussian noise)
  - 探索隱私預算 ε 與模型效能的 trade-off
- [ ] 比較有無 DP 的偵測效能差異

#### D. 進階聯邦異常偵測場景
- [ ] **新攻擊泛化**: 某些 client 從未見過的攻擊類型，能否透過聯邦模型偵測？
- [ ] **惡意 client 防禦 (Byzantine-robust aggregation)**:
  - 模擬被攻陷的 client 發送惡意模型更新
  - 實作 Krum / Trimmed Mean 等拜占庭容錯聚合方法
- [ ] **個人化聯邦學習 (Personalized FL)**:
  - 全域模型 + 本地微調 (fine-tuning)
  - 各場域可以適應自己的流量特徵

#### E. 評估指標
- [ ] 與集中式訓練的效能比較 (centralized vs federated)
- [ ] 各 client 本地效能 vs 聯邦後的效能提升
- [ ] 通訊成本 (communication rounds, 傳輸資料量)
- [ ] 收斂速度 (多少輪聚合達到目標效能)
- [ ] 隱私-效能 trade-off 曲線

### 交付物
- `notebooks/08_federated.ipynb`
- `src/models/federated.py` — 聯邦訓練框架
- Non-IID 分割策略視覺化
- 聯邦 vs 集中式效能比較圖表
- 隱私預算 vs F1-Score 分析

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
- [ ] (選配) 多模態 + 聯邦學習：各場域擁有不同模態的資料

### 交付物
- `notebooks/07_multimodal.ipynb`
- `src/models/multimodal.py` (如需要)
- 融合 vs 單模態的比較分析

---

## 第 7 週：綜合比較與消融實驗

### 目標
- 系統性比較所有方法 (含聯邦學習)
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
  - 聯邦學習消融：IID vs Non-IID, 有無 DP, 聚合策略
- [ ] 計算效率分析
  - 訓練時間 / 推論時間
  - 模型參數量
  - 通訊開銷 (聯邦學習)
  - 適用於即時偵測的可行性
- [ ] 錯誤分析
  - 各方法的 confusion matrix 比較
  - 不同攻擊類型的最佳偵測方法
  - False positive 分析
  - 聯邦學習在不同 Non-IID 程度下的效能退化分析

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
  - Related Work (近 2 年的 ICS 異常偵測 + 聯邦學習方法)
  - Methodology (各方法的詳細描述，含聯邦學習架構)
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
| Federated Learning | 分散式 / 隱私保護 | 不共享原始資料即可協同偵測，解決資料孤島 |
| 多模態融合 | 監督 | 結合物理過程資訊，全面提升偵測效果 |

## 風險與備案

1. **資料量不足**: 45K flow records 對深度學習可能偏少
   - 備案：使用資料增強、遷移學習、或結合其他 ICS 資料集 (SWaT, WUSTL-IIoT)
2. **GPU 資源**: Transformer / GNN 需要 GPU
   - 備案：使用 Google Colab Pro 或縮小模型規模
3. **Non-IID 分割後資料更少**: 聯邦學習分割後每個 client 資料量有限
   - 備案：增加通訊輪數、使用 knowledge distillation、或採用 few-shot 方法
4. **時間壓力**: 8 週時間緊湊
   - 優先級：Transformer > Federated Learning > LSTM/TCN > GNN > Contrastive > 多模態
   - 如果時間不夠，可以聚焦 Transformer + Federated Learning 作為核心貢獻
