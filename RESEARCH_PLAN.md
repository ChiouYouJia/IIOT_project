# 研究計畫：ICS 異常偵測 — 基於 ICS-Flow 資料集的深度學習方法

**時程**: 8 週
**成員**: 3 人
**目標**: 使用現代深度學習方法改進 ICS 網路流量異常偵測，超越論文基線 (RF: F1=0.995 detection, F1=0.52~0.98 identification)

---

## 成員分工

| 成員 | 負責方法 | 核心任務 |
|------|---------|---------|
| **成員 A** | LSTM + Attention | 時序異常偵測 |
| **成員 B** | Federated Learning | 聯邦式學習框架 |
| **成員 C** | GNN (GAT) | 圖結構偵測 |

### 共同任務
- **第 1-2 週**: EDA、環境建置、基線複現
- **第 7 週**: 整理各自實驗結果，綜合比較
- **第 8 週**: 報告撰寫

### 協作時程圖

```
        Week 1    Week 2    Week 3    Week 4    Week 5    Week 6    Week 7    Week 8
       ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
全  員  │  EDA    │ 基線複現 │         │         │         │         │綜合比較 │報告撰寫 │
       ├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
成員 A │         │         │LSTM 實作│LSTM 實驗│ 調參    │         │整理結果 │  撰寫   │
       ├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
成員 B │         │         │FL 框架  │FL 實驗  │Non-IID  │         │整理結果 │  撰寫   │
       ├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
成員 C │         │         │GNN 建模 │GNN 實驗 │ 調參    │         │整理結果 │  撰寫   │
       └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
```

**第 6 週為緩衝週**，用來處理前幾週未完成的工作或進一步改善結果。

---

## 第 1-2 週：資料探索與基線複現

### 目標
- 理解 ICS-Flow 資料集結構
- 複現論文基線模型作為比較基準

### 任務
- [ ] 環境建置 (Python, PyTorch, PyG, Flower, scikit-learn)
- [ ] 下載 ICS-Flow 資料集 (Kaggle)
- [ ] EDA：攻擊類型分佈、特徵相關性
- [ ] 資料前處理：特徵標準化、train/val/test split (50/20/30)
- [ ] 複現基線模型 (DT / RF / ANN)
- [ ] 建立統一評估框架 (Accuracy, F1, Confusion Matrix)

### 交付物
- `notebooks/01_eda.ipynb`, `notebooks/02_baseline.ipynb`
- `src/data/loader.py`, `src/models/baseline.py`, `src/utils/metrics.py`

---

## 第 3-5 週：各成員獨立實作

### 成員 A：LSTM + Attention

**目標**: 用 LSTM 捕捉 ICS 流量的時序異常模式

| 週次 | 任務 |
|------|------|
| 第 3 週 | 將 flow records 轉為滑動窗口時間序列，實作 LSTM + Attention 模型 |
| 第 4 週 | 訓練 + 測試，與基線比較 (binary detection + multi-class) |
| 第 5 週 | 調參（hidden size, 窗口大小），分析 confusion matrix |

**核心實驗**:
- [ ] Binary detection: Normal vs Attack
- [ ] Multi-class identification: 各攻擊類型的 F1
- [ ] 與基線 RF 的效能比較

**交付物**: `src/models/lstm_attention.py`, `notebooks/03_lstm.ipynb`

### 成員 B：Federated Learning

**目標**: 模擬分散式 ICS 場域，驗證聯邦學習在 ICS 異常偵測的可行性

| 週次 | 任務 |
|------|------|
| 第 3 週 | 設計資料分割策略 (IID / Non-IID)，搭建 Flower 框架 |
| 第 4 週 | 實作 FedAvg / FedProx，訓練 + 測試 |
| 第 5 週 | Non-IID 實驗（不同 client 面臨不同攻擊），比較 federated vs centralized |

**核心實驗**:
- [ ] Centralized vs Federated 效能比較
- [ ] IID vs Non-IID 資料分佈的影響
- [ ] FedAvg vs FedProx 聚合策略比較

**交付物**: `src/models/federated.py`, `notebooks/05_federated.ipynb`

### 成員 C：Graph Neural Network

**目標**: 將 ICS 通訊建模為圖結構，利用 GNN 偵測拓撲異常

| 週次 | 任務 |
|------|------|
| 第 3 週 | 圖結構建模（節點=ICS 元件, 邊=通訊流量），實作 GAT |
| 第 4 週 | 訓練 + 測試，與基線比較 |
| 第 5 週 | 調參，分析 IP-Scan 偵測效果（GNN 天然適合偵測新增節點） |

**核心實驗**:
- [ ] Binary detection + multi-class identification
- [ ] IP-Scan F1 改善分析（原論文僅 0.52）
- [ ] 與基線 RF 的效能比較

**交付物**: `src/models/gnn_detector.py`, `notebooks/04_gnn.ipynb`

---

## 第 6 週：緩衝週

- 處理前幾週未完成的實驗
- 改善模型效能或修正 bug
- 準備第 7 週的綜合比較

---

## 第 7 週：綜合比較

### 任務
- [ ] 統一實驗設定下比較三種方法 + 基線
- [ ] 各方法的 Confusion Matrix 比較
- [ ] 整理成結果表格與視覺化圖表

### 交付物
- 完整的比較結果表格
- `results/` 目錄下的實驗結果

---

## 第 8 週：報告撰寫

### 任務
- [ ] 撰寫研究報告
  - Introduction & Motivation
  - Related Work
  - Methodology (三種方法)
  - Experiments & Results
  - Conclusion
- [ ] 程式碼整理，確保可復現
- [ ] GitHub repo 整理

### 交付物
- 研究報告 (PDF)
- 整理完成的 GitHub repo

---

## 相關文獻

### 成員 B 必讀
- **Dehlaghi-Ghadim et al. (ICMLA 2023)** — 原作者在 ICSSIM 上做 FL，結論：federated ≥ centralized > local

### 成員 A 參考
- **Al-Naimi & Belhi (ICTIS 2024)** — ICSSIM 上的深度學習異常偵測

### 成員 C 參考
- **Physics-Guided Contrastive Temporal Graph Learning (Nature Sci. Reports, 2026)** — 引用 ICS-Flow，圖方法在 ICS 上的應用

### 全員參考
- **Dehlaghi-Ghadim et al. (Electronics, 2025)** — 原作者最新方向 (LLM-based IDS)

---

## 預期成果

| 方法 | 負責人 | 預期改善方向 |
|------|-------|-------------|
| LSTM + Attention | 成員 A | 捕捉時序模式，改善 Replay vs MitM 區分 |
| Federated Learning | 成員 B | 不共享原始資料即可協同偵測 |
| GNN (GAT) | 成員 C | 利用圖拓撲變化改善 IP-Scan 偵測 |

## 風險與備案

1. **資料量不足**: 45K flow records 對深度學習可能偏少
   - 備案：資料增強、縮小模型規模
2. **GPU 資源**
   - 備案：使用 Google Colab 或縮小 batch size
3. **時間壓力**: 第 6 週為緩衝，優先完成核心實驗
