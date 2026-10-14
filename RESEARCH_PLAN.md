# 研究計畫：ICS 異常偵測 — 基於 ICS-Flow 資料集的深度學習方法

**時程**: 8 週
**成員**: 3 人
**目標**: 使用現代深度學習方法改進 ICS 網路流量異常偵測，超越我們在共用切分 `episode_block_v1` 上的 RF 基線 (detection F1 0.980；identification macro-F1 0.880，ip-scan F1 0.531)。原論文數字 (detection F1 0.995) 是逐筆隨機切分的結果，條件不同，不作為比較對象。

---

## 分類任務：二元 + 多類別 (全員必做)

每位成員的方法都必須同時完成以下兩個任務，**不可只做二元分類**：

| 任務 | 類別 | 主要指標 |
|------|------|---------|
| **Detection (二元)** | Normal / Attack | F1 (attack)、PR-AUC、FPR |
| **Identification (多類別)** | Normal / ddos / ip-scan / mitm / port-scan / replay (6 類) | Macro-F1、各類別 F1 (特別是 ip-scan) |

- 多類別才是本研究的重點：二元偵測 RF 基線已接近 F1 0.99，改進空間很小；多類別中 ip-scan F1 僅約 0.53，mitm / replay 也容易混淆。
- 多類別模型輸出層為 6 類 softmax，類別編碼固定使用 `src.data.loader.CLASSES`。
- 類別極度不平衡 (ip-scan 在 train 只有 96 筆)，建議嘗試 class-weighted loss / focal loss / 過採樣 (只能作用在 train)。
- 細節見 `docs/EVALUATION_PROTOCOL.md`。

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
- [x] 資料前處理與共用切分：`data/splits/episode_block_v1.csv` (見 `docs/EVALUATION_PROTOCOL.md`)
- [ ] 複現基線模型 (DT / RF / ANN)
- [x] 建立統一評估框架：`src/utils/metrics.py` + `scripts/evaluate.py`

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
| 第 3 週 | 設計 client 切分 (IID / Non-IID，只從 train 切，固定後 commit)，以逐筆 MLP 搭建 Flower 框架 |
| 第 4 週 | FL-MLP：FedAvg / FedProx、IID / Non-IID、federated vs centralized vs local-only。週末檢查 A 的 LSTM 是否在 val macro-F1 明顯勝過 ANN |
| 第 5 週 | 若 LSTM 明顯較好：用與 A 相同的 LSTM 架構跑 FL-LSTM 作為代表變體；否則以 FL-MLP 為代表，繼續 Non-IID 分析 |

**核心實驗**:
- [ ] Binary detection + multi-class identification (兩個任務都要在 FL 設定下完成)
- [ ] Centralized vs Federated 效能比較 (一律與同架構比較：FL-MLP 對 ANN，FL-LSTM 對 A 的 LSTM)
- [ ] IID vs Non-IID 資料分佈的影響 (Non-IID 下部分 client 缺少某些攻擊類別，觀察多類別 per-class F1 的變化)
- [ ] FedAvg vs FedProx 聚合策略比較

**交付物**: `src/models/federated.py`, `notebooks/05_federated.ipynb`, `data/splits/fl_<partition>_v1.csv`

> 本地模型為何先用 MLP：FL 要回答的問題 (資料分佈、聚合策略、聯邦 vs 集中) 與模型架構無關，MLP 訓練快、穩定，結論較乾淨；每個 client 只有約 5 千筆、Non-IID 下部分 client 沒有 ip-scan，LSTM 較難收斂。最後再用 FL-LSTM 回答「最好的模型在聯邦設定下保有多少效能」。

### 成員 C：Graph Neural Network

**目標**: 將 ICS 通訊建模為圖結構，利用 GNN 偵測拓撲異常

| 週次 | 任務 |
|------|------|
| 第 3 週 | 圖結構建模（節點=`sAddress`/`rAddress`, 邊=flow；注意 ARP flow 沒有 IP，ip-scan 全部是 ARP），實作 GAT |
| 第 4 週 | 訓練 + 測試，與基線比較 |
| 第 5 週 | 調參，分析 IP-Scan 偵測效果（GNN 天然適合偵測新增節點） |

**核心實驗**:
- [ ] Binary detection + multi-class identification
- [ ] IP-Scan F1 改善分析（RF 基線 0.531）
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
- [ ] 每人依 val 分數選出代表變體，填入 `docs/EVALUATION_PROTOCOL.md` §4 的表格 (第 7 週開始前)
- [ ] 依 `docs/EVALUATION_PROTOCOL.md` 比較三種方法 + 基線 (`python scripts/evaluate.py`)
- [ ] 二元與多類別兩個任務分別製作比較表
- [ ] 各方法的多類別 Confusion Matrix 比較 (重點：ip-scan、mitm vs replay 混淆)
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
