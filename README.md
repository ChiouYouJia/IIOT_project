# ICS Anomaly Detection with Modern Deep Learning

基於 **ICS-Flow** 資料集，使用近年最新的異常偵測技術對工業控制系統 (ICS) 網路流量進行入侵偵測研究。

## 論文來源

> Dehlaghi-Ghadim, A., Helali Moghadam, M., Balador, A., & Hansson, H. (2023).
> *Anomaly Detection Dataset for Industrial Control Systems.*
> IEEE Access, 11, 107982–107996.

**資料集**: [Kaggle - ICS-Flow (icssim)](https://www.kaggle.com/datasets/alirezadehlaghi/icssim)
**ICSFlowGenerator**: [GitHub](https://github.com/AlirezaDehlaghi/ICSFlow)

## 資料集概述

| 項目 | 說明 |
|------|------|
| 場景 | 瓶裝水填充工廠模擬 (ICSSIM) |
| 協定 | Modbus TCP |
| 原始封包 | 25,000,000+ packets (2GB PCAP) |
| 網路流量記錄 | 45,719 flow records |
| 特徵數 | 50 features + 4 labels |
| 攻擊類型 | Reconnaissance (IP-Scan, Port-Scan), DDoS, MitM (False Data Injection), Replay |
| 標記策略 | IT (Injection Timing), NST (Network Security Tools) |

## 研究目標

1. **超越基線**: 論文使用 DT / RF / ANN，IP-Scan F1 僅約 0.52 (我們的 RF 基線 0.531)，有明確改進空間；二元偵測與多類別辨識都要做
2. **時序異常偵測**: 論文指出 sequence anomaly detection 是未來方向
3. **聯邦式學習**: 模擬分散式 ICS 場域，在保護資料隱私的前提下協同訓練

## 研究方法

### 方法一：LSTM + Attention（成員 A）
- 將流量資料轉為時間序列，利用 LSTM 捕捉攻擊前後的行為變化
- 加入 Attention 機制聚焦關鍵時間步
- 論文明確指出「analyzing the predecessor and successor is a promising technique」

### 方法二：Federated Learning 聯邦式學習（成員 B）
- 模擬多個 ICS 場域各自訓練本地模型，透過聯邦學習聚合全域模型
- 主要實驗以逐筆 MLP 為本地模型；若 LSTM 明顯較好，再加做 FL-LSTM
- 探索 Non-IID 資料分佈下的異常偵測效能
- 比較 FedAvg / FedProx 聯邦策略

### 方法三：Graph Neural Network（成員 C）
- 將網路流量建模為圖結構 (節點=ICS 元件 `sAddress`/`rAddress`, 邊=通訊流量)
- 使用 GAT 捕捉 ICS 元件間的通訊模式異常

## 專案結構

```
IIOT_project/
├── README.md                    # 本文件
├── RESEARCH_PLAN.md             # 詳細研究計畫 (8 週)
├── setup/
│   └── environment.yml          # Conda 環境設定
├── docs/
│   └── EVALUATION_PROTOCOL.md   # 評估規範 (全員必讀)
├── data/
│   ├── raw/                     # 原始資料 (從 Kaggle 下載)
│   ├── splits/                  # 共用切分檔 (已 commit，不可自行重切)
│   └── processed/               # 前處理後的資料
├── notebooks/                   # (待建立) 01_eda / 02_baseline / 03_lstm / 04_gnn / 05_federated
├── src/
│   ├── data/
│   │   └── loader.py            # 資料載入 + 共用切分 (load_split, sort_by_time)
│   ├── models/                  # (待建立) baseline / lstm_attention / gnn_detector / federated
│   └── utils/
│       └── metrics.py           # 評估指標 + save_predictions
├── scripts/
│   ├── make_splits.py           # 產生共用切分 (已執行，勿重跑覆蓋)
│   ├── baseline_rf.py           # RF 基線 + 預測檔範本
│   └── evaluate.py              # 統一評估、產生排行榜
├── configs/
│   └── default.yaml             # 實驗配置 (只有 data 區塊強制共用)
├── results/
│   └── predictions/             # 各方法的預測檔 (commit)
└── references/                  # 參考論文 PDF
```

標示「待建立」的檔案由各成員依研究計畫建立。

## 快速開始

### 1. 環境建置

```bash
# 使用 conda
conda env create -f setup/environment.yml
conda activate ics-anomaly

# 或使用 pip
pip install -r setup/requirements.txt
```

### 2. 下載資料集

```bash
# 方法一：使用 Kaggle CLI
pip install kaggle
kaggle datasets download -d alirezadehlaghi/icssim -p data/raw --unzip

# 方法二：手動下載
# 前往 https://www.kaggle.com/datasets/alirezadehlaghi/icssim
# 下載後解壓至 data/raw/
```

### 3. 跑基線與評估

```bash
# RF 基線 (5 seeds × detection / identification)，輸出到 results/predictions/rf_baseline/
python scripts/baseline_rf.py

# 讀取所有人的預測檔，產生 results/leaderboard_episode_block_v1.md
python scripts/evaluate.py
```

各成員的模型請參考 `scripts/baseline_rf.py` 的寫法，用 `load_split()` 載入資料、`save_predictions()` 輸出預測。

## 評估規範

三位成員的結果須依 **[docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md)** 產出，才能橫向比較：

- 共用切分 `data/splits/episode_block_v1.csv` (以攻擊事件為單位，避免時序 / 圖模型資料洩漏)
- NST 標籤；禁用 IP / MAC / 時間戳記作為特徵
- 主要指標：detection → F1 (attack)、PR-AUC、FPR；identification → Macro-F1、ip-scan F1
- 每個設定 5 seeds (0–4)，報告 mean ± std；代表變體依 val 分數選出
- 以 `save_predictions()` 輸出 test 預測，統一由 `python scripts/evaluate.py` 計算排行榜

## 相關文獻 (使用 ICS-Flow / ICSSIM 的近期研究)

| 論文 | 年份 | 方法 | 關鍵發現 |
|------|------|------|---------|
| Dehlaghi-Ghadim et al., *Federated Learning for Network Anomaly Detection in a Distributed Industrial Environment* (ICMLA 2023) | 2023 | Federated Learning | 聯邦模型超越本地模型，匹配/超過集中式模型 |
| Al-Naimi & Belhi, *Deep Learning-Based Anomaly Detection in ICS Network Traffic* (ICTIS 2024, Springer) | 2024 | CNN (traffic-to-image) | 將 ICSSIM 網路流量轉換為視覺域，用 CNN 分類 |
| Dehlaghi-Ghadim et al., *Domain Knowledge-Infused Synthetic Data Generation for LLM-Based IDS* (Electronics 2025) | 2025 | LLM-based IDS | 用 ICSSIM 建構 6 種 MITRE ATT&CK 場景，以 LLM 做入侵偵測 |
| Omar, *Binary Image-Based Intrusion Detection for OT Networks* (SPIE 2025) | 2025 | SPHBI + CNN | Modbus TCP 封包影像化，加入應用層資訊後 accuracy 達 98.1% |
| *Physics-Guided Contrastive Temporal Graph Learning* (Nature Sci. Reports 2026) | 2026 | GNN + Contrastive | 結合物理約束的時序圖對比學習，引用 ICS-Flow |

## 參考文獻

- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation.
- Deng, A. & Hooi, B. (2021). *Graph Neural Network-Based Anomaly Detection in Multivariate Time Series.* AAAI 2021.
- McMahan, B. et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS 2017.
- Li, T. et al. (2020). *Federated Optimization in Heterogeneous Networks (FedProx).* MLSys 2020.
- Nguyen, T. D. et al. (2019). *DIoT: A Federated Self-learning Anomaly Detection System for IoT.* IEEE ICDCS 2019.

## License

This project is for academic research purposes. The ICS-Flow dataset is licensed under CC BY 4.0.
