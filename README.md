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

1. **超越基線**: 論文使用 DT / RF / ANN，IP-Scan F1 僅 0.52，有明確改進空間
2. **時序異常偵測**: 論文指出 sequence anomaly detection 是未來方向
3. **聯邦式學習**: 模擬分散式 ICS 場域，在保護資料隱私的前提下協同訓練

## 研究方法

### 方法一：LSTM + Attention（成員 A）
- 將流量資料轉為時間序列，利用 LSTM 捕捉攻擊前後的行為變化
- 加入 Attention 機制聚焦關鍵時間步
- 論文明確指出「analyzing the predecessor and successor is a promising technique」

### 方法二：Federated Learning 聯邦式學習（成員 B）
- 模擬多個 ICS 場域各自訓練本地模型，透過聯邦學習聚合全域模型
- 探索 Non-IID 資料分佈下的異常偵測效能
- 比較 FedAvg / FedProx 聯邦策略

### 方法三：Graph Neural Network（成員 C）
- 將網路流量建模為圖結構 (節點=ICS 元件, 邊=通訊流量)
- 使用 GAT 捕捉 ICS 元件間的通訊模式異常

## 專案結構

```
IIOT_project/
├── README.md                    # 本文件
├── RESEARCH_PLAN.md             # 詳細研究計畫 (8 週)
├── setup/
│   └── environment.yml          # Conda 環境設定
├── data/
│   ├── raw/                     # 原始資料 (從 Kaggle 下載)
│   └── processed/               # 前處理後的資料
├── notebooks/
│   ├── 01_eda.ipynb             # 探索性資料分析
│   ├── 02_baseline.ipynb        # 基線模型複現
│   ├── 03_lstm.ipynb            # LSTM + Attention
│   ├── 04_gnn.ipynb             # GNN-based detection
│   └── 05_federated.ipynb       # 聯邦式學習
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py            # 資料載入
│   │   └── preprocess.py        # 資料前處理
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py          # DT / RF / ANN 基線
│   │   ├── lstm_attention.py    # LSTM + Attention
│   │   ├── gnn_detector.py      # GNN
│   │   └── federated.py         # 聯邦式學習
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py           # 評估指標
│   │   └── visualization.py     # 視覺化工具
│   └── train.py                 # 訓練腳本
├── configs/
│   └── default.yaml             # 實驗配置
├── results/                     # 實驗結果
└── references/                  # 參考論文 PDF
```

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

### 3. 執行實驗

```bash
# EDA
jupyter notebook notebooks/01_eda.ipynb

# 訓練模型
python src/train.py --config configs/default.yaml --model lstm
```

## 評估指標

- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC

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
