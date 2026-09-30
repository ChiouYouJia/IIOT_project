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

1. **超越基線**: 論文使用 DT / RF / ANN，最佳 F1-score 在攻擊辨識上仍不理想 (IP-Scan F1=0.52)
2. **時序異常偵測**: 論文明確指出 sequence anomaly detection 是未來方向
3. **多模態融合**: 結合 network flow + process state variables 進行異常偵測
4. **無監督 / 半監督**: 探索不需要標記資料的偵測方法
5. **聯邦式學習**: 模擬分散式 ICS 場域，在保護資料隱私的前提下協同訓練異常偵測模型

## 研究方法

### 方法一：Transformer-based Anomaly Detection
- **Anomaly Transformer** (ICLR 2022) — 利用 Association Discrepancy 進行無監督異常偵測
- 適用於 ICS 流量的時序異常偵測

### 方法二：Graph Neural Network (GNN)
- 將網路流量建模為圖結構 (節點=ICS 元件, 邊=通訊流量)
- 使用 GNN 捕捉 ICS 元件間的通訊模式異常

### 方法三：Contrastive Learning
- 使用對比學習建構 normal behavior representation
- 偏離正常表示的流量即為異常

### 方法四：LSTM / Temporal CNN + Attention
- 將流量資料轉為時間序列，利用序列模型捕捉攻擊前後的行為變化

### 方法五：Federated Learning (聯邦式學習)
- 模擬多個 ICS 場域 (工廠/子站) 各自訓練本地模型，透過聯邦學習聚合全域模型
- 解決 ICS 資料隱私與不可共享的核心問題 — 論文指出「some datasets are highly anonymized and cannot be shared due to confidentiality concerns」
- 探索 Non-IID 資料分佈下的異常偵測效能 (不同場域面臨不同攻擊類型)
- 比較 FedAvg, FedProx, FedBN 等聯邦策略
- 結合差分隱私 (Differential Privacy) 進一步保護敏感工控資料

### 擴展方向：多模態融合 (Network + Process Variables)
- 同時分析網路流量與物理過程變數 (水位、閥門狀態等)
- 使用 Cross-Attention 機制融合兩種模態的資訊

## 專案結構

```
IIOT_project/
├── README.md                    # 本文件
├── RESEARCH_PLAN.md             # 詳細研究計畫 (7-8 週)
├── setup/
│   └── environment.yml          # Conda 環境設定
├── data/
│   ├── raw/                     # 原始資料 (從 Kaggle 下載)
│   └── processed/               # 前處理後的資料
├── notebooks/
│   ├── 01_eda.ipynb             # 探索性資料分析
│   ├── 02_baseline.ipynb        # 基線模型複現
│   ├── 03_transformer.ipynb     # Anomaly Transformer
│   ├── 04_gnn.ipynb             # GNN-based detection
│   ├── 05_contrastive.ipynb     # Contrastive Learning
│   ├── 06_temporal.ipynb        # LSTM / TCN + Attention
│   ├── 07_multimodal.ipynb      # 多模態融合
│   └── 08_federated.ipynb       # 聯邦式學習
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py            # 資料載入
│   │   └── preprocess.py        # 資料前處理
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py          # DT / RF / ANN 基線
│   │   ├── anomaly_transformer.py
│   │   ├── gnn_detector.py
│   │   ├── contrastive.py
│   │   ├── temporal.py
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
python src/train.py --config configs/default.yaml --model transformer
```

## 評估指標

- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- Detection Latency (偵測延遲)
- False Positive Rate (誤報率) — 在 ICS 環境中特別重要
- Communication Cost (通訊成本) — 聯邦學習專用指標

## 參考文獻

- Xu, J. et al. (2022). *Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy.* ICLR 2022.
- Deng, A. & Hooi, B. (2021). *Graph Neural Network-Based Anomaly Detection in Multivariate Time Series.* AAAI 2021.
- Shenkar, T. & Wolf, L. (2022). *Anomaly Detection for Tabular Data with Internal Contrastive Learning.* ICLR 2022.
- Lai, K. et al. (2024). *Nominality Score Conditioned Time Series Anomaly Detection by Point/Sequential Reconstruction.* NeurIPS 2024.
- McMahan, B. et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS 2017.
- Li, T. et al. (2020). *Federated Optimization in Heterogeneous Networks (FedProx).* MLSys 2020.
- Nguyen, T. D. et al. (2019). *DIoT: A Federated Self-learning Anomaly Detection System for IoT.* IEEE ICDCS 2019.
- Mothukuri, V. et al. (2021). *A Survey on Security and Privacy of Federated Learning.* Future Generation Computer Systems.

## License

This project is for academic research purposes. The ICS-Flow dataset is licensed under CC BY 4.0.
