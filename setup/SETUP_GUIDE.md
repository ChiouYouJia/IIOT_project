# 環境建置指南

## 系統需求

- Python 3.11+
- CUDA 12.1+ (GPU 訓練，非必要但建議)
- 至少 8GB RAM
- 約 3GB 磁碟空間 (資料集 + 環境)

## 方法一：Conda (建議)

```bash
# 1. 安裝 Miniconda (如果還沒有的話)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 2. 建立環境
conda env create -f setup/environment.yml

# 3. 啟動環境
conda activate ics-anomaly

# 4. 確認安裝
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## 方法二：pip + venv

```bash
# 1. 建立虛擬環境
python3.11 -m venv venv
source venv/bin/activate

# 2. 安裝依賴
pip install -r setup/requirements.txt

# 3. 安裝 PyTorch (根據你的 CUDA 版本)
# CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 方法三：Google Colab

如果沒有本地 GPU，可以使用 Google Colab：

```python
# 在 Colab notebook 開頭執行
!git clone https://github.com/ChiouYouJia/IIOT_project.git
%cd IIOT_project
!pip install -r setup/requirements.txt
```

## 下載資料集

### Kaggle CLI

```bash
# 1. 前往 https://www.kaggle.com/settings
# 2. 點擊 "Create New Token" 下載 kaggle.json
# 3. 設定 credentials
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 4. 下載資料集
kaggle datasets download -d alirezadehlaghi/icssim -p data/raw --unzip
```

### 手動下載

1. 前往 https://www.kaggle.com/datasets/alirezadehlaghi/icssim
2. 點擊 "Download" 按鈕
3. 解壓至 `data/raw/` 目錄

### 資料集內容

下載後 `data/raw/` 應包含：
- `*.pcap` — 原始網路封包 (約 2GB)
- `*.csv` — 網路流量記錄 (flow dataset)
- Process state variables 記錄
- Attack log 檔案

## 驗證安裝

```bash
# 執行驗證腳本
python -c "
import torch
import numpy as np
import pandas as pd
import sklearn
print('All dependencies installed successfully!')
print(f'PyTorch: {torch.__version__}')
print(f'NumPy: {np.__version__}')
print(f'Pandas: {pd.__version__}')
print(f'Scikit-learn: {sklearn.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

## 常見問題

### Q: CUDA 版本不匹配？
安裝對應版本的 PyTorch：https://pytorch.org/get-started/locally/

### Q: 記憶體不足？
- 減小 batch size
- 使用 `torch.cuda.amp` 做 mixed precision training
- 先用 flow dataset 而非 raw pcap

### Q: Kaggle 下載失敗？
- 確認 `kaggle.json` 權限為 600
- 確認已同意資料集的使用條款 (在 Kaggle 網頁上)
