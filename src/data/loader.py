"""ICS-Flow 資料載入 — 全組共用，規則見 docs/EVALUATION_PROTOCOL.md。

flow_id 定義為 data/raw/Dataset.csv 的原始列索引 (0-based)，
所有方法的預測結果都以 flow_id 對回測試集。
"""
import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
SPLIT_DIR = DATA_DIR / "splits"

DEFAULT_SPLIT = "episode_block_v1"

LABEL_COLS = ["IT_B_Label", "IT_M_Label", "NST_B_Label", "NST_M_Label"]

# 識別資訊與時間戳記：不可作為模型輸入特徵 (攻擊者 IP/MAC 在所有攻擊中固定，會造成捷徑學習)
# GNN 可用 IP 建立圖結構，但不可把 IP 本身當成節點特徵或 embedding。
ID_COLS = ["sAddress", "rAddress", "sMACs", "rMACs", "sIPs", "rIPs"]
TIME_COLS = ["startDate", "endDate", "start", "end", "startOffset", "endOffset"]
CATEGORICAL_COLS = ["protocol"]

# 多類別標籤的固定編碼順序，所有成員必須使用相同 index
CLASSES = ["Normal", "ddos", "ip-scan", "mitm", "port-scan", "replay"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}


def load_raw(data_dir=None):
    """讀取 Dataset.csv，index 即為 flow_id。"""
    data_dir = Path(data_dir) if data_dir else RAW_DIR
    df = pd.read_csv(data_dir / "Dataset.csv")
    df.index.name = "flow_id"
    return df


def load_split_assignment(split_name=DEFAULT_SPLIT):
    """回傳 Series: flow_id -> 'train' | 'val' | 'test'。"""
    s = pd.read_csv(SPLIT_DIR / f"{split_name}.csv", index_col="flow_id")
    return s["split"]


def get_labels(df, task="detection", label_strategy="NST"):
    """detection -> 0/1；identification -> 0..5 (依 CLASSES 順序)。"""
    if task == "detection":
        return df[f"{label_strategy}_B_Label"].astype(int)
    return df[f"{label_strategy}_M_Label"].map(CLASS_TO_ID).astype(int)


def build_features(df):
    """原始數值特徵 + protocol one-hot，尚未標準化。NaN 保留給 normalize 處理。"""
    X = df.drop(columns=ID_COLS + TIME_COLS + LABEL_COLS + CATEGORICAL_COLS)
    X = X.apply(pd.to_numeric, errors="coerce")
    proto = pd.get_dummies(df["protocol"], prefix="proto", dtype=float)
    return pd.concat([X, proto], axis=1)


def normalize(X, train_mask):
    """Min-max 標準化，統計量只用 train 計算；NaN 補 0。"""
    X = X.fillna(0)
    feat_min = X[train_mask].min()
    denom = (X[train_mask].max() - feat_min).replace(0, 1)
    return (X - feat_min) / denom


def load_split(task="detection", label_strategy="NST", split_name=DEFAULT_SPLIT, data_dir=None):
    """標準資料載入介面。

    回傳 dict，X_* 為 DataFrame、y_* 為 Series，index 皆為 flow_id。
    另附 `df` (原始資料，依時間排序前的原順序) 與 `X_all` / `split` 供時序模型與 GNN
    以完整時間軸建立滑動窗口或圖。
    """
    df = load_raw(data_dir)
    split = load_split_assignment(split_name).reindex(df.index)
    y = get_labels(df, task, label_strategy)
    X = normalize(build_features(df), (split == "train").to_numpy())

    out = {"df": df, "X_all": X, "y_all": y, "split": split, "feature_names": list(X.columns)}
    for part in ("train", "val", "test"):
        mask = (split == part).to_numpy()
        out[f"X_{part}"] = X[mask]
        out[f"y_{part}"] = y[mask]
    return out


# 舊介面保留相容
def load_flow_dataset(data_dir=None, label_strategy="NST", task="detection"):
    df = load_raw(data_dir)
    prefix = "B" if task == "detection" else "M"
    return df, f"{label_strategy}_{prefix}_Label"
