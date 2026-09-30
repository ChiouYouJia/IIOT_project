import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

LABEL_COLS = ["IT-M-Label", "IT-B-Label", "NST-M-Label", "NST-B-Label"]
EXCLUDE_COLS = ["sAddress", "rAddress", "start", "startOffset", "end", "endOffset"]


def load_flow_dataset(data_dir=None, label_strategy="NST", task="detection"):
    """Load ICS-Flow dataset.

    Args:
        data_dir: Path to raw data directory
        label_strategy: 'IT' or 'NST' labeling strategy
        task: 'detection' (binary) or 'identification' (multi-class)
    """
    data_dir = Path(data_dir) if data_dir else RAW_DIR

    csv_files = list(data_dir.glob("*.csv"))
    flow_files = [f for f in csv_files if "flow" in f.name.lower() or "dataset" in f.name.lower()]
    if not flow_files:
        flow_files = csv_files

    if not flow_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    df = pd.concat([pd.read_csv(f) for f in flow_files], ignore_index=True)

    if task == "detection":
        label_col = f"{label_strategy}-B-Label"
    else:
        label_col = f"{label_strategy}-M-Label"

    return df, label_col


def preprocess(df, label_col, test_size=0.3, val_size=0.2):
    """Preprocess: drop columns, fill NaN, split, normalize."""
    drop_cols = [c for c in EXCLUDE_COLS + LABEL_COLS if c in df.columns and c != label_col]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = X.pop(label_col) if label_col in X.columns else df[label_col]

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val, random_state=42, stratify=y_temp
    )

    feat_min = X_train.min()
    feat_max = X_train.max()
    denom = (feat_max - feat_min).replace(0, 1)

    X_train = (X_train - feat_min) / denom
    X_val = (X_val - feat_min) / denom
    X_test = (X_test - feat_min) / denom

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "feature_names": list(X_train.columns),
    }
