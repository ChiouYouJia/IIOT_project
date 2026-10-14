"""統一評估指標 — 定義見 docs/EVALUATION_PROTOCOL.md。

成員只需呼叫 save_predictions() 輸出預測檔，正式指標一律由
scripts/evaluate.py 統一計算，不要自行在報告中引用 notebook 算出的數字。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)

from src.data.loader import CLASSES

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
PRED_DIR = RESULTS_DIR / "predictions"


def detection_metrics(y_true, y_pred, score=None):
    """二元偵測 (1 = attack)。主要指標: f1；ICS 場域特別重視 fpr (誤報率)。"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m = {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    if score is not None:
        m["roc_auc"] = roc_auc_score(y_true, score)
        m["pr_auc"] = average_precision_score(y_true, score)
    return m


def identification_metrics(y_true, y_pred, proba=None):
    """多類別辨識。主要指標: macro_f1 (ip-scan 樣本極少，weighted F1 會掩蓋其表現)。"""
    labels = list(range(len(CLASSES)))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    m = {
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    m.update({f"f1_{c}": v for c, v in zip(CLASSES, per_class)})
    if proba is not None:
        m["roc_auc_ovr"] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=labels)
    return m


def save_predictions(method, task, seed, flow_ids, y_pred, score=None, meta=None,
                     split_name="episode_block_v1", part="test"):
    """輸出標準預測檔到 results/predictions/<method>/<split>/<task>_seed<seed>[_val].csv。

    每個 run 要呼叫兩次：part="val" (val 預測，用來自動計算 val_score) 與 part="test" (附 meta)。

    Args:
        method: 方法名稱，如 "rf_baseline", "lstm_attn", "fl_fedavg_noniid", "gnn_gat"
        task: "detection" | "identification"
        seed: 隨機種子 (0-4)
        flow_ids: 該 part 的 flow_id (必須涵蓋整個 val / test split，不可遺漏或重複)
        y_pred: 預測標籤 (detection: 0/1；identification: 0..5 依 CLASSES 順序)
        score: detection -> shape (n,) 的 P(attack)；
               identification -> shape (n, 6) 的各類別機率
        meta: dict，只在 part="test" 時寫出；必填欄位見 docs/EVALUATION_PROTOCOL.md §7
        part: "test" | "val"
    """
    if part not in ("test", "val"):
        raise ValueError(f"part 必須是 'test' 或 'val'，收到 {part!r}")
    out_dir = PRED_DIR / method / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"flow_id": np.asarray(flow_ids), "y_pred": np.asarray(y_pred).astype(int)})
    if score is not None:
        score = np.asarray(score)
        if task == "detection":
            df["score"] = score.reshape(-1)
        else:
            for i, c in enumerate(CLASSES):
                df[f"prob_{c}"] = score[:, i]
    stem = f"{task}_seed{seed}" + ("_val" if part == "val" else "")
    df.to_csv(out_dir / f"{stem}.csv", index=False)

    if part == "test":
        meta = {"method": method, "task": task, "seed": seed, "split": split_name, **(meta or {})}
        (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return out_dir / f"{stem}.csv"


def print_results(results, title="Results"):
    print(f"\n{'=' * 50}\n {title}\n{'=' * 50}")
    for k, v in results.items():
        print(f" {k:<20s} {v:.4f}")


def full_report(y_true, y_pred):
    return classification_report(y_true, y_pred, labels=list(range(len(CLASSES))),
                                 target_names=CLASSES, zero_division=0)
