"""Train and evaluate the global causal-window LSTM + Attention model.

Examples:
    python scripts/lstm_attn_global_w20.py --task identification
    python scripts/lstm_attn_global_w20.py --task detection --seeds 0
"""

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.loader import load_split, sort_by_time
from src.models.lstm_attention import build_model
from src.utils.metrics import detection_metrics, identification_metrics, save_predictions


DEFAULT_METHOD = "lstm_attn_global_w20"
DEFAULT_WINDOW = 20
SEEDS = (0, 1, 2, 3, 4)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_windows(data: dict, parts=("train", "val"), window=DEFAULT_WINDOW):
    """Build global causal windows only for the requested target splits."""
    order = sort_by_time(data["df"]).index.to_numpy()
    X = data["X_all"].loc[order].to_numpy(dtype=np.float32)
    position = pd.Series(np.arange(len(order)), index=order)
    padded = np.vstack([np.zeros((window - 1, X.shape[1]), dtype=np.float32), X])

    result = {}
    for part in parts:
        flow_ids = data[f"X_{part}"].index
        indices = position.loc[flow_ids].to_numpy()
        result[part] = np.stack([padded[i:i + window] for i in indices]).astype(
            np.float32, copy=False
        )
    return result


def predict_proba(model, X, device, batch_size=512):
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=batch_size, shuffle=False)
    chunks = []
    with torch.no_grad():
        for (xb,) in loader:
            chunks.append(torch.softmax(model(xb.to(device)), dim=1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def validation_score(task, y_true, proba):
    y_pred = proba.argmax(axis=1)
    if task == "detection":
        return detection_metrics(y_true, y_pred, proba[:, 1])["f1"]
    return identification_metrics(y_true, y_pred, proba)["macro_f1"]


def train_model(model, X_train, y_train, X_val, y_val, task, device, args):
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train.astype(np.int64)),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(args.class_weights, dtype=torch.float32).to(device) if args.class_weights else None)
    if args.focal_gamma > 0:
        def focal_loss(logits, targets, gamma=args.focal_gamma):
            ce = nn.CrossEntropyLoss(reduction='none')(logits, targets)
            pt = torch.exp(-ce)
            return ((1 - pt) ** gamma * ce).mean()
        criterion = focal_loss
    best_score, best_epoch, best_state = -np.inf, 0, None
    stale = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb.to(device)), yb.to(device))
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        val_proba = predict_proba(model, X_val, device)
        current_score = validation_score(task, y_val, val_proba)
        print(f"  epoch={epoch:02d} val_score={current_score:.4f}")
        if current_score > best_score + 1e-6:
            best_score = current_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_score, best_epoch


def hardware_name(device):
    return torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"


def run_one(task, seed, args, device):
    set_seed(seed)
    data = load_split(task=task, split_name=args.split)

    # Test windows are deliberately built only after training/early stopping.
    windows = build_windows(data, parts=("train", "val"), window=args.window)
    X_train, X_val = windows["train"], windows["val"]
    y_train = data["y_train"].to_numpy()
    y_val = data["y_val"].to_numpy()
    num_classes = 2 if task == "detection" else 6
    model = build_model(
        X_train.shape[-1], num_classes, args.hidden_size,
        dropout=args.dropout, lstm_dropout=args.lstm_dropout,
    ).to(device)

    start = time.time()
    best_val, best_epoch = train_model(
        model, X_train, y_train, X_val, y_val, task, device, args
    )
    train_time = time.time() - start

    print(f"{args.method} {task} seed={seed}: best_val_score={best_val:.4f}")
    if args.val_only:
        return

    val_proba = predict_proba(model, X_val, device)
    val_score = val_proba[:, 1] if task == "detection" else val_proba
    save_predictions(
        args.method, task, seed, data["X_val"].index, val_proba.argmax(axis=1), val_score,
        split_name=args.split, part="val",
    )

    X_test = build_windows(data, parts=("test",), window=args.window)["test"]
    start = time.time()
    test_proba = predict_proba(model, X_test, device)
    infer_time = time.time() - start
    test_score = test_proba[:, 1] if task == "detection" else test_proba
    save_predictions(
        args.method, task, seed, data["X_test"].index, test_proba.argmax(axis=1), test_score,
        meta={
            "member": "A",
            **({"detection_from": "separate"} if task == "detection" else {}),
            "n_params": sum(p.numel() for p in model.parameters()),
            "train_time_s": train_time,
            "infer_time_s": infer_time,
            "hardware": hardware_name(device),
            "notes": (
                f"hidden={args.hidden_size}, lr={args.learning_rate}, epochs={args.epochs}, "
                f"patience={args.patience}, best_epoch={best_epoch}, best_val_score={best_val:.6f}, "
                f"dropout={args.dropout}, lstm_dropout={args.lstm_dropout}, grad_clip={args.grad_clip}, "
                f"weight_decay={args.weight_decay}, focal_gamma={args.focal_gamma}"
            ),
            "sequence": "global",
            "window": args.window,
        },
        split_name=args.split,
    )
    print(f"{args.method} {task} seed={seed} done: test predictions saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("detection", "identification"), required=True)
    parser.add_argument("--split", default="episode_block_v1")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2, help="Attention output dropout")
    parser.add_argument("--lstm-dropout", type=float, default=0.0, help="LSTM layer dropout")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Gradient clipping norm (0=off)")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="L2 regularization")
    parser.add_argument("--lr-scheduler", choices=("none", "cosine"), default="none", help="Learning rate schedule")
    parser.add_argument("--focal-gamma", type=float, default=0.0, help="Focal loss gamma (0=off)")
    parser.add_argument("--class-weights", nargs="+", type=float, default=None,
                        help="Per-class weights for CrossEntropyLoss (identification only)")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--val-only", action="store_true",
                        help="只訓練並輸出 val score，不建立 test window 或寫 prediction 檔")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定 cuda，但目前 PyTorch 沒有可用的 CUDA")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    print(f"task={args.task}, method={args.method}, window={args.window}, device={device}")
    for seed in args.seeds:
        run_one(args.task, seed, args, device)


if __name__ == "__main__":
    main()
