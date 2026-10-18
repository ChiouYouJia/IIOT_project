"""Global causal-window LSTM with temporal attention.

Usage:
    python scripts/lstm_attn_global_w20.py --task identification --seeds 0
    python scripts/lstm_attn_global_w20.py --task identification
    python scripts/lstm_attn_global_w20.py --task detection

The script follows docs/EVALUATION_PROTOCOL.md:
  - load_split() supplies the fixed episode_block_v1 split and train-only scaling.
  - windows are causal and global, with W=20.
  - validation is used for early stopping; test is only predicted once at the end.
  - every seed writes both val and test predictions through save_predictions().
"""

import argparse
import random
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_split, sort_by_time  # noqa: E402
from src.utils.metrics import (  # noqa: E402
    detection_metrics,
    identification_metrics,
    save_predictions,
)


METHOD = "lstm_attn_global_w20"
WINDOW = 20
SEEDS = (0, 1, 2, 3, 4)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_windows(data: dict, window: int = WINDOW):
    """Build causal global windows for the requested split's target flows."""
    order = sort_by_time(data["df"]).index.to_numpy()
    X = data["X_all"].loc[order].to_numpy(dtype=np.float32)
    position = pd.Series(np.arange(len(order)), index=order)
    padded = np.vstack(
        [np.zeros((window - 1, X.shape[1]), dtype=np.float32), X]
    )

    def windows(flow_ids):
        indices = position.loc[flow_ids].to_numpy()
        return np.stack([padded[i : i + window] for i in indices]).astype(
            np.float32, copy=False
        )

    return (
        windows(data["X_train"].index),
        windows(data["X_val"].index),
        windows(data["X_test"].index),
    )


class LSTMAttentionClassifier(nn.Module):
    """LSTM encoder followed by learned attention over the time dimension."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        hidden_states, _ = self.lstm(x)
        attention_logits = self.attention(hidden_states).squeeze(-1)
        attention_weights = torch.softmax(attention_logits, dim=1)
        context = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)
        return self.classifier(self.dropout(context))


def build_model(input_size: int, num_classes: int, hidden_size: int = 128):
    return LSTMAttentionClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=num_classes,
    )


def predict_proba(model, X, device, batch_size=512):
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X)),
        batch_size=batch_size,
        shuffle=False,
    )
    chunks = []
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device))
            chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def validation_score(task, y_true, proba):
    y_pred = proba.argmax(axis=1)
    if task == "detection":
        return detection_metrics(y_true, y_pred, proba[:, 1])["f1"]
    return identification_metrics(y_true, y_pred, proba)["macro_f1"]


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    task,
    device,
    epochs=40,
    batch_size=256,
    learning_rate=1e-3,
    patience=8,
):
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train.astype(np.int64)),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_score = -np.inf
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()

        val_proba = predict_proba(model, X_val, device)
        score = validation_score(task, y_val, val_proba)
        print(f"  epoch={epoch:02d} val_score={score:.4f}")

        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_score, best_epoch


def hardware_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return "cpu"


def run_one(task, seed, args, device):
    set_seed(seed)
    data = load_split(task=task, split_name=args.split)
    Xw_train, Xw_val, Xw_test = build_windows(data, WINDOW)
    y_train = data["y_train"].to_numpy()
    y_val = data["y_val"].to_numpy()
    num_classes = 2 if task == "detection" else 6

    model = build_model(Xw_train.shape[-1], num_classes, args.hidden_size).to(device)
    t0 = time.time()
    best_val, epochs_run = train_model(
        model,
        Xw_train,
        y_train,
        Xw_val,
        y_val,
        task,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )
    train_time = time.time() - t0

    val_proba = predict_proba(model, Xw_val, device)
    if task == "detection":
        val_score = val_proba[:, 1]
    else:
        val_score = val_proba
    save_predictions(
        METHOD,
        task,
        seed,
        flow_ids=data["X_val"].index,
        y_pred=val_proba.argmax(axis=1),
        score=val_score,
        split_name=args.split,
        part="val",
    )

    t0 = time.time()
    test_proba = predict_proba(model, Xw_test, device)
    infer_time = time.time() - t0
    if task == "detection":
        test_score = test_proba[:, 1]
    else:
        test_score = test_proba
    save_predictions(
        METHOD,
        task,
        seed,
        flow_ids=data["X_test"].index,
        y_pred=test_proba.argmax(axis=1),
        score=test_score,
        meta={
            "member": "A",
            **({"detection_from": "separate"} if task == "detection" else {}),
            "n_params": sum(p.numel() for p in model.parameters()),
            "train_time_s": train_time,
            "infer_time_s": infer_time,
            "hardware": hardware_name(device),
            "notes": (
                f"hidden={args.hidden_size}, lr={args.learning_rate}, "
                f"epochs={args.epochs}, patience={args.patience}, "
                f"best_epoch={epochs_run}, best_val_score={best_val:.6f}"
            ),
            "sequence": "global",
            "window": WINDOW,
        },
        split_name=args.split,
    )
    print(
        f"{task} seed={seed} done: best_val_score={best_val:.4f}, "
        f"train_time={train_time:.1f}s"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("detection", "identification"), required=True)
    parser.add_argument("--split", default="episode_block_v1")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 cuda，但目前 PyTorch 看不到 CUDA GPU")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    print(f"task={args.task}, method={METHOD}, device={device}")
    for seed in args.seeds:
        run_one(args.task, seed, args, device)


if __name__ == "__main__":
    main()
