"""Causal global-window LSTM with temporal attention."""
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
from src.utils.metrics import detection_metrics, identification_metrics, save_predictions

METHOD = "lstm_attn_global_w20"
WINDOW = 20


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_windows(data, window=WINDOW):
    order = sort_by_time(data["df"]).index.to_numpy()
    X = data["X_all"].loc[order].to_numpy(np.float32)
    pos = pd.Series(np.arange(len(order)), index=order)
    padded = np.vstack([np.zeros((window - 1, X.shape[1]), np.float32), X])

    def windows(flow_ids):
        idx = pos.loc[flow_ids].to_numpy()
        return np.stack([padded[i:i + window] for i in idx]).astype(np.float32)

    return windows(data["X_train"].index), windows(data["X_val"].index), windows(data["X_test"].index)


class LSTMAttentionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1, bias=False)
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        states, _ = self.lstm(x)
        weights = torch.softmax(self.attention(states).squeeze(-1), dim=1)
        context = (states * weights.unsqueeze(-1)).sum(dim=1)
        return self.classifier(self.dropout(context))


def predict_proba(model, X, device, batch_size=512):
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=batch_size, shuffle=False)
    output = []
    with torch.no_grad():
        for (xb,) in loader:
            output.append(torch.softmax(model(xb.to(device)), dim=1).cpu().numpy())
    return np.concatenate(output)


def score(task, y_true, proba):
    pred = proba.argmax(1)
    if task == "detection":
        return detection_metrics(y_true, pred, proba[:, 1])["f1"]
    return identification_metrics(y_true, pred, proba)["macro_f1"]


def train_model(model, X_train, y_train, X_val, y_val, task, device, args):
    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.int64)))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_score, best_epoch, best_state, stale = -np.inf, 0, None, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()
        val_proba = predict_proba(model, X_val, device)
        val_score = score(task, y_val, val_proba)
        print(f"  epoch={epoch:02d} val_score={val_score:.4f}")
        if val_score > best_score + 1e-6:
            best_score, best_epoch, stale = val_score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    return best_score, best_epoch


def run(task, seed, args, device):
    set_seed(seed)
    data = load_split(task=task, split_name=args.split)
    X_train, X_val, X_test = build_windows(data)
    y_train, y_val = data["y_train"].to_numpy(), data["y_val"].to_numpy()
    classes = 2 if task == "detection" else 6
    model = LSTMAttentionClassifier(X_train.shape[-1], args.hidden_size, classes).to(device)

    start = time.time()
    best_val, best_epoch = train_model(model, X_train, y_train, X_val, y_val, task, device, args)
    train_time = time.time() - start

    val_proba = predict_proba(model, X_val, device)
    val_score = val_proba[:, 1] if task == "detection" else val_proba
    save_predictions(METHOD, task, seed, data["X_val"].index, val_proba.argmax(1), val_score,
                     split_name=args.split, part="val")

    start = time.time()
    test_proba = predict_proba(model, X_test, device)
    infer_time = time.time() - start
    test_score = test_proba[:, 1] if task == "detection" else test_proba
    meta = {
        "member": "A",
        **({"detection_from": "separate"} if task == "detection" else {}),
        "n_params": sum(p.numel() for p in model.parameters()),
        "train_time_s": train_time,
        "infer_time_s": infer_time,
        "hardware": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "notes": f"hidden={args.hidden_size}, lr={args.learning_rate}, epochs={args.epochs}, patience={args.patience}, best_epoch={best_epoch}",
        "sequence": "global",
        "window": WINDOW,
    }
    save_predictions(METHOD, task, seed, data["X_test"].index, test_proba.argmax(1), test_score,
                     meta=meta, split_name=args.split)
    print(f"{task} seed={seed} done: best_val_score={best_val:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["detection", "identification"], required=True)
    parser.add_argument("--split", default="episode_block_v1")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定 cuda，但目前 PyTorch 沒有可用的 CUDA")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    print(f"task={args.task}, method={METHOD}, device={device}")
    for seed in args.seeds:
        run(args.task, seed, args, device)


if __name__ == "__main__":
    main()
