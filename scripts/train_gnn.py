"""成員 C：GNN (GAT 邊分類) 訓練，依 docs/EVALUATION_PROTOCOL.md 輸出 val / test 預測。

用法:
    .venv/bin/python scripts/train_gnn.py --task identification
    .venv/bin/python scripts/train_gnn.py --task detection --seeds 0 --window 5
"""
import argparse
import copy
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_split  # noqa: E402
from src.models.gnn_detector import SPLIT_CODE, EdgeGAT, build_window_graphs  # noqa: E402
from src.utils.metrics import detection_metrics, identification_metrics, save_predictions  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model, loader):
    """回傳 (flow_id, split_code, y_true, proba)，涵蓋 loader 內所有 flow。"""
    model.eval()
    ids, splits, ys, probas = [], [], [], []
    for batch in loader:
        probas.append(torch.softmax(model(batch), dim=1))
        ids.append(batch.flow_id)
        splits.append(batch.flow_split)
        ys.append(batch.flow_y)
    return (torch.cat(ids).numpy(), torch.cat(splits).numpy(),
            torch.cat(ys).numpy(), torch.cat(probas).numpy())


def val_score(task, y, proba):
    if task == "detection":
        return detection_metrics(y, proba.argmax(1), proba[:, 1])["f1"]
    return identification_metrics(y, proba.argmax(1))["macro_f1"]


def class_weights(graphs, n_classes, mode):
    if mode == "none":
        return None
    y = torch.cat([g.flow_y[g.flow_split == SPLIT_CODE["train"]] for g in graphs])
    counts = torch.bincount(y, minlength=n_classes).float()
    w = (counts.sum() / (n_classes * counts)) ** (0.5 if mode == "sqrt" else 1.0)
    return w


def train_one(args, graphs, n_classes, seed):
    set_seed(seed)
    g0 = graphs[0]
    model = EdgeGAT(g0.x.size(1), g0.edge_attr.size(1), g0.flow_x.size(1), n_classes,
                    hidden=args.hidden, heads=args.heads, n_layers=args.layers, dropout=args.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(graphs, n_classes, args.class_weight))

    train_graphs = [g for g in graphs if (g.flow_split == SPLIT_CODE["train"]).any()]
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(graphs, batch_size=args.batch_size * 4)

    best, best_state, bad = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            mask = batch.flow_split == SPLIT_CODE["train"]  # loss 只算 train flow
            loss = loss_fn(model(batch)[mask], batch.flow_y[mask])
            opt.zero_grad()
            loss.backward()
            opt.step()

        _, sp, y, proba = predict(model, eval_loader)
        v = SPLIT_CODE["val"]
        score = val_score(args.task, y[sp == v], proba[sp == v])
        if score > best:
            best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if args.verbose:
            print(f"  epoch {epoch:3d} loss={loss.item():.4f} val={score:.4f}")
        if bad >= args.patience:
            break

    model.load_state_dict(best_state)
    return model, eval_loader, best, epoch + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["detection", "identification"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--split", default="episode_block_v1")
    ap.add_argument("--window", type=float, default=10.0, help="每張圖的時間長度 (秒)")
    ap.add_argument("--node-key", choices=["address", "address_merged"], default="address")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32, help="每個 batch 的圖數")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--class-weight", choices=["none", "sqrt", "balanced"], default="none")
    ap.add_argument("--method", default=None, help="預設 gnn_gat_w<window>")
    ap.add_argument("--no-save", action="store_true", help="調參用：只印 val 分數，不輸出預測檔")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    method = args.method or f"gnn_gat_w{args.window:g}" + ("_merged" if args.node_key == "address_merged" else "")
    n_classes = 2 if args.task == "detection" else 6
    data = load_split(task=args.task, split_name=args.split)
    graphs = build_window_graphs(data, window=args.window, node_key=args.node_key)
    print(f"{len(graphs)} graphs, {sum(g.num_nodes for g in graphs) / len(graphs):.1f} nodes/graph avg")

    for seed in args.seeds:
        t0 = time.time()
        model, eval_loader, best, n_epochs = train_one(args, graphs, n_classes, seed)
        train_time = time.time() - t0

        t0 = time.time()
        fid, sp, _, proba = predict(model, eval_loader)
        infer_time = time.time() - t0
        print(f"[{method}] {args.task} seed={seed} val={best:.4f} epochs={n_epochs} ({train_time:.0f}s)")
        if args.no_save:
            continue

        for part in ("val", "test"):
            m = sp == SPLIT_CODE[part]
            score = proba[m, 1] if args.task == "detection" else proba[m]
            meta = None
            if part == "test":
                meta = {
                    "member": "C",
                    **({"detection_from": "separate"} if args.task == "detection" else {}),
                    "n_params": sum(p.numel() for p in model.parameters()),
                    "train_time_s": train_time, "infer_time_s": infer_time, "hardware": "cpu",
                    "graph_window": args.window, "node_key": args.node_key,
                    "notes": (f"hidden={args.hidden}, heads={args.heads}, layers={args.layers}, "
                              f"dropout={args.dropout}, lr={args.lr}, wd={args.weight_decay}, "
                              f"batch={args.batch_size}, class_weight={args.class_weight}, "
                              f"epochs_run={n_epochs}, patience={args.patience}"),
                }
            save_predictions(method, args.task, seed, flow_ids=fid[m], y_pred=proba[m].argmax(1),
                             score=score, meta=meta, split_name=args.split, part=part)


if __name__ == "__main__":
    main()
