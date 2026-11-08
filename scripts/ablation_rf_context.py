"""成員 C 消融：RF + 窗口上下文特徵 (不用 GNN)。

問題：GNN 的提升是來自 message passing，還是只來自「同一時間窗口內、同一主機的其他 flow」這份上下文？
作法：用與 GNN 完全相同的建圖 (build_window_graphs)，把每筆 flow 的特徵接上
      來源 / 目的節點的聚合特徵 (進 / 出邊 flow 特徵平均 + log 度數)，再訓練 RF。
      位址只用來決定哪些 flow 聚合在一起，不進入特徵 (與 GNN 規則相同)。

用法:
    .venv/bin/python scripts/ablation_rf_context.py --task identification --window 10 --node-key address_merged
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_split, sort_by_time  # noqa: E402
from src.models.gnn_detector import SPLIT_CODE, build_window_graphs, mac_to_ip_map  # noqa: E402
from src.utils.metrics import detection_metrics, identification_metrics, save_predictions  # noqa: E402


def context_features(graphs):
    """每筆 flow → [flow 特徵, 來源節點特徵, 目的節點特徵]。"""
    X, y, sp, fid = [], [], [], []
    for g in graphs:
        s, d = g.flow_index
        X.append(np.hstack([g.flow_x.numpy(), g.x[s].numpy(), g.x[d].numpy()]))
        y.append(g.flow_y.numpy())
        sp.append(g.flow_split.numpy())
        fid.append(g.flow_id.numpy())
    return np.vstack(X), np.concatenate(y), np.concatenate(sp), np.concatenate(fid)


def window_global_features(graphs):
    """每筆 flow → [flow 特徵, 整個窗口所有 flow 特徵平均, log flow 數]。不分主機，無法辨認特定主機。"""
    X, y, sp, fid = [], [], [], []
    for g in graphs:
        m = g.flow_x.size(0)
        ctx = np.hstack([g.flow_x.numpy().mean(0), [np.log1p(m)]])
        X.append(np.hstack([g.flow_x.numpy(), np.tile(ctx, (m, 1))]))
        y.append(g.flow_y.numpy())
        sp.append(g.flow_split.numpy())
        fid.append(g.flow_id.numpy())
    return np.vstack(X), np.concatenate(y), np.concatenate(sp), np.concatenate(fid)


def _past_mean(t, X, key_rows, key_query, W):
    """對每筆 query flow i，聚合 key_rows == key_query[i] 且 start ∈ [t_i - W, t_i] 的 flow 特徵 (平均, log 數量)。

    key 為 None 時代表不分主機 (全部 flow)。t 已依時間排序。
    """
    n, f = X.shape
    mean, cnt = np.zeros((n, f), np.float32), np.zeros(n, np.float32)
    rows_by_key = {None: np.arange(n)} if key_rows is None else \
        {k: np.flatnonzero(key_rows == k) for k in np.unique(key_rows)}
    q_by_key = {None: np.arange(n)} if key_query is None else \
        {k: np.flatnonzero(key_query == k) for k in np.unique(key_query)}
    for k, q in q_by_key.items():
        r = rows_by_key.get(k)
        if r is None or len(r) == 0:
            continue
        tr = t[r]
        csum = np.vstack([np.zeros((1, f)), np.cumsum(X[r], 0, dtype=np.float64)])
        lo = np.searchsorted(tr, t[q] - W, "left")
        hi = np.searchsorted(tr, t[q], "right")
        c = hi - lo
        mean[q] = (csum[hi] - csum[lo]) / np.maximum(c, 1)[:, None]
        cnt[q] = c
    return mean, np.log1p(cnt)[:, None]


def causal_features(data, W, node_key, scope):
    """只看過去：每筆 flow 只聚合 start 在過去 W 秒內 (含自己) 的 flow，與 LSTM 的因果規則一致。

    scope="node"：同主機 (來源 / 目的節點各取 出邊平均、入邊平均、log 出 / 入數量，與 GNN 節點特徵同構)。
    scope="global"：不分主機，全網路過去 W 秒。
    """
    df = sort_by_time(data["df"])
    fid = df.index.to_numpy()
    X = data["X_all"].loc[fid].to_numpy(np.float32)
    y = data["y_all"].loc[fid].to_numpy(np.int64)
    sp = data["split"].loc[fid].map(SPLIT_CODE).to_numpy(np.int64)
    t = df["start"].to_numpy(np.float64)
    if scope == "global":
        m, c = _past_mean(t, X, None, None, W)
        return np.hstack([X, m, c]), y, sp, fid
    src, dst = df["sAddress"].to_numpy(), df["rAddress"].to_numpy()
    if node_key == "address_merged":
        m2i = mac_to_ip_map(data)
        src = np.array([m2i.get(a, a) for a in src])
        dst = np.array([m2i.get(a, a) for a in dst])
    parts = [X]
    for host in (src, dst):
        out_m, out_c = _past_mean(t, X, src, host, W)
        in_m, in_c = _past_mean(t, X, dst, host, W)
        parts += [out_m, in_m, out_c, in_c]
    return np.hstack(parts), y, sp, fid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["detection", "identification"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--split", default="episode_block_v1")
    ap.add_argument("--window", type=float, default=10.0)
    ap.add_argument("--node-key", choices=["address", "address_merged"], default="address")
    ap.add_argument("--context", choices=["node", "global", "causal_node", "causal_global"], default="node",
                    help="node: 窗口內同主機聚合 (與 GNN 同)；global: 窗口內全部 flow 聚合；"
                         "causal_*: 只看過去 window 秒")
    ap.add_argument("--method", default=None)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    prefix = {"node": "rf_ctx", "global": "rf_gctx", "causal_node": "rf_cctx", "causal_global": "rf_cgctx"}
    merged = args.node_key == "address_merged" and args.context in ("node", "causal_node")
    method = args.method or f"{prefix[args.context]}_w{args.window:g}" + ("_merged" if merged else "")
    data = load_split(task=args.task, split_name=args.split)
    if args.context.startswith("causal"):
        X, y, sp, fid = causal_features(data, args.window, args.node_key, args.context.split("_")[1])
    else:
        graphs = build_window_graphs(data, window=args.window, node_key=args.node_key)
        X, y, sp, fid = (context_features if args.context == "node" else window_global_features)(graphs)
    tr, va, te = (sp == SPLIT_CODE[k] for k in ("train", "val", "test"))

    for seed in args.seeds:
        clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
        t0 = time.time()
        clf.fit(X[tr], y[tr])
        train_time = time.time() - t0

        pv = clf.predict_proba(X[va])
        if args.task == "detection":
            v = detection_metrics(y[va], pv.argmax(1), pv[:, 1])["f1"]
        else:
            v = identification_metrics(y[va], pv.argmax(1))["macro_f1"]
        print(f"[{method}] {args.task} seed={seed} val={v:.4f} ({train_time:.1f}s)")
        if args.no_save:
            continue

        t0 = time.time()
        pt = clf.predict_proba(X[te])
        infer_time = time.time() - t0
        for part, m, p in (("val", va, pv), ("test", te, pt)):
            meta = None
            if part == "test":
                meta = {"member": "C",
                        **({"detection_from": "separate"} if args.task == "detection" else {}),
                        "n_params": sum(e.tree_.node_count for e in clf.estimators_),
                        "train_time_s": train_time, "infer_time_s": infer_time, "hardware": "cpu",
                        "graph_window": args.window, "node_key": args.node_key,
                        "context": args.context,
                        "notes": f"ablation: RF (n_estimators=100), context={args.context}, "
                                 "flow features + aggregated context features; no message passing"}
            save_predictions(method, args.task, seed, flow_ids=fid[m], y_pred=p.argmax(1),
                             score=p[:, 1] if args.task == "detection" else p,
                             meta=meta, split_name=args.split, part=part)


if __name__ == "__main__":
    main()
