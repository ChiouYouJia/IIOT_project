"""DT / ANN 參考基線，超參數依原論文 Table 11 (Dehlaghi-Ghadim et al., 2023)。

原論文用 MATLAB，對應到 sklearn 的方式：
- DT「Maximum number of splits」N → max_leaf_nodes = N + 1。
  binary 的 split criterion「Maximum deviance reduction」→ criterion="entropy"；
  multi-class 的「Twoing rule」sklearn 沒有，改用 criterion="gini"。
- ANN「Fully connected layers: 1, Activation: Sigmoid, First layer size」→
  MLPClassifier(hidden_layer_sizes=(size,), activation="logistic")，
  solver 用 L-BFGS、最多 1000 次迭代 (與 MATLAB fitcnet 預設相同)。
- 原論文另用 MRMR 選出 23 個特徵；這裡與 rf_baseline 一致，使用 load_split() 的全部特徵。

用法:
    python scripts/baseline_dt_ann.py                              # dt + ann, 5 seeds x 2 tasks
    python scripts/baseline_dt_ann.py --model dt --split random_stratified_v1
"""
import argparse
import sys
import time
from pathlib import Path

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_split  # noqa: E402
from src.utils.metrics import save_predictions  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]

PAPER_PARAMS = {
    ("dt", "detection"): dict(criterion="entropy", max_leaf_nodes=315),
    ("dt", "identification"): dict(criterion="gini", max_leaf_nodes=1001),
    ("ann", "detection"): dict(hidden_layer_sizes=(79,), activation="logistic", solver="lbfgs", max_iter=1000),
    ("ann", "identification"): dict(hidden_layer_sizes=(257,), activation="logistic", solver="lbfgs", max_iter=1000),
}


def build(model, task, seed):
    params = PAPER_PARAMS[(model, task)]
    if model == "dt":
        return DecisionTreeClassifier(random_state=seed, **params), params
    return MLPClassifier(random_state=seed, **params), params


def n_params(clf):
    if isinstance(clf, DecisionTreeClassifier):
        return clf.tree_.node_count
    return sum(w.size for w in clf.coefs_) + sum(b.size for b in clf.intercepts_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["dt", "ann"], nargs="+", default=["dt", "ann"])
    ap.add_argument("--split", default="episode_block_v1")
    args = ap.parse_args()

    for task in ("detection", "identification"):
        data = load_split(task=task, split_name=args.split)
        for model in args.model:
            method = f"{model}_baseline"
            for seed in SEEDS:
                clf, params = build(model, task, seed)
                t0 = time.time()
                clf.fit(data["X_train"], data["y_train"])
                train_time = time.time() - t0

                val_proba = clf.predict_proba(data["X_val"])
                save_predictions(method, task, seed, flow_ids=data["X_val"].index,
                                 y_pred=val_proba.argmax(1),
                                 score=val_proba[:, 1] if task == "detection" else val_proba,
                                 split_name=args.split, part="val")

                t0 = time.time()
                proba = clf.predict_proba(data["X_test"])
                infer_time = time.time() - t0

                save_predictions(
                    method, task, seed,
                    flow_ids=data["X_test"].index,
                    y_pred=proba.argmax(1),
                    score=proba[:, 1] if task == "detection" else proba,
                    meta={"member": "shared",
                          **({"detection_from": "separate"} if task == "detection" else {}),
                          "n_params": int(n_params(clf)),
                          "train_time_s": train_time, "infer_time_s": infer_time,
                          "hardware": "cpu",
                          "notes": f"paper Table 11: {params}, all load_split() features (no MRMR)"},
                    split_name=args.split,
                )
                print(f"{method} {task} seed={seed} done ({train_time:.1f}s)")


if __name__ == "__main__":
    main()
