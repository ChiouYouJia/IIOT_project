"""Random Forest 參考基線，同時作為「如何產出標準預測檔」的範本。

用法:
    python scripts/baseline_rf.py                              # 5 seeds x 2 tasks
    python scripts/baseline_rf.py --split random_stratified_v1
"""
import argparse
import sys
import time
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_split  # noqa: E402
from src.utils.metrics import save_predictions  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="episode_block_v1")
    args = ap.parse_args()

    for task in ("detection", "identification"):
        data = load_split(task=task, split_name=args.split)
        for seed in SEEDS:
            clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
            t0 = time.time()
            clf.fit(data["X_train"], data["y_train"])
            train_time = time.time() - t0

            t0 = time.time()
            proba = clf.predict_proba(data["X_test"])
            infer_time = time.time() - t0

            save_predictions(
                "rf_baseline", task, seed,
                flow_ids=data["X_test"].index,
                y_pred=proba.argmax(1),
                score=proba[:, 1] if task == "detection" else proba,
                meta={"member": "shared", "train_time_s": train_time, "infer_time_s": infer_time,
                      "n_params": sum(e.tree_.node_count for e in clf.estimators_)},
                split_name=args.split,
            )
            print(f"{task} seed={seed} done ({train_time:.1f}s)")


if __name__ == "__main__":
    main()
