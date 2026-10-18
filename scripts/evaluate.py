"""統一評估腳本：讀取 results/predictions/ 下所有方法的預測檔，驗證格式後計算指標並產生排行榜。

用法:
    python scripts/evaluate.py                          # 預設 episode_block_v1
    python scripts/evaluate.py --split random_stratified_v1

輸出:
    results/leaderboard_<split>.md          各方法 mean ± std (跨 seed)，val_score 由 *_val.csv 自動計算
    results/runs_<split>.csv                每個 run 的完整指標
    results/confusion/<split>/<method>_<task>.csv   跨 seed 加總的 confusion matrix
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import CLASSES, get_labels, load_raw, load_split_assignment  # noqa: E402
from src.utils.metrics import PRED_DIR, RESULTS_DIR, detection_metrics, identification_metrics  # noqa: E402

REQUIRED_SEEDS = 5
PRIMARY = {"detection": ["f1", "pr_auc", "fpr", "recall", "precision", "mcc", "val_score"],
           "identification": ["macro_f1", "f1_ip-scan", "f1_mitm", "f1_replay", "balanced_acc", "weighted_f1",
                              "val_score"]}
FILE_RE = re.compile(r"(detection|identification)_seed(\d+)\.csv$")


def validate(pred, task, expected_ids, path, part="test"):
    errors = []
    if pred["flow_id"].duplicated().any():
        errors.append("flow_id 有重複")
    missing = expected_ids.difference(pred["flow_id"])
    extra = set(pred["flow_id"]).difference(expected_ids)
    if missing:
        errors.append(f"缺少 {len(missing)} 筆 {part} flow_id (必須對整個 {part} split 預測)")
    if extra:
        errors.append(f"有 {len(extra)} 筆 flow_id 不屬於 {part} split")
    n_cls = 2 if task == "detection" else len(CLASSES)
    if not pred["y_pred"].between(0, n_cls - 1).all():
        errors.append(f"y_pred 超出範圍 0..{n_cls - 1}")
    if task == "identification":
        prob_cols = [f"prob_{c}" for c in CLASSES]
        present = [c for c in prob_cols if c in pred.columns]
        if present and len(present) != len(prob_cols):
            errors.append(f"機率欄位不完整，需要 {prob_cols}")
    if errors:
        raise ValueError(f"{path.relative_to(ROOT)}:\n  - " + "\n  - ".join(errors))


def score_run(pred, task, y_true):
    y = y_true.loc[pred["flow_id"]].to_numpy()
    if task == "detection":
        score = pred["score"].to_numpy() if "score" in pred.columns else None
        return detection_metrics(y, pred["y_pred"].to_numpy(), score), y
    prob_cols = [f"prob_{c}" for c in CLASSES]
    proba = pred[prob_cols].to_numpy() if prob_cols[0] in pred.columns else None
    return identification_metrics(y, pred["y_pred"].to_numpy(), proba), y


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="episode_block_v1")
    ap.add_argument("--label-strategy", default="NST")
    args = ap.parse_args()

    df = load_raw()
    split = load_split_assignment(args.split).reindex(df.index)
    test_ids = set(df.index[split == "test"])
    val_ids = set(df.index[split == "val"])
    labels = {t: get_labels(df, t, args.label_strategy) for t in PRIMARY}

    rows, confusions, failed, no_val = [], {}, [], []
    for path in sorted(PRED_DIR.glob(f"*/{args.split}/*.csv")):
        m = FILE_RE.search(path.name)
        if not m:
            continue
        task, seed = m.group(1), int(m.group(2))
        method = path.parent.parent.name
        pred = pd.read_csv(path)
        try:
            validate(pred, task, test_ids, path)
        except ValueError as e:
            failed.append(str(e))
            continue
        metrics, y = score_run(pred, task, labels[task])
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        val_score = np.nan
        val_path = path.with_name(f"{task}_seed{seed}_val.csv")
        if val_path.exists():
            val_pred = pd.read_csv(val_path)
            try:
                validate(val_pred, task, val_ids, val_path, part="val")
                val_score = score_run(val_pred, task, labels[task])[0][PRIMARY[task][0]]
            except ValueError as e:
                failed.append(str(e))
        else:
            no_val.append(str(val_path.relative_to(ROOT)))

        rows.append({"method": method, "task": task, "seed": seed, **metrics, "val_score": val_score,
                     **{k: meta.get(k) for k in ("member", "n_params", "train_time_s", "infer_time_s")}})
        cm = confusion_matrix(y, pred["y_pred"], labels=range(2 if task == "detection" else len(CLASSES)))
        confusions[(method, task)] = confusions.get((method, task), 0) + cm

    if failed:
        print("以下預測檔驗證失敗 (test 失敗則該 run 不納入；val 失敗則 val_score 留空):\n" + "\n".join(failed),
              file=sys.stderr)
    if no_val:
        print(f"{len(no_val)} 個 run 缺少 val 預測檔，val_score 留空:\n  " + "\n  ".join(no_val), file=sys.stderr)
    if not rows:
        print(f"找不到有效預測檔: {PRED_DIR.relative_to(ROOT)}/<method>/{args.split}/<task>_seed<k>.csv")
        return 1

    runs = pd.DataFrame(rows)
    runs.to_csv(RESULTS_DIR / f"runs_{args.split}.csv", index=False)

    cm_dir = RESULTS_DIR / "confusion" / args.split
    cm_dir.mkdir(parents=True, exist_ok=True)
    for (method, task), cm in confusions.items():
        names = ["Normal", "Attack"] if task == "detection" else CLASSES
        pd.DataFrame(cm, index=[f"true_{n}" for n in names], columns=[f"pred_{n}" for n in names]) \
            .to_csv(cm_dir / f"{method}_{task}.csv")

    lines = [f"# Leaderboard — split `{args.split}`, labels `{args.label_strategy}`", "",
             "數值為跨 seed 的 mean ± std；`n` 為 seed 數，少於 5 者不列入正式比較。",
             "`val_score` 由 val 預測檔自動計算 (detection: F1，identification: macro-F1)，代表變體須依此選出 (見 docs/EVALUATION_PROTOCOL.md §4)。", ""]
    for task, cols in PRIMARY.items():
        sub = runs[runs.task == task]
        if sub.empty:
            continue
        cols = [c for c in cols if c in sub.columns]
        agg = sub.groupby("method")[cols].agg(["mean", "std"])
        n = sub.groupby("method").size()
        order = agg[(cols[0], "mean")].sort_values(ascending=False).index
        lines += [f"## {task}", "", "| method | n | " + " | ".join(cols) + " |",
                  "|---|---|" + "---|" * len(cols)]
        for meth in order:
            cells = []
            for c in cols:
                mu, sd = agg.loc[meth, (c, "mean")], agg.loc[meth, (c, "std")]
                cells.append("—" if np.isnan(mu) else f"{mu:.4f} ± {0 if np.isnan(sd) else sd:.4f}")
            flag = "" if n[meth] >= REQUIRED_SEEDS else " ⚠"
            lines.append(f"| {meth} | {n[meth]}{flag} | " + " | ".join(cells) + " |")
        lines.append("")

    out = RESULTS_DIR / f"leaderboard_{args.split}.md"
    # Explicit UTF-8 keeps the warning symbol (⚠) portable on Windows
    # consoles whose default encoding is commonly cp950.
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {out.relative_to(ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
