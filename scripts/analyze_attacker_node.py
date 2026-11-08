"""成員 C 分析：攻擊者節點捷徑 (shortcut) 檢查。

ICS-Flow 中攻擊者固定為 192.168.0.41 / 02:42:c0:a8:00:29，且整個資料集只有 9 筆 Normal flow 涉及它。
窗口上下文 / 圖模型可能藉由「攻擊者節點的行為輪廓」間接辨認主機。此腳本把 test flow 分成
「涉及攻擊者位址」與「不涉及」兩群，分別計算各方法的 per-class recall (5 seeds 平均)。

用法: .venv/bin/python scripts/analyze_attacker_node.py [--part test|val] method1 method2 ...
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data.loader import CLASSES, load_split  # noqa: E402

ATTACKER = {"192.168.0.41", "02:42:c0:a8:00:29"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("methods", nargs="+")
    ap.add_argument("--part", default="test")
    ap.add_argument("--split", default="episode_block_v1")
    args = ap.parse_args()

    d = load_split(task="identification", split_name=args.split)
    df, y = d["df"], d["y_all"]
    involves = df["sAddress"].isin(ATTACKER) | df["rAddress"].isin(ATTACKER)
    suffix = "_val" if args.part == "val" else ""
    rows = []
    for m in args.methods:
        for seed in range(5):
            p = pd.read_csv(ROOT / f"results/predictions/{m}/{args.split}/identification_seed{seed}{suffix}.csv",
                            index_col="flow_id")
            t = pd.DataFrame({"y": y.loc[p.index], "pred": p["y_pred"], "inv": involves.loc[p.index]})
            for (inv, c), g in t.groupby(["inv", "y"]):
                rows.append({"method": m, "attacker_node": "yes" if inv else "no", "class": CLASSES[c],
                             "n": len(g), "recall": (g["pred"] == g["y"]).mean(),
                             "det_recall": ((g["pred"] != 0) == (g["y"] != 0)).mean()})
    r = pd.DataFrame(rows).groupby(["attacker_node", "class", "method"], sort=False).mean(numeric_only=True)
    r["n"] = r["n"].astype(int)
    with pd.option_context("display.width", 200, "display.max_rows", 200, "display.max_columns", 50):
        out = r["recall"].unstack("method")[args.methods].round(3)
        out.insert(0, "n", r["n"].groupby(level=[0, 1], sort=False).first())
        print(out)


if __name__ == "__main__":
    main()
