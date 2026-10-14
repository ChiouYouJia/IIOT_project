"""產生全組共用的 train / val / test 切分檔 (只需執行一次，產出檔案需 commit)。

切分方式 (episode-block split)：
  - 以 attacker_machine_summary.csv 的 74 次攻擊事件 (episode) 為單位切出時間區塊，
    每個區塊 = 一次攻擊事件 + 其前後的正常流量 (邊界取相鄰事件間隔的中點)。
  - 第一次攻擊前的純正常時段，切成每 60 秒一個區塊。
  - 依攻擊類型分層，將「整個區塊」分到 train / val / test (50 / 20 / 30)。
  同一次攻擊事件的所有 flow 只會出現在同一個 split，避免時序模型 / 圖模型
  透過相鄰 flow 偷看到測試資料。

另外輸出一份逐筆隨機分層切分 (random_stratified)，僅供與原論文基線對照使用。

用法:
    python scripts/make_splits.py
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
SPLIT_DIR = ROOT / "data" / "splits"

SEED = 42
RATIOS = {"train": 0.5, "val": 0.2, "test": 0.3}
NORMAL_BLOCK_SECONDS = 60


def assign_blocks(df, episodes):
    """回傳每筆 flow 的 block_id 與該 block 的攻擊類型 ("Normal" 表示純正常區塊)。"""
    episodes = episodes.sort_values("startStamp").reset_index(drop=True)
    starts = episodes["startStamp"].to_numpy()
    ends = episodes["endStamp"].to_numpy()

    # 第 k 個 episode 區塊的左邊界 = 前一個 episode 結束與本 episode 開始的中點
    left = np.empty(len(episodes))
    left[1:] = (ends[:-1] + starts[1:]) / 2
    left[0] = starts[0] - (starts[1] - ends[0]) / 2

    t = df["start"].to_numpy()
    block_id = np.empty(len(df), dtype=object)
    block_type = np.empty(len(df), dtype=object)

    pre = t < left[0]
    k = ((t[pre] - t.min()) // NORMAL_BLOCK_SECONDS).astype(int)
    block_id[pre] = [f"N{i:03d}" for i in k]
    block_type[pre] = "Normal"

    ep_idx = np.searchsorted(left, t[~pre], side="right") - 1
    block_id[~pre] = [f"E{i:02d}" for i in ep_idx]
    block_type[~pre] = episodes["attack"].to_numpy()[ep_idx]
    return block_id, block_type


def episode_block_split(df, episodes):
    block_id, block_type = assign_blocks(df, episodes)
    blocks = pd.DataFrame({"block_id": block_id, "block_type": block_type}).drop_duplicates("block_id")

    rng = np.random.default_rng(SEED)
    block_split = {}
    for _, group in blocks.groupby("block_type"):
        ids = rng.permutation(group["block_id"].to_numpy())
        n = len(ids)
        n_train = int(round(n * RATIOS["train"]))
        n_val = max(1, int(round(n * RATIOS["val"])))
        for i, b in enumerate(ids):
            block_split[b] = "train" if i < n_train else "val" if i < n_train + n_val else "test"

    return pd.DataFrame({
        "flow_id": df.index,
        "split": [block_split[b] for b in block_id],
        "block_id": block_id,
    })


def random_stratified_split(df):
    ids = df.index.to_numpy()
    y = df["NST_M_Label"].to_numpy()
    temp, test = train_test_split(ids, test_size=RATIOS["test"], random_state=SEED, stratify=y)
    rel_val = RATIOS["val"] / (1 - RATIOS["test"])
    train, val = train_test_split(temp, test_size=rel_val, random_state=SEED, stratify=y[temp])
    split = pd.Series("train", index=ids)
    split[val] = "val"
    split[test] = "test"
    return pd.DataFrame({"flow_id": ids, "split": split.to_numpy(), "block_id": ""})


def summarize(name, split_df, df):
    table = pd.crosstab(df.loc[split_df["flow_id"], "NST_M_Label"].to_numpy(), split_df["split"].to_numpy())
    print(f"\n[{name}] NST_M_Label x split")
    print(table[["train", "val", "test"]])
    assert (table[["train", "val", "test"]] > 0).all().all(), "每個 split 都必須包含所有類別"


def main():
    df = pd.read_csv(RAW_DIR / "Dataset.csv")
    episodes = pd.read_csv(RAW_DIR / "attacker_machine_summary.csv")
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    for name, split_df in [
        ("episode_block_v1", episode_block_split(df, episodes)),
        ("random_stratified_v1", random_stratified_split(df)),
    ]:
        summarize(name, split_df, df)
        path = SPLIT_DIR / f"{name}.csv"
        split_df.to_csv(path, index=False)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        print(f"-> {path.relative_to(ROOT)}  sha256[:16]={digest}")


if __name__ == "__main__":
    main()
