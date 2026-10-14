# 成員使用說明

這份說明告訴你 **怎麼用共用的資料 / 評估工具跑出可以比較的結果**。
規則的完整內容與理由在 [`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md)，有疑問以那份為準。

---

## 0. 三分鐘重點

1. 資料一律用 `load_split()` 載入，**不要自己切 train / test**。
2. 用 train 訓練，用 val 調參；test 只在最後輸出預測時碰一次。
3. 每個設定跑 **seed 0–4**，**detection 與 identification 都要做**。
4. 每個 run 用 `save_predictions()` 存 **兩份**：`part="val"` 和 test。
5. commit 自己的 `results/predictions/<method>/`，執行 `python scripts/evaluate.py` 看排行榜。

---

## 1. 開始前

```bash
git pull
conda activate ics-anomaly          # 或 pip install -r setup/requirements.txt
```

- `data/raw/Dataset.csv` 與切分檔 `data/splits/*.csv` 都已在 repo 裡，**不需要另外下載**。
- 所有指令都在 **專案根目錄** 執行。
- 在 `notebooks/` 裡的 notebook 要先加這兩行才能 import：

  ```python
  import sys; sys.path.insert(0, "..")
  ```

### 先跑一次基線，確認環境正常

```bash
python scripts/baseline_rf.py      # 約 10 秒
python scripts/evaluate.py
```

看到下面的數字就代表環境沒問題 (這也是大家要超越的基準)：

| 任務 | 指標 | RF 基線 |
|------|------|------:|
| detection | F1 | 0.9803 |
| identification | macro-F1 | 0.8801 |
| identification | ip-scan F1 | 0.5308 |

---

## 2. 載入資料

```python
from src.data.loader import load_split, sort_by_time, CLASSES

data = load_split(task="detection")        # 或 task="identification"
```

| key | 內容 |
|-----|------|
| `X_train`, `X_val`, `X_test` | 已標準化的特徵 (DataFrame，index = `flow_id`，52 欄) |
| `y_train`, `y_val`, `y_test` | 標籤 (Series)。detection: 0/1；identification: 0–5 |
| `X_all`, `y_all`, `split` | 全部 45,718 筆 (給 LSTM / GNN 建窗口或圖用) |
| `df` | 原始資料 (含位址、時間欄位，**只能用來排序 / 建圖，不能當特徵**) |
| `feature_names` | 特徵欄位名稱 |

- identification 的類別順序固定為 `CLASSES = ["Normal", "ddos", "ip-scan", "mitm", "port-scan", "replay"]`，輸出層第 i 個節點就是 `CLASSES[i]`。
- 特徵已經排除 IP / MAC / 時間戳記，標準化也只用 train 的統計量，**直接用就好，不要自己重做**。

---

## 3. 標準訓練流程 (所有方法共用骨架)

```python
import random, time
import numpy as np
import torch
from src.data.loader import load_split
from src.utils.metrics import detection_metrics, identification_metrics, save_predictions

METHOD = "lstm_attn_global_w20"     # 小寫加底線；不同變體用不同名稱
TASK = "identification"             # 兩個任務都要跑


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


data = load_split(task=TASK)

for seed in [0, 1, 2, 3, 4]:
    set_seed(seed)
    model = build_model()                                  # 你的模型

    t0 = time.time()
    train(model, data, early_stop_on="val")                # early stopping 只看 val
    train_time = time.time() - t0

    # --- val 預測 (evaluate.py 用它自動算 val_score) ---
    val_proba = predict_proba(model, data["X_val"])         # (11534, 6)；detection 則為 (11534,) 的 P(attack)
    save_predictions(METHOD, TASK, seed, flow_ids=data["X_val"].index,
                     y_pred=val_proba.argmax(1), score=val_proba, part="val")

    # --- test 預測 + meta ---
    t0 = time.time()
    proba = predict_proba(model, data["X_test"])            # (12588, 6)
    infer_time = time.time() - t0
    save_predictions(METHOD, TASK, seed, flow_ids=data["X_test"].index,
                     y_pred=proba.argmax(1), score=proba,
                     meta={"member": "A",
                           "n_params": sum(p.numel() for p in model.parameters()),
                           "train_time_s": train_time, "infer_time_s": infer_time,
                           "hardware": "RTX 3060",
                           "notes": "hidden=128, lr=1e-3, epochs=40",
                           "sequence": "global", "window": 20})   # 方法專屬欄位見第 5 節
```

### detection 任務要注意

- `score` 傳 **一維** 的 P(attack)，`y_pred` 傳 0/1。
- `meta` 要多一個 `"detection_from"`：
  - `"separate"`：另外訓練二元模型
  - `"identification"`：直接用多類別模型，`score = 1 - proba[:, 0]`，`y_pred = (score >= 0.5)`
- threshold 若不是 0.5，必須是在 val 上決定的，並寫進 `meta["threshold"]`。

### 調參時怎麼看分數

用和 `evaluate.py` **同一個函式** 算 val，大家的分數定義就會一致：

```python
detection_metrics(data["y_val"], y_pred_val, score_val)["f1"]
identification_metrics(data["y_val"], y_pred_val)["macro_f1"]
identification_metrics(data["y_val"], y_pred_val)["f1_ip-scan"]
```

不要用 sklearn 的 `f1_score(average="weighted")` 或 accuracy 來調參 — Normal 佔 80%，會看不出小類別的好壞。

---

## 4. 預測檔長什麼樣子

`save_predictions()` 會自動產生，你不需要手寫：

```
results/predictions/lstm_attn_global_w20/episode_block_v1/
├── identification_seed0.csv        # test 預測
├── identification_seed0.json       # meta
├── identification_seed0_val.csv    # val 預測
├── ...                             # seed 1–4
└── detection_seed0.csv ...         # detection 同樣一組
```

一個方法完整提交 = 2 任務 × 5 seeds × 3 檔 = **30 個檔案**。

檔案內 **沒有 y_true**，真實標籤由 `evaluate.py` 自己取，所以不會有標籤對錯的問題。

---

## 5. 各方法專屬說明

所有方法最後都要對 **每一筆** val / test flow 給預測，不可以丟掉任何一筆。

### 成員 A — LSTM

**建立滑動窗口** (可直接使用)：

```python
import numpy as np, pandas as pd
from src.data.loader import load_split, sort_by_time

data = load_split(task="identification")
W = 20

order = sort_by_time(data["df"]).index.to_numpy()           # 依時間排好的 flow_id
X = data["X_all"].loc[order].to_numpy(np.float32)
pos = pd.Series(np.arange(len(order)), index=order)         # flow_id -> 時間序位置
Xpad = np.vstack([np.zeros((W - 1, X.shape[1]), np.float32), X])   # 開頭補 0

def windows(flow_ids):
    """回傳 (n, W, 52)，每個窗口最後一筆就是目標 flow，只包含過去的 flow。"""
    idx = pos.loc[flow_ids].to_numpy()
    return np.stack([Xpad[i:i + W] for i in idx])

Xw_train = windows(data["X_train"].index);  y_train = data["y_train"].to_numpy()
Xw_val   = windows(data["X_val"].index);    y_val   = data["y_val"].to_numpy()
Xw_test  = windows(data["X_test"].index)
# 存檔時 flow_ids 用 data["X_val"].index / data["X_test"].index，順序與窗口一致
```

- 窗口裡的前幾筆可能屬於 val / test，這是允許的 (只用特徵，沒用標籤)。
- **一定要用 `sort_by_time()`**，不要自己 `sort_values("start")` — 有一組時間完全相同，順序會不一致。
- 若改用 `per_pair` (依 `sAddress`, `rAddress` 分組各自排序)，meta 的 `sequence` 填 `"per_pair"`。
- meta 必填：`sequence`、`window`。

### 成員 B — Federated Learning

**主要實驗用逐筆 MLP**，輸入就是 `X_train`，跟 RF / ANN 基線一樣。第 4 週底看 A 的 LSTM 是否明顯勝過 ANN，再決定要不要做 FL-LSTM。

**建立 client 切分** (只做一次，存檔後 commit，之後 5 個 seed 都用同一份)：

```python
import numpy as np, pandas as pd
from src.data.loader import load_raw

split = pd.read_csv("data/splits/episode_block_v1.csv")
train = split[split["split"] == "train"].copy()               # 只能從 train 切
train["label"] = load_raw().loc[train["flow_id"], "NST_M_Label"].to_numpy()

# 每個 block 的攻擊類型 (同一次攻擊事件不拆開)
btype = train.groupby("block_id")["label"].agg(
    lambda s: s[s != "Normal"].mode().iat[0] if (s != "Normal").any() else "Normal")
# 目前 train 有：Normal 30 塊、ip-scan 12、port-scan 9、mitm 8、replay 4、ddos 4

# 範例：IID —— 隨機把 block 平均分給 4 個 client
rng = np.random.default_rng(42)
client_of = {b: i % 4 for i, b in enumerate(rng.permutation(btype.index))}
train["client"] = train["block_id"].map(client_of)
train[["flow_id", "client"]].to_csv("data/splits/fl_iid_v1.csv", index=False)

# Non-IID：依 btype 決定哪些攻擊類型給哪個 client，存成 fl_non_iid_attack_v1.csv
```

- 各 client 的 val 選 round / 調參用全域 `X_val`；全域模型對 **同一個** `X_test` 預測。
- method 名稱寫出架構：`fl_mlp_fedavg_iid`、`fl_mlp_fedprox_noniid`、`local_mlp_client0`、`fl_lstm_fedprox_noniid`。
- FL 一律和 **同架構** 的 centralized 版本比：FL-MLP ↔ `ann_baseline`，FL-LSTM ↔ A 的 LSTM。
- meta 必填：`partition`、`num_clients`、`rounds`、`local_epochs`、`strategy`、`local_model`。

### 成員 C — GNN

**建立節點與邊** (可直接使用)：

```python
import numpy as np, pandas as pd
from src.data.loader import load_split, sort_by_time

data = load_split(task="identification")
df = sort_by_time(data["df"])

nodes = pd.Index(pd.unique(df[["sAddress", "rAddress"]].to_numpy().ravel()))   # 19 個節點
src = nodes.get_indexer(df["sAddress"])
dst = nodes.get_indexer(df["rAddress"])
edge_x = data["X_all"].loc[df.index].to_numpy(np.float32)    # 邊特徵 = flow 特徵
edge_y = data["y_all"].loc[df.index].to_numpy()
edge_split = data["split"].loc[df.index].to_numpy()         # 用來做 loss mask
graph_id = ((df["start"] - df["start"].min()) // 10).astype(int).to_numpy()   # 10 秒一張圖
```

- **用 `sAddress` / `rAddress`，不要用 `sIPs` / `rIPs`**：ARP 流量沒有 IP，全部 192 筆 ip-scan 都是 ARP，用 IP 建圖會整批消失。
- 節點特徵 **不可** 用位址 one-hot 或 learnable node embedding (攻擊者位址固定，會變成作弊)。可以用度數、流量統計這類結構特徵。
- 同一張圖裡會混有 train / val / test 的邊，**loss 只算 `edge_split == "train"` 的邊**。
- 輸出要是 **每條邊 (flow)** 的預測。
- meta 必填：`graph_window`、`node_key` (`"address"` 或 `"address_merged"`)。

---

## 6. 提交與查看結果

```bash
python scripts/evaluate.py                  # 確認沒有驗證錯誤
git add results/predictions/<你的method>/ src/ notebooks/
git commit -m "Add <method> predictions"
git pull --rebase && git push
```

- **只 commit 自己的 method 目錄**，不要改別人的。
- 排行榜 (`results/leaderboard_*.md`) 和 confusion matrix 不 commit，每個人自己產生。
- 想看別人的結果：`git pull` 後跑一次 `python scripts/evaluate.py`。

`evaluate.py` 的輸出：

| 檔案 | 內容 |
|------|------|
| `results/leaderboard_episode_block_v1.md` | 各方法 test 指標 mean ± std，加上 val_score |
| `results/runs_episode_block_v1.csv` | 每個 run 的完整指標 (含所有類別的 F1) |
| `results/confusion/episode_block_v1/<method>_<task>.csv` | 5 seeds 加總的 confusion matrix |

---

## 7. 常見錯誤

| 訊息 / 狀況 | 原因與解法 |
|------|------|
| `缺少 N 筆 test flow_id` | 有 flow 沒預測到。LSTM 檢查開頭是否有補 padding；GNN 檢查是否漏了 ARP 邊 |
| `缺少 N 筆 val flow_id` | 同上，val 也要完整 |
| `有 N 筆 flow_id 不屬於 test split` | 存檔時 `flow_ids` 傳錯 (例如傳成 `X_val.index` 或 0..n 的流水號) |
| `y_pred 超出範圍` | identification 要用 0–5，不要傳類別名稱字串；detection 只能 0/1 |
| `機率欄位不完整` | identification 的 `score` 要傳 (n, 6) 的完整機率矩陣 |
| `缺少 val 預測檔，val_score 留空` | 忘了 `part="val"` 那次呼叫，這個 run 不能當代表變體 |
| 排行榜 `n` 旁邊有 ⚠ | seed 不足 5 個 |
| val 很高、test 差很多 | 正常範圍內的差距不必擔心；若差距很大，檢查是否不小心用到 val/test 的資訊做 fit |

---

## 8. 第 7 週前要做的事

1. 每個人從自己的變體中，依 **排行榜上的 val_score** 選出代表變體 (不是看 test)。
2. 把 method 名稱填進 `EVALUATION_PROTOCOL.md` §4 的表格並 commit。
3. 提交前對照 `EVALUATION_PROTOCOL.md` §9 的檢查清單。

有任何規則不清楚的地方，先在群組提出，確定後再更新到 `EVALUATION_PROTOCOL.md`，不要各自假設。
