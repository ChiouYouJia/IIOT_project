# 評估規範 (Evaluation Protocol) v1.1

三位成員 (LSTM / FL / GNN) 的模型必須依照本規範產出結果，才能在第 7 週做公平的橫向比較。
**原則：資料切分、標籤、指標計算全部共用；各自只負責模型本身。**

> 操作步驟與範例程式碼見 [`USAGE_GUIDE.md`](USAGE_GUIDE.md)。

---

## 1. 一句話流程

```
src.data.loader.load_split()  →  訓練 (只用 train，調參只看 val)
  →  對 val 與 test split 每一筆 flow 輸出預測  →  src.utils.metrics.save_predictions(part="val" / "test")
  →  python scripts/evaluate.py  →  results/leaderboard_episode_block_v1.md
```

### 誰來跑？

**每個成員自己跑，不需要由一個人統一調。** 規範統一的是「怎麼算分」，不是「誰來算」。

| 階段 | 誰 | 用什麼 | 資料 |
|------|----|-------|------|
| 調參、early stopping、挑變體 | 各成員自己，隨時 | `detection_metrics()` / `identification_metrics()` (與 `evaluate.py` 內部是同一份程式碼) | **val** |
| 正式評估 | 任何人，結果都一樣 | `python scripts/evaluate.py` (讀取已 commit 的預測檔) | val + test |

正式報告中的數字 **一律取自 `scripts/evaluate.py` 的輸出**，不採用各自 notebook 算的數字。

**比較基準是我們自己在 `episode_block_v1` 上跑的 `rf_baseline`** (detection F1 0.980、identification macro-F1 0.880、ip-scan F1 0.531)，
不是原論文的數字 (原論文是逐筆隨機切分，條件不同)。

---

## 2. 資料與標籤

| 項目 | 規定 |
|------|------|
| 資料 | `data/raw/Dataset.csv` (45,718 筆 flow)，`flow_id` = 原始列索引 (0-based) |
| 標籤策略 | **NST** (`NST_B_Label` / `NST_M_Label`)。IT 標籤會把攻擊期間的正常流量也標成攻擊，不作為主要結果 |
| 任務 | **detection 與 identification 兩個都必做**，見下表 |
| PLC snapshot 資料 | 本版不使用；若要用，需另外說明並單獨列為額外實驗 |

| 任務 | 類別 |
|------|------|
| Detection (二元) | 0 = Normal, 1 = Attack |
| Identification (多類別) | 固定編碼 `0 Normal, 1 ddos, 2 ip-scan, 3 mitm, 4 port-scan, 5 replay` (`loader.CLASSES`) |

### 資料特性 (建模前必讀)

- **ARP 流量沒有 IP**：1,552 筆 ARP flow 的 `sIPs` / `rIPs` 是空值，其中包含 **全部 192 筆 ip-scan**，以及 mitm 526 筆、port-scan 359 筆、replay 266 筆。
- `sAddress` / `rAddress` 沒有空值：有 IP 的流量填 IP，ARP 流量填 MAC (`rAddress` 可能是廣播位址 `ff:ff:ff:ff:ff:ff`)。
- 有 1 組 flow 的 `start` 完全相同。需要依時間排序時一律使用 `loader.sort_by_time(df)` (以 `start`、`flow_id` 排序)，確保大家的順序一致。

### 禁用特徵 (避免捷徑學習)

攻擊者固定是 `192.168.0.41 / 02:42:c0:a8:00:29`，模型只要記住位址就能「偵測」攻擊。因此：

- `sAddress, rAddress, sMACs, rMACs, sIPs, rIPs` **不可作為輸入特徵**。
- `startDate, endDate, start, end, startOffset, endOffset` 不可作為特徵 (可用於排序 / 建窗口 / 建圖)。
- **GNN 例外**：可以用位址決定圖的節點與連邊 (拓撲)，但 **不可** 把位址 one-hot、learnable node-ID embedding 或任何可識別特定主機的資訊放進節點 / 邊特徵。

`loader.build_features()` 已自動排除上述欄位，並對 `protocol` 做 one-hot。

### 前處理

- 標準化 (min-max) 的統計量 **只能用 train 計算** (`load_split()` 已處理)。
- 若自行做額外特徵工程 / 特徵選擇 / 資料增強 / 過採樣，也只能使用 train。

---

## 3. 資料切分

### 主要切分：`episode_block_v1` (正式比較用)

檔案：`data/splits/episode_block_v1.csv` (已 commit，**不可自行重新切分**)

- 依 `attacker_machine_summary.csv` 的 74 次攻擊事件，把時間軸切成區塊：每個區塊 = 一次攻擊事件 + 前後正常流量；第一次攻擊前的純正常時段每 60 秒一塊。
- 依攻擊類型分層，**整個區塊** 分配到 train / val / test (目標 50 / 20 / 30，實際 47 / 25 / 27)。
- 每個 split 都包含全部 6 個類別。

| 類別 | train | val | test |
|------|------:|----:|-----:|
| Normal | 17,059 | 9,490 | 10,157 |
| ddos | 969 | 489 | 476 |
| ip-scan | 96 | 40 | 56 |
| mitm | 1,295 | 484 | 805 |
| port-scan | 1,003 | 447 | 494 |
| replay | 1,174 | 584 | 600 |

**為什麼不用逐筆隨機切分？** 攻擊 flow 在時間上是成群出現的。隨機切分時，測試 flow 的前後鄰居幾乎都在訓練集裡，LSTM 的滑動窗口和 GNN 的時間窗口圖會因此「看到」與測試樣本幾乎相同的訓練樣本，分數被灌水，而 RF 這類逐筆模型則不受影響 — 這樣的橫向比較是不公平的。以攻擊事件為單位切分可以避免這個問題。

**為什麼不用純時間順序切分？** 前 38% 的資料完全沒有攻擊，且各攻擊類型分佈在時間軸不同段落，純時間切分會讓某些類別在 test 中缺席。

### 次要切分：`random_stratified_v1` (僅供對照原論文)

逐筆隨機分層 50/20/30。只用來說明「我們的 RF 基線與原論文一致」，**不作為方法間比較的依據**。

參考：RF 基線在兩種切分下的差異 (5 seeds)

| split | detection F1 | identification macro-F1 | ip-scan F1 |
|-------|-----:|-----:|-----:|
| random_stratified_v1 | 0.9908 | 0.8827 | 0.529 |
| episode_block_v1 | 0.9803 | 0.8801 | 0.531 |

---

## 4. 訓練規則

| 項目 | 規定 |
|------|------|
| Test set | 只在最後產出預測時使用一次。**不可用 test 調參、選 epoch、選 threshold、選變體** |
| 超參數調整 / early stopping / model selection | 只看 val。detection 用 val F1，identification 用 val macro-F1 |
| 二元 threshold | 預設 0.5；若要調整，只能在 val 上決定並記錄在 meta |
| Seeds | 每個設定跑 **5 個 seed：0, 1, 2, 3, 4**，切分固定不變，seed 只影響模型初始化 / shuffle |
| 報告數字 | 5 seeds 的 mean ± std |
| 訓練超參數 | 各方法自訂 (`configs/default.yaml` 中只有 `data` 區塊是強制的)，最終使用的值寫進 meta 的 `notes` |

### Detection 的兩種做法 (擇一，記在 meta)

| `detection_from` | 做法 |
|------|------|
| `separate` | 另外訓練一個二元模型 |
| `identification` | 直接用多類別模型：`score = 1 - P(Normal)`，`y_pred = (score >= threshold)` |

兩種都允許，但同一個方法的 5 個 seed 必須一致。

### 代表變體 (每人在總表中只能有一個代表)

每個人會有很多變體 (不同窗口、FedAvg/FedProx、不同圖結構…)，全部都可以提交，但 **第 7 週的總比較表中每個方法只放一個「代表變體」**：

1. 代表變體依 **val 分數** 選出 (identification 看 val macro-F1)，**不可看 test 排行榜挑**。
2. 每個 run 都要用 `save_predictions(..., part="val")` 輸出 val 預測，`evaluate.py` 會 **自動計算** `val_score` (detection: F1，identification: macro-F1) 並與 test 分數並排顯示。val_score 不由成員自行填寫，確保大家的 val 分數算法一致、選擇過程可檢查。
3. 第 7 週開始前，每人在下表填好代表變體的 method 名稱並 commit：

| 成員 | 代表變體 (detection) | 代表變體 (identification) |
|------|------|------|
| A — LSTM |lstm_attn_global_w20 |lstm_attn_global_w20 |
| B — FL | | |
| C — GNN | | |

其餘變體放在各自章節做消融 / 分析用。

---

## 5. 各方法的對齊規則

所有方法最終都必須 **對 val split 與 test split 中的每一筆 flow 各給出一個預測** (val 11,534 筆、test 12,588 筆)，評估單位一律是 flow。

### 成員 A — LSTM + Attention

- **序列定義** 必須在 meta 的 `sequence` 欄位註明，二選一 (也可兩者都做成不同變體)：
  - `global`：所有 flow 依 `sort_by_time()` 排成一條序列。
  - `per_pair`：依 (`sAddress`, `rAddress`) 分組 (共 25 組)，各組內依 `sort_by_time()` 排序。
- 對每個目標 flow 取「它自己 + 前 W-1 筆」組成窗口，預測目標 flow 的標籤；W 記在 meta 的 `window`。
- 只能用 **過去** 的 flow (causal)，不可用未來的 flow。
- 窗口中的 context flow 可以來自任何 split (部署時本來就看得到過去流量)，但 **只能用其特徵，不可用其標籤**。
- 訓練樣本 = 目標 flow 屬於 train 的窗口；測試樣本 = 目標 flow 屬於 test 的窗口。
- 序列開頭不足 W 筆的 flow 用 padding 補齊，不可丟掉 (否則會缺少測試 flow_id)。

### 成員 B — Federated Learning

**本地模型分兩階段：**

1. **主要實驗用逐筆 MLP**：IID vs Non-IID、FedAvg vs FedProx、federated vs centralized vs local-only 全部用 MLP 完成。輸入與 RF / ANN 基線相同 (`load_split()` 的特徵)。
2. **第 4 週底檢查**：若 A 的 LSTM 在 val macro-F1 明顯勝過 ANN，第 5 週用 **與 A 相同的 LSTM 架構與窗口設定** 跑 FL-LSTM，作為 B 的代表變體；否則以 FL-MLP 為代表。

**規則：**

- FL 一律與 **同架構** 的 centralized 版本比較：FL-MLP 對 `ann_baseline` (或 `centralized_mlp`)，FL-LSTM 對 A 的 LSTM。
- method 名稱必須寫出架構，例如 `fl_mlp_fedavg_iid`、`fl_mlp_fedprox_noniid`、`local_mlp_client0`、`fl_lstm_fedprox_noniid`。
- **Client 切分只能從 train split 切出**，val 用於選 round / 調參，全域模型在同一個 test split 上評估。
- **Client 切分固定**：每種切分方式只產生一次 (seed 42)，存成 `data/splits/fl_<partition>_v1.csv` (欄位 `flow_id, client`，只含 train flow) 並 commit。5 個 model seed 共用同一份切分，seed 只影響模型訓練。
- Non-IID 切分建議以 `episode_block_v1.csv` 的 `block_id` 為單位分配給 client (同一次攻擊事件不拆開)，這樣 FL-LSTM 的窗口在 client 內也是連續的。
- meta 額外記錄：`partition`、`num_clients`、`rounds`、`local_epochs`、`strategy`、`local_model`。

### 成員 C — GNN (GAT)

- **節點以 `sAddress` / `rAddress` 建立**，不可只用 `sIPs` / `rIPs`，否則所有 ARP flow (含全部 ip-scan) 會掉出圖外。
  - 同一台主機可能同時以 IP 與 MAC 出現 (例如 `192.168.0.41` 與 `02:42:c0:a8:00:29`)，可以用 TCP flow 中的 `sIPs`↔`sMACs` 對應把它們合併成同一節點，作法記在 meta 的 `notes`。
  - 廣播位址 `ff:ff:ff:ff:ff:ff` 當作一個普通節點。
- 輸出必須是 **flow (edge) 層級** 的預測。若模型是 graph-level 分類，需把圖的預測對應回該圖內每一筆 flow，並在 meta 中註明。
- 建圖時間窗口內的 flow 若跨 split，訓練 loss 只能計算 train flow；val/test flow 的標籤不可參與訓練。
- 遵守第 2 節的位址使用限制 (只能決定拓撲，不能當特徵)。
- meta 額外記錄：`graph_window` (秒)、`node_key` (例如 `address` / `address_merged`)。

### 基線

- `scripts/baseline_rf.py` 已提供 RF 基線 (`rf_baseline`)，也是產出預測檔的範本。
- DT / ANN 基線請比照同樣格式提交 (`dt_baseline`, `ann_baseline`)。ANN 同時是 FL-MLP 的 centralized 對照組。

---

## 6. 評估指標

全部由 `src/utils/metrics.py` 計算。

### Detection (二元)

| 指標 | 角色 |
|------|------|
| **F1 (attack class)** | **主要排名指標** |
| PR-AUC | 主要 (需提供 score)，不受 threshold 影響，適合不平衡資料 |
| FPR (誤報率) | 主要；ICS 場域誤報代價高 |
| Recall, Precision, MCC | 輔助 |
| ROC-AUC, Accuracy | 僅供參考 (資料不平衡下偏高、鑑別力低) |

### Identification (多類別)

| 指標 | 角色 |
|------|------|
| **Macro-F1** | **主要排名指標** — 每個類別權重相同 |
| 各類別 F1，特別是 **ip-scan** | 主要；RF 基線 ip-scan F1 僅 0.53，是本研究的主要改進目標 |
| Balanced accuracy, MCC | 輔助 |
| Weighted-F1, Accuracy | 僅供參考 — Normal 佔 80%，會掩蓋小類別的表現 |

### 解讀注意事項

- **ip-scan 在 test 只有 56 筆，來自少數幾次攻擊事件**。seed 的 std 只反映訓練隨機性，不反映測試樣本少造成的不確定性。ip-scan F1 相差 0.05 左右可能只是雜訊，報告中不要據此宣稱某方法較好，需搭配 confusion matrix 說明實際多對 / 少對幾筆。
- 兩個方法差距小於彼此的 std 時，視為無明顯差異。

### 效率 (寫在 meta，不參與排名)

`n_params`、`train_time_s`、`infer_time_s`、`hardware`。

---

## 7. 提交格式

用 `save_predictions()` 自動產生，路徑：

```
results/predictions/<method>/<split>/<task>_seed<k>.csv       # test 預測 (part="test")
results/predictions/<method>/<split>/<task>_seed<k>.json      # meta (隨 test 預測寫出)
results/predictions/<method>/<split>/<task>_seed<k>_val.csv   # val 預測 (part="val")
```

val 與 test 預測檔的欄位格式相同。

CSV 欄位：

| 任務 | 欄位 |
|------|------|
| detection | `flow_id, y_pred, score` (`score` = P(attack)，或任何「越大越像攻擊」的分數) |
| identification | `flow_id, y_pred, prob_Normal, prob_ddos, prob_ip-scan, prob_mitm, prob_port-scan, prob_replay` |

- **不需要也不要提供 y_true**，評估腳本會從原始資料取真實標籤，避免標籤對錯。
- `method` 命名：小寫加底線，例如 `lstm_attn_global_w20`、`fl_mlp_fedprox_noniid`、`gnn_gat_w10`。同一方法的不同變體用不同名稱。

### meta 欄位

| 欄位 | 必填 | 說明 |
|------|:---:|------|
| `member` | ✔ | `A` / `B` / `C` / `shared` |
| `detection_from` | detection 必填 | `separate` / `identification` |
| `threshold` | | 若不是 0.5 |
| `n_params`, `train_time_s`, `infer_time_s`, `hardware` | ✔ | 效率資訊 |
| `notes` | ✔ | 最終超參數 |
| 方法專屬欄位 | ✔ | A: `sequence`, `window`；B: `partition`, `num_clients`, `rounds`, `local_epochs`, `strategy`, `local_model`；C: `graph_window`, `node_key` |

### 範例

```python
from src.data.loader import load_split
from src.utils.metrics import identification_metrics, save_predictions

data = load_split(task="identification")          # 預設 episode_block_v1, NST
# ... 用 data["X_train"], data["y_train"] 訓練 ...
# 調參時自己看 val 分數 (與 evaluate.py 同一個函式)
print(identification_metrics(data["y_val"], model.predict(data["X_val"]))["macro_f1"])

# 定案後：val 與 test 各輸出一次
val_proba = model.predict_proba(data["X_val"])      # shape (11534, 6)
save_predictions("lstm_attn_global_w20", "identification", seed=0,
                 flow_ids=data["X_val"].index, y_pred=val_proba.argmax(1), score=val_proba, part="val")
proba = model.predict_proba(data["X_test"])         # shape (12588, 6)
save_predictions("lstm_attn_global_w20", "identification", seed=0,
                 flow_ids=data["X_test"].index, y_pred=proba.argmax(1), score=proba,
                 meta={"member": "A", "sequence": "global", "window": 20,
                       "n_params": 123456, "train_time_s": 300.0, "infer_time_s": 2.1,
                       "hardware": "RTX 3060", "notes": "hidden=128, lr=1e-3, epochs=40"})
```

---

## 8. 產生排行榜與 Git 規則

```bash
python scripts/evaluate.py                           # 主要比較
python scripts/evaluate.py --split random_stratified_v1   # 對照原論文
```

腳本會檢查每個預測檔：flow_id 是否完整涵蓋 val / test split、有無重複、y_pred 範圍、機率欄位是否齊全。
test 檔不合格 → 該 run 不列入排行榜；val 檔缺少或不合格 → 該 run 的 val_score 留空 (不能作為代表變體)。

輸出 (皆在本地產生，**不 commit**，已加入 `.gitignore`)：
- `results/leaderboard_<split>.md` — 各方法 test 指標與自動計算的 val_score (mean ± std)，seed 數不足 5 會標 ⚠
- `results/runs_<split>.csv` — 每個 run 的完整指標
- `results/confusion/<split>/<method>_<task>.csv` — 跨 seed 加總的 confusion matrix

**Git 規則**：只 commit 自己的 `results/predictions/<method>/` 與程式碼；排行榜由每個人 pull 後自行執行 `evaluate.py` 產生，避免互相衝突。不要修改別人 method 目錄下的檔案。

---

## 9. 提交前自我檢查

- [ ] 用的是 `load_split()` 或 `data/splits/episode_block_v1.csv`，沒有自己重新切分
- [ ] 特徵中沒有 IP / MAC / 時間戳記
- [ ] ARP flow (含 ip-scan) 有被納入，沒有因為 IP 是空值而被丟掉
- [ ] 正規化與任何 fit 動作只用 train
- [ ] 沒有用 test 做任何調參、early stopping 或變體挑選
- [ ] detection 與 identification 都跑了 seed 0–4
- [ ] 每個 run 都有 val 預測 (11,534 筆) 與 test 預測 (12,588 筆)
- [ ] meta 填了 `member` 與方法專屬欄位
- [ ] `python scripts/evaluate.py` 沒有出現驗證錯誤

---

## 版本紀錄

- **v1.1** (2026-10-14)：補充 ARP / ip-scan 無 IP 的資料特性與 GNN 建圖規則、LSTM 序列定義、FL 本地模型兩階段策略與固定 client 切分、detection 可由多類別模型推導、代表變體以 val 選出 (val 預測檔 + 自動計算 val_score)、成員自行調參的分工、meta 必填欄位、排行榜不 commit、ip-scan 小樣本解讀注意事項。切分檔不變。
- **v1** (2026-10-14)：初版。切分檔 `episode_block_v1.csv`、`random_stratified_v1.csv` (由 `scripts/make_splits.py`, seed 42 產生)。
  切分檔若需修改，必須升版為 v2 並通知全員重跑，不可覆蓋 v1。
