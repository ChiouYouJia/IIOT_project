# 評估規範 (Evaluation Protocol) v1

三位成員 (LSTM / FL / GNN) 的模型必須依照本規範產出結果，才能在第 7 週做公平的橫向比較。
**原則：資料切分、標籤、指標計算全部共用；各自只負責模型本身。**

---

## 1. 一句話流程

```
src.data.loader.load_split()  →  訓練 (只用 train，調參只看 val)
  →  對 test split 每一筆 flow 輸出預測  →  src.utils.metrics.save_predictions()
  →  python scripts/evaluate.py  →  results/leaderboard_episode_block_v1.md
```

正式報告中的數字 **一律取自 `scripts/evaluate.py` 的輸出**，不採用各自 notebook 算的數字。

---

## 2. 資料與標籤

| 項目 | 規定 |
|------|------|
| 資料 | `data/raw/Dataset.csv` (45,718 筆 flow)，`flow_id` = 原始列索引 (0-based) |
| 標籤策略 | **NST** (`NST_B_Label` / `NST_M_Label`)。IT 標籤會把攻擊期間的正常流量也標成攻擊，不作為主要結果 |
| 二元偵測 (detection) | 0 = Normal, 1 = Attack |
| 多類別辨識 (identification) | 固定編碼 `0 Normal, 1 ddos, 2 ip-scan, 3 mitm, 4 port-scan, 5 replay` (`loader.CLASSES`) |
| PLC snapshot 資料 | 本版不使用；若要用，需另外說明並單獨列為額外實驗 |

### 禁用特徵 (避免捷徑學習)

攻擊者固定是 `192.168.0.41 / 02:42:c0:a8:00:29`，模型只要記住 IP 就能「偵測」攻擊。因此：

- `sAddress, rAddress, sMACs, rMACs, sIPs, rIPs` **不可作為輸入特徵**。
- `startDate, endDate, start, end, startOffset, endOffset` 不可作為特徵 (可用於排序 / 建窗口)。
- **GNN 例外**：可以用 IP 決定圖的節點與連邊 (拓撲)，但 **不可** 把 IP 編號 one-hot、learnable node-ID embedding 或任何可識別特定主機的資訊放進節點特徵。

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
| Test set | 只在最後產出預測時使用一次。**不可用 test 調參、選 epoch、選 threshold** |
| 超參數調整 / early stopping / model selection | 只看 val。detection 用 val F1，identification 用 val macro-F1 |
| 二元 threshold | 預設 0.5；若要調整，只能在 val 上決定並記錄在 meta |
| Seeds | 每個設定跑 **5 個 seed：0, 1, 2, 3, 4**，切分固定不變，seed 只影響模型初始化 / shuffle |
| 報告數字 | 5 seeds 的 mean ± std |
| 兩個任務 | detection 與 identification 都要做 |

---

## 5. 各方法的對齊規則

所有方法最終都必須 **對 test split 中的每一筆 flow 給出一個預測** (共 12,588 筆)，評估單位一律是 flow。

### 成員 A — LSTM + Attention

- 先依 `start` 時間排序整份資料，對每個目標 flow 取「它自己 + 前 W-1 筆」組成窗口，預測目標 flow 的標籤。
- 只能用 **過去** 的 flow (causal)，不可用未來的 flow。
- 窗口中的 context flow 可以來自任何 split (部署時本來就看得到過去流量)，但 **只能用其特徵，不可用其標籤**。
- 訓練樣本 = 目標 flow 屬於 train 的窗口；測試樣本 = 目標 flow 屬於 test 的窗口。
- 序列開頭不足 W 筆的 flow 用 padding 補齊，不可丟掉 (否則會缺少測試 flow_id)。

### 成員 B — Federated Learning

- Client 資料只能從 **train split** 切出 (IID / Non-IID 皆可)，val 用於選 round / 調參。
- 全域模型在 **同一個 test split** 上評估並輸出預測。
- Local-only 與 centralized 對照組也必須用相同 train/val/test，並分別以不同 method 名稱提交 (例：`fl_fedavg_noniid`, `fl_local_client0`, `fl_centralized`)。
- meta 額外記錄：client 數、切分方式、rounds、local epochs、通訊量 (可選)。

### 成員 C — GNN (GAT)

- 輸出必須是 **flow (edge) 層級** 的預測。若模型是 graph-level 分類，需把圖的預測對應回該圖內每一筆 flow，並在 meta 中註明。
- 建圖時間窗口內的 flow 若跨 split，訓練 loss 只能計算 train flow；val/test flow 的標籤不可參與訓練。
- 遵守第 2 節的 IP 使用限制 (只能決定拓撲，不能當特徵)。

### 基線

- `scripts/baseline_rf.py` 已提供 RF 基線 (`rf_baseline`)，也是產出預測檔的範本。
- DT / ANN 基線請比照同樣格式提交 (`dt_baseline`, `ann_baseline`)。

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
| 各類別 F1，特別是 **ip-scan** | 主要；原論文 ip-scan F1 僅約 0.52，是本研究的主要改進目標 |
| Balanced accuracy, MCC | 輔助 |
| Weighted-F1, Accuracy | 僅供參考 — Normal 佔 80%，會掩蓋小類別的表現 |

### 效率 (寫在 meta，不參與排名)

`n_params`、`train_time_s`、`infer_time_s`、`hardware`。

---

## 7. 提交格式

用 `save_predictions()` 自動產生，路徑：

```
results/predictions/<method>/<split>/<task>_seed<k>.csv
results/predictions/<method>/<split>/<task>_seed<k>.json   # meta
```

CSV 欄位：

| 任務 | 欄位 |
|------|------|
| detection | `flow_id, y_pred, score` (`score` = P(attack)，或任何「越大越像攻擊」的分數) |
| identification | `flow_id, y_pred, prob_Normal, prob_ddos, prob_ip-scan, prob_mitm, prob_port-scan, prob_replay` |

- **不需要也不要提供 y_true**，評估腳本會從原始資料取真實標籤，避免標籤對錯。
- `method` 命名：小寫加底線，例如 `lstm_attn_w20`、`fl_fedprox_noniid`、`gnn_gat`。同一方法的不同變體用不同名稱。

### 範例

```python
from src.data.loader import load_split
from src.utils.metrics import save_predictions

data = load_split(task="identification")          # 預設 episode_block_v1, NST
# ... 用 data["X_train"], data["y_train"] 訓練；用 data["X_val"], data["y_val"] 調參 ...
proba = model.predict_proba(data["X_test"])         # shape (12588, 6)
save_predictions("lstm_attn_w20", "identification", seed=0,
                 flow_ids=data["X_test"].index, y_pred=proba.argmax(1), score=proba,
                 meta={"member": "A", "n_params": 123456, "train_time_s": 300.0,
                       "infer_time_s": 2.1, "hardware": "RTX 3060", "notes": "W=20, hidden=128"})
```

---

## 8. 產生排行榜

```bash
python scripts/evaluate.py                           # 主要比較
python scripts/evaluate.py --split random_stratified_v1   # 對照原論文
```

腳本會檢查每個預測檔：flow_id 是否完整涵蓋 test split、有無重複、y_pred 範圍、機率欄位是否齊全；不合格的檔案會被列出並排除。

輸出：
- `results/leaderboard_<split>.md` — 各方法 mean ± std，seed 數不足 5 會標 ⚠
- `results/runs_<split>.csv` — 每個 run 的完整指標
- `results/confusion/<split>/<method>_<task>.csv` — 跨 seed 加總的 confusion matrix

---

## 9. 提交前自我檢查

- [ ] 用的是 `load_split()` 或 `data/splits/episode_block_v1.csv`，沒有自己重新切分
- [ ] 特徵中沒有 IP / MAC / 時間戳記
- [ ] 正規化與任何 fit 動作只用 train
- [ ] 沒有用 test 做任何調參或 early stopping
- [ ] detection 與 identification 都跑了 seed 0–4
- [ ] 預測涵蓋全部 12,588 筆 test flow
- [ ] `python scripts/evaluate.py` 沒有出現驗證錯誤

---

## 版本紀錄

- **v1** (2026-10-14)：初版。切分檔 `episode_block_v1.csv`、`random_stratified_v1.csv` (由 `scripts/make_splits.py`, seed 42 產生)。
  切分檔若需修改，必須升版為 v2 並通知全員重跑，不可覆蓋 v1。
