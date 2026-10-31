# 成員 C — GNN (EdgeGAT) 實驗紀錄

程式：`src/models/gnn_detector.py` (建圖 + 模型)、`scripts/train_gnn.py` (訓練 + 輸出預測)。
環境：`setup/requirements-gnn.txt` (CPU 版 PyTorch + PyG)。

## 方法

- **建圖**：依 `sort_by_time()` 排序後，以固定時間窗口 (秒) 切段，每段一張圖。
- **節點**：`sAddress` / `rAddress`。`address_merged` 變體再把 ARP flow 的 MAC 換成該主機的 IP
  (MAC→IP 對應由 train split 的 TCP flow 多數決得出，例如 `02:42:c0:a8:00:29` → `192.168.0.41`)，
  讓同一台主機的 ARP 與 TCP 行為落在同一個節點上。位址只決定拓撲，不進入任何特徵。
- **節點特徵**：進 / 出邊 flow 特徵平均 + log 進 / 出度數。
- **邊**：每筆 flow 一條邊 + 反向邊 (方向旗標)，邊特徵 = `load_split()` 的 flow 特徵。
- **模型**：2 層 GATConv (hidden 64, 4 heads, `edge_dim`) → 對每筆 flow 取 `[h_src, h_dst, flow 特徵]` → MLP 分類。
- **訓練**：loss 只計算 train flow；early stopping 依 val (detection F1 / identification macro-F1)。

## 調參 (只看 val，seed 0–1)

Identification val macro-F1：

| 變體 | val |
|------|----:|
| window 2 | 0.914 |
| window 5 | 0.933 |
| **window 10** | **0.935** |
| window 30 | 0.894 |
| window 60 | 0.15–0.56 (不穩定) |
| window 10 + class_weight sqrt | 0.937 |
| window 10, 1 層 / 3 層 GAT | 0.938 / 0.933 |
| window 10, hidden 128 | 0.937 |
| **window 10 + address_merged** | **0.959** |
| window 5 + address_merged | 0.949 |
| window 10 + address_merged + patience 20 (5 seeds) | 0.960 ± 0.001 |

class weight、層數、hidden size 的差異都在雜訊範圍內；**節點合併是唯一明顯有效的改動**。
patience 10 時 seed 4 會過早停止 (val 0.933)，改 patience 20 後 5 個 seed 都穩定。

Detection val F1：window 2 = 0.985、3 = 0.988、**5 = 0.988**、10 = 0.980；
`address_merged` 對 detection 沒有幫助 (w5: 0.984)；由 identification 模型推導 (1 − P(Normal)) 為 0.979，不如另外訓練二元模型。

## 代表變體 (依 val 選出)

| 任務 | method | 設定 |
|------|--------|------|
| detection | `gnn_gat_w5` | window 5, node_key address, patience 10, `detection_from=separate` |
| identification | `gnn_gat_w10_merged` | window 10, node_key address_merged, patience 20 |

`gnn_gat_w10` (未合併節點) 保留作為消融對照。

## Test 結果 (`scripts/evaluate.py`，5 seeds)

| task | method | 主要指標 | 備註 |
|------|--------|---------|------|
| detection | rf_baseline | F1 0.980 ± 0.001 | PR-AUC 0.994, FPR 0.008 |
| detection | gnn_gat_w5 | F1 0.970 ± 0.002 | PR-AUC 0.994, FPR 0.011 |
| identification | rf_baseline | macro-F1 0.880 ± 0.000 | ip-scan 0.531, mitm 0.903, replay 0.923 |
| identification | gnn_gat_w10 | macro-F1 0.924 ± 0.016 | ip-scan 0.935, mitm 0.921, replay 0.800 |
| identification | **gnn_gat_w10_merged** | **macro-F1 0.950 ± 0.007** | ip-scan 0.953, mitm 0.973, replay 0.873 |

## 分析

- **ip-scan**：RF 會把 port-scan / mitm / replay 的 ARP flow 誤判為 ip-scan (5 seeds 合計 465 筆誤報)，
  GNN 只有 28 筆 (merged)。ip-scan 的 56 筆 test 樣本在所有 seed 都全部抓到。
  逐筆特徵難以分辨不同攻擊中的 ARP flow，但圖上同一窗口內攻擊者節點的其他行為 (掃描 / TCP 連線) 提供了上下文。
- **節點合併的效果**：未合併時攻擊者的 ARP 行為 (MAC 節點) 與 TCP 行為 (IP 節點) 被拆開，
  replay 常被誤判為 mitm (5 seeds 合計 330 筆)；合併後降到 9 筆，mitm F1 0.921 → 0.973、replay 0.800 → 0.873。
- **剩餘錯誤集中在單一攻擊事件** (僅分析，未用於選模型)：
  - ddos → replay 408 筆全部來自事件 `E39`；另一個 test ddos 事件 `E45` 完全正確。val 上 ddos 幾乎沒有錯誤，屬於事件層級的泛化問題。
  - replay → Normal 314 筆來自 `E21` / `E37`，每個事件每個 seed 約 35 筆 (約 12%)。
  - test 中 ddos / replay 各只有 2 個事件，這類單一事件的錯誤會大幅影響類別 F1，報告中需一併說明。
- **Detection**：GNN 略低於 RF (F1 0.970 vs 0.980)，PR-AUC 相同 (0.994)，FPR 稍高 (0.011 vs 0.008)。
  圖結構對「是否為攻擊」幫助有限，主要價值在攻擊類型辨識。

## 重現

```bash
.venv/bin/python scripts/train_gnn.py --task detection --window 5
.venv/bin/python scripts/train_gnn.py --task identification --window 10 --node-key address_merged --patience 20
.venv/bin/python scripts/train_gnn.py --task identification --window 10   # 消融：未合併節點
.venv/bin/python scripts/evaluate.py
```
