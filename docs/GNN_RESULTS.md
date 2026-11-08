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

## 消融：提升來自哪裡？(2026-11-09)

問題：GNN 的提升是來自 GAT 的 message passing，還是只來自「同一時間窗口內其他 flow」的上下文？
窗口 / 變體一律依 val (seed 0–1) 選出，再跑 5 seeds；以下為 `evaluate.py` 的 test 結果 (mean)。

### A. GNN 內部消融 (identification，window 10，address_merged，epochs 400 / patience 30)

| method | 說明 | val | test macro-F1 |
|---|---|---:|---:|
| `gnn_flowonly_mlp` | 分類器只看 flow 特徵 (同一套訓練流程) | 0.797 | 0.779 ± 0.048 |
| `gnn_ctx_w10_merged` | 0 層 GAT：flow + 來源 / 目的節點聚合特徵，**無 message passing** | 0.959 | 0.964 ± 0.011 |
| `gnn_gat_w10_merged` | 完整 2 層 GAT (代表變體) | 0.960 | 0.950 ± 0.007 |

**message passing 幾乎沒有貢獻**：拿掉 GAT 層後 val 相同、test 甚至略高。
提升來自節點聚合特徵 (= 同一窗口內同一主機的其他 flow)。原 100 epochs 上限時兩者都沒收斂，因此消融改用 400 epochs。

### B. 不用 GNN：RF + 上下文特徵 (`scripts/ablation_rf_context.py`)

把上下文特徵直接接在 flow 特徵後面，交給 RF (n_estimators=100) 訓練。2×2 設計：

| context | 聚合範圍 | 時間範圍 | detection method | det F1 / FPR | identification method | macro-F1 | ip-scan | val (ident) |
|---|---|---|---|---|---|---:|---:|---:|
| (無) | — | — | `rf_baseline` | 0.980 / 0.0082 | `rf_baseline` | 0.880 | 0.531 | 0.886 |
| `node` | 同主機 | 固定窗口 (含窗口內未來) | `rf_ctx_w10` | 0.991 / 0.0037 | `rf_ctx_w10_merged` | **0.990** | **0.984** | **0.969** |
| `global` | 全網路 (不分主機) | 固定窗口 | `rf_gctx_w5` | 0.989 / 0.0014 | `rf_gctx_w5` | 0.904 | 0.536 | 0.956 |
| `causal_node` | 同主機 | 只看過去 W 秒 | `rf_cctx_w5_merged` | **0.994 / 0.0024** | `rf_cctx_w10_merged` | 0.934 | 0.669 | 0.934 |
| `causal_global` | 全網路 | 只看過去 W 秒 | `rf_cgctx_w10` | 0.990 / 0.0017 | `rf_cgctx_w10` | 0.901 | 0.501 | 0.930 |

- 與 GNN 用同一份建圖 / 同一套位址規則：位址只決定「哪些 flow 聚合在一起」，不進入特徵。
- `rf_ctx_w10_merged` 在 test 上勝過 GNN (0.990 vs 0.950)，訓練只要約 1 秒；E39 的 ddos→replay 408 筆錯誤也消失。
- 注意 val 與 test 的落差：`rf_ctx_w10_merged` val 0.969 / test 0.990；`rf_gctx_w5` val 0.956 / test 0.904 (主要是 ip-scan)。

### C. 攻擊者節點捷徑檢查 (`scripts/analyze_attacker_node.py`)

攻擊者固定是 `192.168.0.41 / 02:42:c0:a8:00:29`，**整個資料集只有 9 筆 Normal flow 涉及它**，
因此「認出攻擊者主機」幾乎等於「認出攻擊」。位址雖不是特徵，但同主機聚合特徵可能間接描繪出攻擊者主機。

Test per-class recall (5 seeds 平均)，依 flow 是否涉及攻擊者位址分群：

| 涉及攻擊者 | class | n | rf_baseline | gnn_gat_w10_merged | rf_ctx (node) | rf_gctx (global) | rf_cctx (causal node) | rf_cgctx (causal global) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 否 | Normal | 10156 | 0.992 | 0.996 | 0.998 | 0.999 | 0.999 | 0.999 |
| 否 | mitm | 642 | 0.997 | 0.981 | 0.994 | 0.995 | 0.999 | 0.997 |
| 否 | replay | 160 | 0.999 | 0.633 | 0.981 | 0.984 | 0.992 | 0.994 |
| 是 | mitm | 163 | 0.568 | 0.990 | 1.000 | 0.951 | 0.788 | 0.880 |
| 是 | replay | 440 | 0.880 | 0.987 | 0.945 | 0.924 | 0.943 | 0.887 |
| 是 | ddos | 476 | 0.997 | 0.829 | 1.000 | 0.992 | 1.000 | 0.998 |
| 是 | ip-scan | 56 | 1.000 | 1.000 | 1.000 | 0.432 | 0.936 | 0.536 |
| 是 | port-scan | 494 | 0.907 | 1.000 | 0.997 | 0.960 | 1.000 | 0.993 |

### 結論 (報告用)

1. **GNN 的提升來自窗口上下文，不是 GAT 結構**。同樣的上下文給 RF 用，效果更好也更快。
2. **Detection 的提升不依賴主機身分**：完全不分主機的 `global` / `causal_global` 也有 F1 0.989–0.990、FPR 0.0014–0.0017 (RF 0.980 / 0.0082)。
   這是「網路當下整體狀態」的資訊，捷徑疑慮較小；且 causal 版本符合即時部署條件。
3. **mitm / replay 的改善大部分不依賴主機身分**：global context 的 mitm F1 0.981 (RF 0.903)。
4. **ip-scan 的改善需要「同主機 + 窗口內未來」的上下文**：只有 `node` (非因果) 能到 0.98；
   拿掉主機區分 (global) 回到 0.54，只看過去 (causal_node) 為 0.67。
   推測 ip-scan 的 ARP flow 要靠「同一主機接下來做了什麼」(後續掃描) 才能和其他攻擊的 ARP flow 區分。
   因為本資料集攻擊者主機幾乎不產生正常流量，**這部分提升無法排除「間接辨認攻擊者主機」的可能**，
   在報告中需列為限制 (ICS-Flow 只有單一攻擊者，無法驗證跨主機泛化)。
5. 非因果窗口 (會看到窗口內最多 W 秒的未來 flow) 在即時偵測時代表最多 W 秒的延遲；與 LSTM (嚴格因果) 比較時需註明。

### 待討論 (未更動代表變體)

- `rf_ctx_w10_merged` 的 val (0.969) 高於 GNN 代表變體 (0.960)，但它不是 GNN，目前只作為消融，
  §4 的代表變體仍為 `gnn_gat_w10_merged` / `gnn_gat_w5`。是否改列為「GNN 方法的簡化版」或獨立方法，需組內討論。

## 重現

```bash
.venv/bin/python scripts/train_gnn.py --task detection --window 5
.venv/bin/python scripts/train_gnn.py --task identification --window 10 --node-key address_merged --patience 20
.venv/bin/python scripts/train_gnn.py --task identification --window 10   # 消融：未合併節點
.venv/bin/python scripts/evaluate.py
```

```bash
# 消融 (2026-11-09)
.venv/bin/python scripts/train_gnn.py --task identification --window 10 --node-key address_merged --patience 30 --epochs 400 --layers 0 --method gnn_ctx_w10_merged
.venv/bin/python scripts/train_gnn.py --task identification --window 10 --node-key address_merged --patience 30 --epochs 400 --flow-only --method gnn_flowonly_mlp
.venv/bin/python scripts/ablation_rf_context.py --task detection      --context node          --window 10
.venv/bin/python scripts/ablation_rf_context.py --task identification --context node          --window 10 --node-key address_merged
.venv/bin/python scripts/ablation_rf_context.py --task detection      --context global        --window 5
.venv/bin/python scripts/ablation_rf_context.py --task identification --context global        --window 5
.venv/bin/python scripts/ablation_rf_context.py --task detection      --context causal_node   --window 5  --node-key address_merged
.venv/bin/python scripts/ablation_rf_context.py --task identification --context causal_node   --window 10 --node-key address_merged
.venv/bin/python scripts/ablation_rf_context.py --task detection      --context causal_global --window 10
.venv/bin/python scripts/ablation_rf_context.py --task identification --context causal_global --window 10
.venv/bin/python scripts/evaluate.py
.venv/bin/python scripts/analyze_attacker_node.py rf_baseline gnn_gat_w10_merged rf_ctx_w10_merged rf_gctx_w5 rf_cctx_w10_merged rf_cgctx_w10
```
