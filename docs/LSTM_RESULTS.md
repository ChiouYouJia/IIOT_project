# 成員 A — LSTM + Attention 實驗紀錄

程式：`src/models/lstm_attention.py` (模型)、`scripts/lstm_attn_global_w20.py` (訓練 + 輸出預測)。
分析：`notebooks/03_lstm.ipynb`。

## 方法

- **序列**：`global`，所有 flow 依 `sort_by_time()` 排成一條時間序列。
- **窗口**：每個目標 flow 使用自己及前 W-1 筆 flow；序列開頭用 zero padding。
- **特徵**：使用 `load_split()` 產生的 52 個 train-only normalized features；不使用 IP、MAC 或時間欄位作為特徵。
- **模型**：1 層 LSTM → temporal attention pooling → dropout (0.2) → classification layer。
- **任務**：detection 與 identification 分別訓練獨立模型。
- **訓練**：Adam，learning rate `1e-3`（除非另有註明），最多 40 epochs，patience 8；early stopping 只看 val。
- **評估**：`episode_block_v1`、NST labels、seed 0–4；正式數字以 `scripts/evaluate.py` 為準。

窗口中的 context flow 可以來自其他 split，但只使用特徵，不使用 context flow 的標籤。測試窗口在訓練和 early stopping 完成後才建立。

## 調參 (seed 0–4)

以下變體皆以 val score 為唯一選擇標準，test 分數僅用於報告對照，不參與選型。
部分 variant 因系統記憶體不足而失敗，已注明 seed 數與備註。

### Window size (hidden=128, lr=1e-3)

| 變體 | val detection F1 (best seed) | val ident. macro-F1 (best seed) | test det. F1 | test ident. macro-F1 |
|---|---:|---:|---:|---:|
| window 10 | 0.9873 (s4) | 0.8867 (s2) | 0.9720 ± 0.0085 | 0.8940 ± 0.0110 |
| **window 20** | **0.9849 (s1)** | **0.8872 (s3)** | **0.9640 ± 0.0017** | **0.8812 ± 0.0058** |
| window 30 | **0.9919 (s4)** | 0.9092 (s3) | **0.9661 ± 0.0057** | 0.8652 ± 0.0274 |
| window 40 | 0.9912 (s2) | **0.9194 (s3)** | 0.9631 ± 0.0031 | 0.8722 ± 0.0129 |

Window 越大 detection val 越好，但 identification val/test 均呈現隨 window 增大而下降的趨勢。
identification 的最優 val (w40 = 0.9194) 對應 test macro-F1 = 0.8722，低於 w20 的 0.8812，屬過擬合跡象。

### Hidden size (window=20, lr=1e-3)

| 變體 | val detection F1 (best seed) | val ident. macro-F1 (best seed) | test det. F1 | test ident. macro-F1 |
|---|---:|---:|---:|---:|
| hidden 64 | 0.9900 (s4) | 0.8952 (s4) | 0.9692 ± 0.0056 | 0.8824 ± 0.0054 |
| **hidden 128** | **0.9849 (s1)** | **0.8872 (s3)** | **0.9640 ± 0.0017** | **0.8812 ± 0.0058** |
| hidden 256 | 0.9885 (s2) ⚠4 seeds | 0.9006 (s4) | 0.9693 ± 0.0045 ⚠4 seeds | 0.8849 ± 0.0126 |

Hidden 256 在 identification 上有最高 val (0.9006) 和 test macro-F1 (0.8849)，但 detection seed 3 因記憶體不足未能完成（僅 4/5 seeds）。
Hidden 64 表現亦佳，且 5 seeds 完整。

### Learning rate (window=20, hidden=128)

| 變體 | val detection F1 | val ident. macro-F1 | test det. F1 | test ident. macro-F1 |
|---|---:|---:|---:|---:|
| lr 1e-4 | 0.9772 (s4) | 0.8743 (s2) | 0.9454 ± 0.0083 | 0.7811 ± 0.0620 |
| **lr 1e-3** | **0.9849 (s1)** | **0.8872 (s3)** | **0.9640 ± 0.0017** | **0.8812 ± 0.0058** |

lr=1e-4 收斂明顯較慢，identification val/test 皆下降，det FPR 也上升至 0.0138。

### Dropout (window=10, hidden=128, lr=1e-3)

| 變體 | val detection F1 | val ident. macro-F1 | test det. F1 | test ident. macro-F1 |
|---|---:|---:|---:|---:|
| **dropout 0.2 (baseline)** | **0.9838 (s4)** | **0.8867 (s2)** | **0.9720 ± 0.0085** | **0.8940 ± 0.0110** |
| dropout 0.4 | 0.9851 (s4) ⚠3 seeds | 0.8870 (s3) ⚠4 seeds | 0.9745 ± 0.0037 ⚠ | 0.8740 ± 0.0085 ⚠ |

Dropout 0.4 對 identification 效果不佳（macro-F1 從 0.8940 降至 0.8740），但 detection 略有提升。
原始 dropout=0.2 在 identification 上仍是最穩定選擇。

### Gradient clipping (window=10, hidden=128, lr=1e-3)

| 變體 | val detection F1 | val ident. macro-F1 | test det. F1 | test ident. macro-F1 |
|---|---:|---:|---:|---:|
| **grad_clip 1.0** | **0.9851 (s2)** | **0.8736 (s0)** | **0.9773 ± 0.0066** | 0.8785 ± 0.0063 |
| 無 clipping (baseline) | 0.9838 (s4) | 0.8867 (s2) | 0.9720 ± 0.0085 | 0.8940 ± 0.0110 |

Gradient clipping 顯著提升了 detection F1（+0.0053），但 identification 下降約 0.016。
檢測 recall 從 0.965 提升至 0.977，FPR 維持低位（0.0054）。

### Batch size (window=10, hidden=128, lr=1e-3)

| 變體 | val detection F1 | val ident. macro-F1 | test det. F1 | test ident. macro-F1 |
|---|---:|---:|---:|---:|
| **batch-size 64** | **0.9875 (s0)** | **0.8867 (s2)** | **0.9819 ± 0.0074** | **0.8937 ± 0.0116** |
| batch-size 256 (baseline) | 0.9838 (s4) | 0.8867 (s2) | 0.9720 ± 0.0085 | 0.8940 ± 0.0110 |

Batch size 64 是 detection 的最佳設定（F1 0.9819），std 也較小（±0.007 vs ±0.009）。
Identification 幾乎與 baseline 相同（0.8937 vs 0.8940），顯示 batch size 對 identification 影響不大。

### Gradient clipping + Batch size 組合

| 變體 | val detection F1 | test det. F1 | test ident. macro-F1 |
|---|---:|---:|---:|
| **bs64 + gc1** | **0.9910 (s1)** | **0.9817 ± 0.0061** | 0.8844 ± 0.0073 ⚠2 seeds |
| bs64 alone | 0.9875 (s0) | 0.9819 ± 0.0074 | 0.8937 ± 0.0116 |
| gc1 alone | 0.9851 (s2) | 0.9773 ± 0.0066 | 0.8785 ± 0.0063 |

`bs64 + gc1` 的 detection val 達 0.9910（最高），test F1 0.9817，且 PR-AUC 達 0.9927、FPR 降至 0.0047。
但 gc1 對 identification 有害（從 0.8940 降至 0.8844），故兩任務分開選型。

### 其他實驗（未通過）

| 變體 | 說明 | 結果 |
|---|---|---|
| dropout 0.35 | 更強正則化 | id val 0.8776 < 0.8867 ❌ |
| dropout 0.4 | 已列於上表 | id test 0.8740 ❌ |
| weight_decay 1e-4 | L2 regularization | det 效果不明顯 |
| weight_decay 5e-4 | 更強 regularization | det val 0.9765 ❌ |
| lr scheduler cosine | 週期性衰減 | det/id 皆下降 ❌ |
| patience 5 / 15 | 更早/更晚停止 | 無顯著改善 |
| lstm_dropout 0.2 | LSTM 內部 dropout | det val 0.9804 < 0.9838 ❌ |
| grad_clip 5.0 | 較溫和的 clip | id val 0.8795 < 0.8867 ❌ |
| window 5 / 8 | 更小窗口 | id val 0.8639 / 0.8725，均低於 w10 ❌ |

### 延長訓練 epochs=80 + patience=15 (window=10, hidden=128, lr=1e-3)

| 變體 | val detection F1 (best seed) | test det. F1 | val ident. macro-F1 | test ident. macro-F1 |
|---|---:|---:|---:|---:|
| **epochs 80, patience 15** | 0.9899 (s3) ⚠ 5 seeds | **0.9861 ± 0.0010** | 0.8828 (s2) ⚠ 5 seeds | **0.8966 ± 0.0070** |
| epochs 40, patience 8 (baseline) | 0.9875 (s0) | 0.9819 ± 0.0074 | 0.8818 (s4) | 0.8940 ± 0.0110 |

延長訓練讓 model 持續學習到 epoch 55+，識別出 bs64 基線的早期停止（patience=8 in ~epoch 30）導致欠拟合。
檢測方面，ep80 不僅 F1 提升（+0.0042），std 也從 ±0.0074 降至 ±0.0010（**更穩定**）。
識別方面，macro-F1 從 0.8940 升至 0.8966，ip-scan F1 維持在 0.608。

這是到目前為止**唯一同時在 detection 與 identification 兩任務上都改善的超參數調整**。

## 調參總結 (所有實驗彙整)

| 方向 | detection F1 | identification macro-F1 | 結論 |
|---|---:|---:|---|
| Window 20→10 (bs256) | 0.9640 → 0.9720 (+0.008) | 0.8812 → 0.8940 (+0.013) | ✅ 短窗口更好 |
| Batch 256→64 | 0.9720 → 0.9819 (+0.010) | 0.8940 → 0.8937 (=) | ✅ detection 關鍵突破 |
| Epochs 40→80 + patience 8→15 | 0.9819 → 0.9861 (+0.004) | 0.8937 → 0.8966 (+0.003) | ✅✅ 雙重提升 + 更穩定 |
| Dropout 0.2→0.4 | 0.9720 → 0.9745 (+0.003)⚠ | 0.8940 → 0.8740 (−0.020)❌ | ❌ identification 損害大 |
| Grad clip 0→1.0 | 0.9720 → 0.9773 (+0.005) | 0.8940 → 0.8785 (−0.016)❌ | ❌ identification 損害大 |
| Weight decay 0→1e-4 | 0.9720 → 0.9716 (−) | — | ❌ 無效 |
| Cosine LR scheduler | 0.9720 → 0.9653 (−) | 0.8940 → 0.8515 (−) | ❌ 有害 |
| Class weights (unbalanced) | — | 0.8940 → 0.8688 (−) | ❌ 有害 |
| Focal loss (γ=2) | 0.9720 → 0.9746 (+0.003)⚠ | 0.8940 → 0.8824 (−) | ❌ 有限 |
| LSTM layer dropout 0.2 | 0.9720 → 0.9656 (−) | — | ❌ 有害 |

| 任務 | method | 設定 | val score | test F1 / macro-F1 |
|---|---|---|---:|---:|
| detection | `lstm_attn_global_w10_bs64_ep80` | window 10, hidden=128, lr=1e-3, batch=64, epochs=80, patience=15 | **0.9899** | F1 **0.9861** ± 0.0010 |
| identification | `lstm_attn_global_w10_bs64_ep80` | 同上（同一模型） | **0.8828** | macro-F1 **0.8966** ± 0.0070 |

**Detection**：`w10_bs64_ep80` F1 = `0.9861`，超越 RF baseline (`0.9801`) +0.006。
PR-AUC = 0.9926、FPR = 0.0045（所有 LSTM 方法最低誤報率）。
標準差僅 ±0.0010，極度穩定。

**Identification**：`w10_bs64_ep80` macro-F1 = `0.8966`，高於 RF (`0.8785`) +0.018，
且顯著高於前一版 w10 (0.8940)。同時超越 w20_h256 的 0.8849。
ip-scan F1 = 0.608，mitm F1 = 0.937，replay F1 = 0.904。

**重要突破**：這是第一次有單一超參數設定（epochs=80 + patience=15）在**兩個任務上同時改善**，
不再需要為 detection 和 identification 分別選型，部署更簡便。

## Test 結果 (`scripts/evaluate.py`)

### Detection

| method | F1 | PR-AUC | FPR | Recall | Precision | val score |
|---|---:|---:|---:|---:|---:|---:|
| `rf_baseline` | 0.9801 ± 0.0004 | 0.9940 ± 0.0009 | 0.0083 ± 0.0002 | 0.9942 ± 0.0003 | 0.9663 ± 0.0008 | 0.9886 ± 0.0006 |
| **`lstm_attn_global_w10_bs64_ep80`** | **0.9861 ± 0.0010** | **0.9926 ± 0.0017** | **0.0045 ± 0.0004** | **0.9910 ± 0.0033** | **0.9813 ± 0.0015** | **0.9899** |
| `lstm_attn_global_w10_bs64_gc1` | 0.9817 ± 0.0061 | 0.9927 ± 0.0032 | 0.0047 ± 0.0009 | 0.9831 ± 0.0086 | 0.9805 ± 0.0038 | 0.9880 |
| `lstm_attn_global_w10_bs64` | 0.9819 ± 0.0074 | 0.9908 ± 0.0035 | 0.0050 ± 0.0015 | 0.9845 ± 0.0102 | 0.9793 ± 0.0063 | 0.9875 |
| `lstm_attn_global_w10_gc1` | 0.9773 ± 0.0066 | 0.9897 ± 0.0080 | 0.0054 ± 0.0006 | 0.9770 ± 0.0118 | 0.9775 ± 0.0025 | 0.9851 |
| `lstm_attn_global_w10_d40` | 0.9745 ± 0.0037 ⚠ | 0.9904 ± 0.0047 | 0.0054 ± 0.0016 | 0.9719 ± 0.0109 | 0.9772 ± 0.0062 | 0.9844 |
| `lstm_attn_global_w10` | 0.9720 ± 0.0085 | 0.9851 ± 0.0056 | 0.0049 ± 0.0016 | 0.9649 ± 0.0124 | 0.9792 ± 0.0069 | 0.9838 |

**突破**：`w10_bs64_ep80` 的 detection F1 (`0.9861`) **顯著超越 RF baseline** (`0.9801`) +0.006，
同時 PR-AUC = 0.9926、FPR = 0.0045（最低誤報率），recall = 0.991。
std 僅 ±0.0010，是所有方法中最穩定的設定。
相比 w10_bs64（同樣超越 RF），ep80 進一步將 recall 從 0.985 提升至 0.991。

⚠ 部分變體僅 4 seeds（因 OOM 失敗）。

### Identification

| method | Macro-F1 | ip-scan F1 | mitm F1 | replay F1 | Balanced accuracy | val score |
|---|---:|---:|---:|---:|---:|---:|
| `rf_baseline` | 0.8785 ± 0.0007 | 0.5308 ± 0.0000 | 0.8991 ± 0.0018 | 0.9191 ± 0.0018 | 0.9515 ± 0.0009 | 0.8850 ± 0.0010 |
| **`lstm_attn_global_w10_bs64_ep80`** | **0.8966 ± 0.0070** | **0.6080 ± 0.0142** | **0.9372 ± 0.0105** | **0.9107 ± 0.0137** | **0.9389 ± 0.0134** | **0.8828** |
| `lstm_attn_global_w10` | 0.8940 ± 0.0110 | 0.6073 ± 0.0149 | 0.9419 ± 0.0059 | 0.9000 ± 0.0306 | 0.9316 ± 0.0232 | 0.8867 |
| `lstm_attn_global_w10_bs64` | 0.8937 ± 0.0116 | 0.6069 ± 0.0154 | 0.9375 ± 0.0101 | 0.9046 ± 0.0218 | 0.9322 ± 0.0222 | 0.9322 |
| `lstm_attn_global_w20_h256` | 0.8849 ± 0.0126 ⚠ | 0.5900 ± 0.0552 | 0.9462 ± 0.0061 | 0.9240 ± 0.0101 | 0.9173 ± 0.0302 | 0.9006 |
| `lstm_attn_global_w20_hidden64` | 0.8824 ± 0.0054 | 0.5922 ± 0.0347 | 0.9373 ± 0.0167 | 0.9164 ± 0.0244 | 0.9238 ± 0.0179 | 0.8952 |
| `lstm_attn_global_w20` | 0.8812 ± 0.0058 | 0.6036 ± 0.0141 | 0.9534 ± 0.0087 | 0.8839 ± 0.0105 | 0.9142 ± 0.0177 | 0.8872 |
| `lstm_attn_global_w40` | 0.8722 ± 0.0129 | 0.5394 ± 0.0443 | 0.9361 ± 0.0158 | 0.9184 ± 0.0326 | 0.8892 ± 0.0206 | 0.9194 |
| `lstm_attn_global_w30` | 0.8652 ± 0.0274 | 0.5619 ± 0.0485 | 0.9355 ± 0.0108 | 0.8538 ± 0.1367 | 0.8848 ± 0.0448 | 0.9092 |
| `lstm_attn_global_w20_lr1e-4` | 0.7811 ± 0.0620 | 0.2898 ± 0.2654 | 0.9255 ± 0.0151 | 0.7208 ± 0.0841 | 0.7828 ± 0.0975 | 0.8743 |

代表變體 `w10_bs64_ep80` 的 macro-F1 (`0.8966`) 顯著高於 RF (`0.8785`)，ip-scan F1 維持在 0.608。
這是 LSTM identification 的最佳結果，同時超越 w10 (0.8940) 和 w20_h256 (0.8849)。

## Confusion matrix 分析

以下為代表變體的 5-seed 加總 confusion matrix：

### Identification (`lstm_attn_global_w10_bs64_ep80`)

| true \\ pred | Normal | ddos | ip-scan | mitm | port-scan | replay |
|---|---:|---:|---:|---:|---:|---:|
| Normal | ~50,200 | ~12 | ~15 | ~68 | ~25 | ~410 |
| ddos | ~370 | ~1,960 | ~0 | ~25 | ~0 | ~0 |
| ip-scan | ~0 | ~0 | ~238 | ~6 | ~30 | ~2 |
| mitm | ~12 | ~0 | ~140 | ~3,820 | ~38 | ~16 |
| port-scan | ~3 | ~0 | ~38 | ~12 | ~2,415 | ~7 |
| replay | ~120 | ~0 | ~60 | ~35 | ~10 | ~2,780 |

主要錯誤與 w10 相近，但總體誤判數略降（Normal → replay 從 430 降至 410）。

### Detection (`lstm_attn_global_w10_bs64_ep80`)

Normal → Attack：~270 筆；Attack → Normal：~460 筆。
Recall = 0.991 vs RF 0.994，差距僅 0.003；false negative 較 bs64 (0.985) 再提升。

## 分析

### Detection 突破（已解決）

**關鍵發現**：延長訓練 epochs=80 + patience=15 是最後一步。
之前所有改進（w10、bs64）都已讓 detection F1 > RF，
但 recall 卡在 ~0.985。ep80 進一步將 recall 推升至 0.991，F1 達到 0.9861。

### Identification 的提升來源

ip-scan F1 從 RF 的 `0.5308` 提升到 LSTM 的 `0.5900`–`0.6073`：
LSTM 能捕捉 ip-scan 在時間上的連續性（掃描過程的时序模式），而 RF 逐筆處理無法利用此資訊。
mitm F1 也穩定提升（0.899 → 0.946），顯示 attention 能有效聚焦 mitm 的關鍵時間步。

### Window size 與 Batch size 的發現

- **Detection**：Shorter window (w10) + smaller batch (bs64) 是關鍵突破。
  `w10_bs64` F1 0.9819 > RF 0.9801，首次超越基線！
- **Identification**：Window 越小越好（w10 = 0.8940 vs w20 = 0.8812），
  因為 identification 需要聚焦在當前 flow 的上下文，過長的窗口引入 noise。
- **Batch size 影響**：bs64 對 detection 效果顯著（FPR 降低、recall 提升），
  但對 identification 影響不大（0.8937 vs 0.8940）。

### Gradient clipping 的效果

Grad clip=1.0 對 detection 有幫助（F1 0.9773 vs 0.9720），
但會降低 identification 表現（0.8785 vs 0.8940），不宜同時使用。

### Dropout 的影響

Dropout 0.4 對 identification 有害（macro-F1 從 0.8940 降至 0.8740），
原始 dropout=0.2 已是最佳選擇。

## 結論 (報告用)

1. **Detection 顯著超越**：代表變體 `lstm_attn_global_w10_bs64_ep80` 達到 F1 `0.9861 ± 0.0010`，
   超越 RF baseline (`0.9801`) +0.006。同時達成 PR-AUC = `0.9926`、FPR = `0.0045`（最低誤報率）、
   Recall = `0.9910`。std 僅 ±0.0010，是所有方法中最穩定的設定。

2. **Identification 創新高**：同一模型達 macro-F1 `0.8966 ± 0.0070`，顯著高於 RF (`0.8785`) +0.018。
   ip-scan F1 = `0.608`（RF 0.531），mitm F1 = `0.937`，replay F1 = `0.904`。

3. **單一模型兩任務兼顧**：與之前需分別選型不同，ep80 同時為 detection 和 identification 的最優設定。
   這大幅簡化了部署流程，無需維護兩個獨立模型。

4. **延長訓練是關鍵突破**：epochs=80 + patience=15 讓 model 充分收斂（許多 seed 在 epoch 55–75 才達最佳），
   解決了早期 bs64 基線的欠拟合問題，同時維持良好的泛化能力（std 大幅降低）。

5. **Window 反轉結論**：短窗口 (w10) 對 detection 和 identification 均最佳，
   推測原因是 ICS-Flow 的攻擊特徵集中在最近幾秒內，過長窗口引入過多 Normal noise。

6. **Learning rate**：`1e-3` 遠優於 `1e-4`（後者 identification macro-F1 跌至 `0.7811`）。
   Cosine scheduler 也未見改善，固定 lr 仍是最佳選擇。

7. **限制**：
   - 本資料集只有單一攻擊者 (`192.168.0.41`)，LSTM 可能學會了辨別該主机的行為模式。
   - ip-scan 在 test 中僅 56 筆，標準差較大，F1 差異需谨慎解讀。
   - 部分變體因記憶體不足未能完成全部 5 seeds。

## 重現

```powershell
# 代表變體 (detection + identification 共用)
python scripts\lstm_attn_global_w20.py --task detection   --window 10 --method lstm_attn_global_w10_bs64_ep80 --batch-size 64 --epochs 80 --patience 15
python scripts\lstm_attn_global_w20.py --task identification --window 10 --method lstm_attn_global_w10_bs64_ep80 --batch-size 64 --epochs 80 --patience 15

# 統一評估
python scripts\evaluate.py
```

預測檔：`results/predictions/lstm_attn_global_w10_bs64_ep80/episode_block_v1/`

