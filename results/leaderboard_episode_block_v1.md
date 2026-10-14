# Leaderboard — split `episode_block_v1`, labels `NST`

數值為跨 seed 的 mean ± std；`n` 為 seed 數，少於 5 者不列入正式比較。

## detection

| method | n | f1 | pr_auc | fpr | recall | precision | mcc |
|---|---|---|---|---|---|---|---|
| rf_baseline | 5 | 0.9803 ± 0.0006 | 0.9938 ± 0.0008 | 0.0082 ± 0.0003 | 0.9944 ± 0.0004 | 0.9667 ± 0.0011 | 0.9757 ± 0.0008 |

## identification

| method | n | macro_f1 | f1_ip-scan | f1_mitm | f1_replay | balanced_acc | weighted_f1 |
|---|---|---|---|---|---|---|---|
| rf_baseline | 5 | 0.8801 ± 0.0003 | 0.5308 ± 0.0000 | 0.9030 ± 0.0015 | 0.9231 ± 0.0015 | 0.9529 ± 0.0003 | 0.9813 ± 0.0002 |
