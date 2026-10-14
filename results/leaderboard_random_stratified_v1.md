# Leaderboard — split `random_stratified_v1`, labels `NST`

數值為跨 seed 的 mean ± std；`n` 為 seed 數，少於 5 者不列入正式比較。

## detection

| method | n | f1 | pr_auc | fpr | recall | precision | mcc |
|---|---|---|---|---|---|---|---|
| rf_baseline | 5 | 0.9908 ± 0.0006 | 0.9987 ± 0.0002 | 0.0023 ± 0.0002 | 0.9909 ± 0.0009 | 0.9906 ± 0.0007 | 0.9885 ± 0.0007 |

## identification

| method | n | macro_f1 | f1_ip-scan | f1_mitm | f1_replay | balanced_acc | weighted_f1 |
|---|---|---|---|---|---|---|---|
| rf_baseline | 5 | 0.8827 ± 0.0006 | 0.5294 ± 0.0000 | 0.9155 ± 0.0015 | 0.9193 ± 0.0011 | 0.9387 ± 0.0010 | 0.9844 ± 0.0003 |
