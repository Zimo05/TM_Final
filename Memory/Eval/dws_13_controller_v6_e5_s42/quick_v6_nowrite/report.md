# Hawkes Memory Tree Evaluation

## Predictive quality

- **semantic_only**: NLL/event=2.743964, ACC=0.2572, macro-F1=0.2545, local-time MAE=5.1935
- **no_episodic**: NLL/event=2.743897, ACC=0.2562, macro-F1=0.2535, local-time MAE=5.1934
- **no_working**: NLL/event=2.743889, ACC=0.2572, macro-F1=0.2545, local-time MAE=5.1933
- **full_frozen**: NLL/event=2.743822, ACC=0.2562, macro-F1=0.2535, local-time MAE=5.1932
- **full_online**: NLL/event=2.743822, ACC=0.2562, macro-F1=0.2535, local-time MAE=5.1932

## Memory contribution

- **episodic_gain**: mean ΔNLL=+0.000075, improved events=11.6%, 95% CI=[5.404964590875002e-05, 9.592913198643005e-05]
- **working_gain**: mean ΔNLL=+0.000067, improved events=34.1%, 95% CI=[1.9629720526819047e-05, 0.00011565349996089935]
- **total_memory_gain**: mean ΔNLL=+0.000142, improved events=38.8%, 95% CI=[9.116775606973814e-05, 0.00019170728718073896]
- **online_vs_frozen_gain**: mean ΔNLL=+0.000000, improved events=0.1%, 95% CI=[0.0, 1.8339890700120193e-09]

## Controller utility diagnostics

- **retrieve**: Spearman=0.2051723718934797, ROC-AUC=0.9742884177190747, PR-AUC=0.6306910695976378, regret/event=0.000023.
- **adapt**: Spearman=0.42127501429982267, ROC-AUC=0.8141840024918774, PR-AUC=0.5264422734028615, regret/event=0.000108.
- **write**: Spearman=0.031513820969078815, ROC-AUC=0.47595651560073415, PR-AUC=0.9440760949975854, regret/event=0.555772.
- **split**: unavailable (No Sleep-evaluated Split labels in checkpoint replay.).
- Research thresholds overall: **FAIL**; details: `{'mean_writes_le_2': True, 'write_budget_utilization_le_0_5': True, 'memory_gain_target': False, 'adapt_spearman_ge_0_35': True, 'retrieve_spearman_gt_0_10': True, 'write_roc_auc_gt_0_60': False, 'write_positive_negative_ge_20': True, 'online_harmful_fraction_lt_0_45': True, 'full_online_gain_gt_0': True, 'frozen_acc_macro_f1_not_below_v4': None}`.

## Tree and memory health

- Nodes/leaves: 15/8; max depth: 7.
- Owner top-1 share: 27.1%; effective owners: 6.37.
- Memory rows: 494 (internal=357, leaf=137).
- Label diagnostics: `{'available': True, 'sequence_count': 26, 'owner_purity': 0.3076923076923077, 'nmi': 0.654107474531054, 'ari': 0.19719583898688375}`.

## Warnings

- No threshold-based health warning was triggered.

> Time MAE/RMSE use the model's documented local-constant-rate approximation.
