# Hawkes Memory Tree Evaluation

Evaluation regime: **transductive**.

## Predictive quality

- **full_frozen**: NLL/event=2.640401, ACC=0.2857, macro-F1=0.2821, local-time MAE=5.0244
- **no_episodic**: NLL/event=2.640429, ACC=0.2853, macro-F1=0.2818, local-time MAE=5.0239
- **no_working**: NLL/event=2.640642, ACC=0.2849, macro-F1=0.2815, local-time MAE=5.0245
- **semantic_only**: NLL/event=2.640661, ACC=0.2849, macro-F1=0.2815, local-time MAE=5.0240
- **full_online**: NLL/event=2.640400, ACC=0.2857, macro-F1=0.2821, local-time MAE=5.0244

## Memory contribution

- **episodic_gain**: mean ΔNLL=+0.000028, improved events=40.5%, 95% CI=[1.9098673810196274e-05, 3.704691976856659e-05]
- **working_gain**: mean ΔNLL=+0.000241, improved events=47.5%, 95% CI=[0.00018710073747891276, 0.0002982961952083689]
- **total_memory_gain**: mean ΔNLL=+0.000259, improved events=47.7%, 95% CI=[0.00020566330298178774, 0.0003132684207658338]
- **online_vs_frozen_gain**: mean ΔNLL=+0.000001, improved events=46.6%, 95% CI=[-4.0329818148165945e-06, 7.2993861977010965e-06]

## Controller utility diagnostics

- **retrieve**: Spearman=0.13643921090766226, ROC-AUC=0.8189065142120944, PR-AUC=0.594457762491272, regret/event=0.000125.
- **adapt**: Spearman=0.31505777878375496, ROC-AUC=0.6125735811050554, PR-AUC=0.49306597509308664, regret/event=0.000420.
- **write**: Spearman=0.38753490919818046, ROC-AUC=0.834214565921883, PR-AUC=0.9960462338000696, regret/event=0.005958.
  Write ranking: pair-acc=0.7651, NDCG@4=0.6485, Top-4=0.044139, random-Top-4=0.030658, regret/sequence=0.024145.
- **split**: unavailable (No Sleep-evaluated Split labels in checkpoint replay.).
- Research thresholds overall: **FAIL**; details: `{'mean_writes_le_2': False, 'write_budget_utilization_le_0_5': False, 'memory_gain_target': True, 'adapt_spearman_ge_0_35': False, 'retrieve_spearman_gt_0_10': True, 'write_roc_auc_gt_0_60': True, 'write_ranking_spearman_gt_0_10': True, 'write_pairwise_accuracy_gt_0_55': True, 'write_top4_uplift_positive': True, 'write_ranking_better_than_baseline': True, 'write_positive_negative_ge_20': True, 'online_harmful_fraction_lt_0_45': False, 'full_online_gain_gt_0': False, 'frozen_acc_macro_f1_not_below_v4': None, 'frozen_controller_output_unchanged': None, 'retrieve_policy_unchanged': None, 'adapt_policy_unchanged': None}`.

## Memory call-chain diagnostics

- Read/nonempty/owner-path coverage: 100.0% / 100.0% / 100.0%.
- Raw→gated residual ratio: 0.4468; mean retrieve gate: 0.4464.
- Write funnel: argmax=6, candidate=203, gate-pass=2720, priority-pass=0, window-complete=0, accepted=0.
- Accepted-write reuse/beneficial rate: 0.0% / 0.0%.
- Retrieval/NLL correlations: `{'alpha_mass': {'pearson': 0.016560406152546607, 'spearman': -0.022894941838289395}, 'alpha_per_visited_node': {'pearson': -0.01859832215594017, 'spearman': -0.04106396463749806}, 'similarity': {'pearson': 0.03418225933721897, 'spearman': -0.08557858228362765}, 'raw_residual_norm': {'pearson': 0.03091512171210421, 'spearman': -0.04420752767227813}, 'gated_residual_norm': {'pearson': 0.3606075315779416, 'spearman': 0.08590556523511489}, 'retrieve_gate': {'pearson': 0.39432290569107614, 'spearman': 0.13643921090766226}}`.

## Tree and memory health

- Nodes/leaves: 23/12; max depth: 5.
- Owner top-1 share: 24.6%; effective owners: 12.93.
- Memory rows: 2648 (internal=1408, leaf=1240).
- Label diagnostics: `{'available': True, 'sequence_count': 34, 'owner_purity': 0.14705882352941177, 'nmi': 0.3172834487555334, 'ari': 0.026737422185937342}`.

## Warnings

- No threshold-based health warning was triggered.

> Time MAE/RMSE use the model's documented local-constant-rate approximation.
