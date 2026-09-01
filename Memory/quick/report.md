# Hawkes Memory Tree Evaluation

## Predictive quality

- **semantic_only**: NLL/event=2.737764, ACC=0.2577, macro-F1=0.2574, local-time MAE=5.2955
- **full_frozen**: NLL/event=2.737553, ACC=0.2582, macro-F1=0.2578, local-time MAE=5.2955
- **full_online**: NLL/event=2.737548, ACC=0.2582, macro-F1=0.2578, local-time MAE=5.2955

## Memory contribution

- **total_memory_gain**: mean ΔNLL=+0.000211, improved events=47.9%, 95% CI=[0.000152490729949652, 0.0002778518604687773]
- **online_vs_frozen_gain**: mean ΔNLL=+0.000005, improved events=48.6%, 95% CI=[-1.8466205801814795e-06, 1.144736441067205e-05]

## Tree and memory health

- Nodes/leaves: 15/8; max depth: 7.
- Owner top-1 share: 22.6%; effective owners: 7.82.
- Memory rows: 552 (internal=459, leaf=93).
- Label diagnostics: `{'available': True, 'sequence_count': 26, 'owner_purity': 0.3076923076923077, 'nmi': 0.6345721069444085, 'ari': 0.18050234142188165}`.

## Warnings

- No threshold-based health warning was triggered.

> Time MAE/RMSE use the model's documented local-constant-rate approximation.
