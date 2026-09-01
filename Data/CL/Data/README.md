# Continual Hawkes benchmark

Benchmark: `recurrence`
Event dimension: `8`; exponential decay bases: `[0.5, 1.5]`

Model-facing files contain only `event_times,event_types`; do not use the oracle manifest for initialization.
Initialize the model on `task_00`, train each later task while retaining model/tree/memory/optimizer state, and evaluate all frozen anchors after every task.

## Stage schedule

| task_id | stage | regime mixture | shift | recurrence_of |
|---:|---|---|---|---|
| 0 | A_1_initial | A_1:1 | initial |  |
| 1 | B_1_novel | B_1:1 | novel |  |
| 2 | C_1_novel | C_1:1 | novel |  |
| 3 | A_1_exact_recurrence | A_1:1 | exact_recurrence | A_1 |
| 4 | D_1_novel | D_1:1 | novel |  |
| 5 | B_prime_1_near_recurrence | B_prime_1:1 | near_recurrence | B_1 |
| 6 | A_2_specialization | A_2:1 | specialization | A_1 |
| 7 | E_1_novel | E_1:1 | novel |  |
| 8 | A_1_long_gap_recurrence | A_1:1 | long_gap_recurrence | A_1 |
| 9 | E_B_mixture | E_1:0.7, B_1:0.3 | mixture |  |

`stream_manifest.csv` and `ground_truth/regimes.*` are oracle-only artifacts for evaluation and plotting.
Frozen anchor banks under `anchors/` are newly sampled from the same law, not copies of stream sequences.
