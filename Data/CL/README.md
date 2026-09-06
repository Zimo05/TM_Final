# Continual Hawkes data generator

`generate_continual_hawkes.py` creates the structured continual-learning Hawkes streams described in the linked strategy. It uses fixed `D=8` event types and exponential bases `beta=[0.5, 1.5]`, generates stable non-negative excitation matrices with `rho(K)<1`, and samples sequences with Ogata thinning.

Install the only required dependency:

```bash
python3 -m pip install numpy
```

Generate the main A→B→C→A recurrence benchmark:

```bash
python3 generate_continual_hawkes.py \
  --output Data/continual_hawkes \
  --benchmark recurrence \
  --seed 7 \
  --train-per-stage 128 \
  --val-per-stage 32 \
  --test-per-stage 32 \
  --seq-len 64
```

Use `--benchmark drift`, `hierarchy`, `transient`, or `all` for the other protocols. For a quick smoke test, use `--train-per-stage 2 --val-per-stage 1 --test-per-stage 1 --anchor-count 2 --seq-len 8`.

Each benchmark contains `task_XX/{train,val,test}.csv` with only `event_times,event_types` JSON-list columns, independent frozen samples under `anchors/`, and oracle-only metadata under `stream_manifest.csv` / `ground_truth_manifest.csv` and `ground_truth/`. Keep the oracle files out of model initialization and use the anchors in read-only evaluation after every task.

## Train the CL memory profile

`run_CL` now passes the DWS routing/alignment profile explicitly instead of
using the generic CLI defaults. The first task runs five offline alignment
epochs with the Multi-Attention sequence summary; tasks 1 and later only
resume the previous Memory checkpoint. If task 0 starts from a root-only
H-tree, alignment is recorded as `skipped_root_only_tree` and training
continues; alignment runs normally whenever internal branches exist. The
CL-specific first experiment uses
`merge_min_replay=16`, `merge_budget_ratio=1.0`, and
`deep_evidence_budget=64`. Its Merge law-distinction penalties default to
`merge_stale_weight=0.2` and `merge_dynamics_weight=0.25`.

```bash
DEVICE=cuda DEVICES=0 PYTHON=/path/to/python \
  bash Data/CL/run_CL.sh all
```

To run a single task after the upstream artifacts are ready:

```bash
DEVICE=cuda DEVICES=0 PYTHON=/path/to/python \
  bash Data/CL/run_CL.sh task 0
DEVICE=cuda DEVICES=0 PYTHON=/path/to/python \
  bash Data/CL/run_CL.sh task 1
```

The CL values can be overridden with environment variables such as
`MERGE_MIN_REPLAY`, `MERGE_BUDGET_RATIO`, `DEEP_EVIDENCE_BUDGET`, or
`FRONTIER_ROUTING_TEMPERATURE`. The two Merge penalties are controlled by
`MERGE_STALE_WEIGHT` and `MERGE_DYNAMICS_WEIGHT`; see
`bash Data/CL/run_CL.sh help` for the full list. The same two variables are
available in `run_HM.sh`.

## Evaluate the continual-learning checkpoints

After one or more checkpoints have been produced, run the CL-specific
evaluation entry point from the repository root:

```bash
bash Data/CL/evaluate_CL.sh \
  --device cuda \
  --eval-batch-size 32 \
  --resume
```

The launcher defaults to `Data/CL/Data`,
`Data/CL/runs/cl_dws_aligned/memory_checkpoints`, and
`Memory/Eval/CL`. It discovers matching `task_XX/test.csv` and
`task_XX.pt` files automatically. The main CL evaluator runs only the
`full_frozen` variant: `task_k.pt` is evaluated on its own test set, the next
task's test set before learning (when available), and every independent
frozen anchor. For each checkpoint, these evaluation sets share one padded GPU
batch pipeline for prefix encoding, routing-static tables, and scalar metric
accumulation; the causal Working Memory recurrence remains sequence-local.
`--eval-batch-size` controls the number of variable-length sequences prepared
per batch. Use `--current-only` for a quick current-task-only evaluation. Add
`--save-event-predictions` only when the large event-level CSV is needed.

By default the evaluator also uses the oracle-only
`ground_truth/regimes.npz` for the Hawkes-law layer. It computes strict-causal
intensity-curve NISE in GPU batches and saves representative total/type plots.
`X_transient` is emitted to `ood_metrics.csv` separately and is excluded from
CLNLL and forgetting. Use `--no-hawkes-law-evaluation` to skip this layer, or
adjust `--intensity-samples` and `--intensity-plot-anchors` for plotting cost.

Aggregate summary figures are generated automatically in `plots/`. They cover
current-task quality, CLNLL/forgetting, the checkpoint-by-anchor NLL heatmap,
stage adaptation, Hawkes intensity NISE, tree/memory growth, and available
special recurrence cases. Use `--no-summary-plots` only when CSV-only output
is desired.

Important output files are `report.md`, `summary.json`, `task_metrics.csv`,
`anchor_metrics.csv`, `continual_summary.csv`, `anchor_nll_matrix.csv`,
`intensity_summary.csv`, `ood_metrics.csv`, and `checkpoint_tree.csv`. The
last file reports the checkpoint's node/leaf counts and memory rows;
`report.md` combines predictive CL, frozen-law retention, Hawkes intensity
NISE, and the generated summary plots. Mechanism ablations and retrieval
diagnostics are intentionally kept out of this main evaluator.
