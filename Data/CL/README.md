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
