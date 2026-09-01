# Wake/Sleep Training and Inference

## Files

- `Train.py`: backward-compatible training entry point; composes the modular
  pipeline and re-exports the established public API.
- `TrainingComponents.py`: shared configuration, routing utilities, and the
  causal prefix encoder.
- `TrainingLifecycle.py` and `TrainingWakeSupport.py`: trainer construction,
  checkpoint restoration, routing, memory-write, and encoding helpers.
- `TrainingWake.py`, `TrainingLikelihood.py`, and `TrainingObjectives.py`:
  event-scale wake updates and cross-sequence objectives.
- `TrainingSleep.py`, `TrainingLoop.py`, and `TrainingCheckpoint.py`: sleep
  cycles, epoch scheduling, end-to-end training, and checkpoint writing.
- `TrainingTrainer.py` and `TrainingCLI.py`: trainer composition and the
  command-line pipeline.
- `Inference.py`: checkpoint loading, wake-only online inference, delayed
  memory writes, and next-event forecasting.
- `ConstructTree.py`: CSV loading and optional Hawkes cold start.

## Objective mapping

The event-wise wake loss is

\[
L_{wake,t}=L^{pred}_t
+\lambda_{wm}\|\Delta\theta_t^{wm}\|_2^2
+\lambda_{write}\operatorname{ST}[p_t^{write}].
\]

`prediction_nll` is evaluated with the routed semantic parameters, retrieved
episodic residual, and current trainable working-memory residual. The gradient
with respect to that exact working residual is applied to the persistent fast
state immediately. Encoder, Router, semantic/hypernetwork, query/retrieval
parameters are frozen during this pass, and Wake performs **zero global
optimizer steps**.

The write gate has the exact hard forward value
`1[action in {MEMORIZE, QUEUE-SPLIT}]`. Its backward path uses smooth surprise
and novelty threshold gates for objective accounting; it is not reused to
update persistent parameters from the event-level working-memory graph.

Training memory writes are queued at the triggering event and committed only
after the complete write horizon has been observed. Therefore a residual built
from events `k:k+h` cannot be retrieved while predicting any event in that
same window.

Sleep is split into two bounded operators:

```text
Light Sleep (normally every epoch):
  fixed topology
  -> allocate one global replay-evidence budget across active leaves
  -> cluster residuals into a bounded number of directions
  -> validate only a few representatives per direction
  -> absorb coherent, useful directions into leaf semantics
  -> exactly rebase every residual in the changed leaf

Deep Sleep (pressure-triggered):
  Light Sleep first
  -> bounded Split/Merge/Prune evidence
  -> structural transaction through the existing Coordinator
```

Light Sleep does not run Encoder, Router, Retrieval, or a global backward
pass. Both replay evaluation and candidate inspection are bounded: candidate
inspection is at most `light_replay_budget * light_scan_budget_multiplier`.
A cyclic cursor covers a different Bank region on later cycles, while a capped
per-leaf direction index retains residual centroids and support without
reclustering the complete Bank. A leaf update preserves every episodic
effective parameter exactly:

\[
\theta_\ell^+=\theta_\ell^-+\delta_\ell,\qquad
\Delta\theta_r^+=\theta_\ell^-+\Delta\theta_r^--\theta_\ell^+.
\]

Deep Sleep is activated only when prediction/residual pressure, memory
pressure, or structural evidence persists and its minimum interval has
elapsed. Split batches, merge replay, and shared-memory promotion all use
bounded evidence. Merge evaluates all eligible sibling pairs with one flattened
GPU tensor graph: child evidence is a memory-mass likelihood mixture rather
than an oracle maximum. A differentiable keep gate trades prediction gain
against a projected-dual global complexity budget. After its optimizer step,
the detached MAP decision is passed into the no-gradient Coordinator; semantic
distance and embedding distance remain diagnostics and promotion remains a
separate operation for pairs that stay split. Residual-energy pressure is
estimated from the same bounded Light sample. Episodic-memory Prune is also a
Deep-Sleep proposal: all eligible replay queries share the packed GPU retrieval
path, a Concrete survival gate multiplies (but does not replace) entmax
relevance, and a zero-residual null memory prevents normalization from reviving
suppressed rows. Replay NLL, Hawkes intensity-law preservation, a
usage/staleness/retrieval prior, and a projected-dual memory budget optimize
Sleep-local logits. The Coordinator then performs global Top-K commit only on
banks untouched by Merge, leaf prune, Split, or promotion. Leaf-mass prune is
unchanged and remains the only Prune operation that changes tree topology.

Before a split is committed, the two new local node embeddings are optimized
with an initialization fitting loss against the proposed child Hawkes
parameters. The remaining approximation error is then absorbed by the local
semantic offset, preserving exact structural commit semantics.

Between Wake and Sleep, persistent parameters are recomputed from
cross-sequence batches:

\[
\mathcal L_{global}
=\mathcal L_{pred}
+\lambda_{mix}\mathcal L_{mix}
-\lambda_{MI}\left[
H\left(\frac1{|B|}\sum_s\bar r_s\right)
-\frac1{|B|}\sum_s H(\bar r_s)
\right]
+\lambda_{prior}KL\left(
\frac1{|B|}\sum_s\bar r_s\middle\|U_L
\right),
\qquad
\bar r_s=\frac1{T_s}\sum_t r_{s,t}.
\]

The likelihood-aligned term evaluates every leaf before routing mixture:

\[
E_{s\ell}=\frac1{T_s}\sum_t
\ell_{pred}(s,t;\theta_\ell),\qquad
\mathcal L_{mix}=-\frac1{|B|}\sum_s
\log\sum_\ell \bar r_{s\ell}\exp(-E_{s\ell}/\tau_E).
\]

It prevents MI from amplifying an arbitrary random partition: Router receives
a direct signal that a sequence should prefer the leaf whose Hawkes dynamics
explain it better.

Prediction gradients are accumulated one sequence at a time without stepping;
the parameters stay fixed until the whole cross-sequence batch is complete.
This is mathematically one batch gradient while avoiding retention of every
sequence graph in GPU memory.

The resulting timescales are:

```text
event:          working-memory update only
sequence batch: one persistent-parameter optimizer step
sleep:          replay consolidation and structural transaction
```

## Python API

```python
from Train.Train import CausalPrefixEncoder, MemoryTreeTrainer

encoder = CausalPrefixEncoder(num_event_types=D, z_dim=tree.z_dim)
trainer = MemoryTreeTrainer(tree, hawkes, encoder)
history = trainer.train(dataset)
```

### Attention Encoder integration (H_tree only)

Offline Multi-Attention Phase 1–3 produces static node embeddings
`H_tree_refined` (`AttenEncoderMain_v1.py --node_only`). Those embeddings
initialize MESH slow semantic parameters `u_n` (`tree.node_emb`). Online
wake/sleep uses `CausalPrefixEncoder` for `z_t = f_phi(prefix)`. Runtime does
not consume the offline `D_tree`; routing instead compares the causal state
with semantic node embeddings through a shared `Compat(z_t, u_n)` scorer.

```python
from AttentionEncoderAdapter import initialize_tree_from_h_tree_file
from Train.Train import CausalPrefixEncoder, MemoryTreeTrainer

initialize_tree_from_h_tree_file(
    tree,
    "Data/tree_20/h_tree.pt",
    synchronize_topology=True,
)
# Re-fit additive path offsets after replacing node embeddings. alpha=0.1
# keeps 90% of the stable Hawkes cold start and exposes 10% of Hyper(u_n).
tree.initialize_semantics_from_hawkes(hawkes, semantic_blend=0.1)
tree.initialize_router_weights(gain=0.05)
# Require tree.node_dim == H_tree_refined.size(-1) (encoder d_model).

encoder = CausalPrefixEncoder(num_event_types=D, z_dim=tree.z_dim)
trainer = MemoryTreeTrainer(tree, hawkes, encoder)
history = trainer.train(dataset)
```

CLI:

```bash
python -m Train.Train \
  --data-path data.csv \
  --node-dim 128 \
  --h-tree ../Data/tree_20/h_tree.pt \
  --semantic-blend 0.1 \
  --router-prior-mode compat \
  --routing-mode frontier \
  --frontier-budget 4 \
  --frontier-routing-temperature 1.5 \
  --frontier-exploration 0.05 \
  --route-balance-weight 0.05 \
  --light-replay-budget 32 \
  --deep-min-interval 3 \
  --deep-evidence-budget 32 \
  --epochs 20 \
  --checkpoint checkpoints/memory_tree.pt
```

Reuse an existing Hawkes cold-start checkpoint and validate only the semantic
bridge (no Wake/Sleep updates and no Memory checkpoint write):

```bash
python -m Train.Train \
  --data-path ../Data/tree_13/hawkes_dataset_13.csv \
  --node-dim 128 \
  --h-tree ../Data/tree_13/h_tree_13.pt \
  --hawkes-checkpoint Checkpoints/hawkes_backbone_tree13.pt \
  --semantic-smoke-only \
  --max-sequences 1 \
  --max-events-per-sequence 32 \
  --device cuda
```

`--hawkes-checkpoint` takes precedence over `--cold-start-epochs`.

With `--h-tree`, the CLI starts from depth 0 and reconstructs the exact binary
topology stored in the encoder's `node_ids`; `--node-dim` must match the saved
`d_model`. For a bounded end-to-end smoke circle, add for example
`--max-sequences 1 --max-events-per-sequence 32 --epochs 1`.

The default runtime construction is active-frontier routing with
frontier-only retrieval. It uses the same interaction features as the
Attention Encoder:

```text
Compat(z_t, u_n) =
    MLP([u_n, W_z z_t, u_n * W_z z_t, |u_n - W_z z_t|]).
```

Each prediction starts with `{root}` and produces a padded fixed-width
frontier (`K_min=2`, `K_max=4`). Search uses semantic compatibility, a neutral
descendant-target-mass prior, historical refinement gain, confidence, and
compute cost. It does not use targets, Hawkes energy, retrieval, or a full-leaf
projection. Retrieval is one packed read of the actual frontier path union.

After the target arrives, training forms `q+ ∝ m- exp(-E/tau)` only on the
computed frontier. This posterior supervises the likelihood mixture, expanded
local branches, prototypes, retrieval credit, and coarse-to-fine memory owner.
Ambiguous ownership uses the credible-set LCA; confident singleton ownership
uses that frontier node. No coarse mass is copied into unseen leaves.

Semantic startup is a controllable raw-parameter blend:

```text
theta_n = (1 - semantic_blend) * theta_cold
          + semantic_blend * Hyper(u_n).
```

The default `semantic_blend=0.1` cancels exact Hawkes uniformity while retaining
a stable cold-start anchor. `semantic_blend=0` restores the legacy exact start.
The blend target is realized through the existing additive path offsets; offset
support for exact split/merge/rebase operations is not removed. Afterwards only
leaves receive an additional zero-mean perturbation
(`leaf_symmetry_scale=0.01`). That perturbation is added to each leaf's own
blended target, so it no longer erases pre-existing semantic differences.
Startup logs report semantic deviation, routing prior mode, perturbation, and
leaf spectral radius, and reject an unstable initialization.

Persistent parameters are trained from complete sequences with

```text
mean event prediction NLL
+ route_mix_weight * likelihood-aligned frontier mixture NLL
+ route_posterior_weight * KL(q+ || m-)
+ route_distill_weight * local branch posterior distillation
- route_mi_weight * local branch MI
+ route_balance_weight * KL(local marginal || neutral structural prior).
```

The local MI term rewards high branch marginal entropy and low conditional
entropy without comparing heterogeneous frontier slots, so it does not make every sequence uniformly
ambiguous. Router parameters now default to the normal learning rate because
they no longer receive event-wise optimizer steps. Leaf pruning remains
disabled for the first `prune_warmup_epochs` (default 10). Relevant controls:

```bash
--route-mix-weight 1.0 \
--route-posterior-weight 1.0 \
--route-distill-weight 1.0 \
--route-energy-temperature 1.0 \
--route-mi-weight 0.2 \
--route-balance-weight 0.05 \
--route-balance-batch-size 65 \
--frontier-min-experts 2 \
--frontier-budget 4 \
--frontier-routing-temperature 1.5 \
--frontier-exploration 0.05 \
--frontier-confidence-weight 0.25 \
--frontier-compute-cost 0.05 \
--frontier-posterior-temperature 1.0 \
--frontier-credible-mass 0.90 \
--frontier-owner-confidence 0.80 \
--max-writes-per-sequence 8 \
--router-lr-scale 1.0 \
--router-init-gain 0.05 \
--semantic-blend 0.1 \
--leaf-symmetry-scale 0.01 \
--route-encoder-warmup-epochs 2 \
--route-encoder-grad-scale 0.1 \
--prune-warmup-epochs 10 \
--memory-prune-budget-ratio 0.7 \
--memory-prune-steps 12 \
--memory-prune-lr 0.01 \
--memory-prune-temperature-start 1.0 \
--memory-prune-temperature-end 0.3 \
--memory-prune-dynamics-weight 0.1 \
--memory-prune-prior-kl-weight 0.01 \
--memory-prune-dual-lr 0.01 \
--memory-prune-null-logit 0.0 \
--memory-prune-min-replay 2
```

For the first two epochs, Router MI sees detached Encoder states. Afterwards
the forward route stays identical but only `0.1` of its MI gradient enters the
Encoder; the compatibility scorer and semantic node embeddings retain their
route gradients. Encoder always continues to learn from the cross-sequence
prediction and likelihood-mixture objectives.

`--route-balance-max-steps` and `--route-balance-target-kl` remain accepted
only so old shell commands do not fail; they no longer cause repeated updates.

Epoch logs report local branch conditional/marginal entropy and MI, neutral
prior KL, frontier posterior KL, branch distillation, and `global_steps`.
`owner_hard` contains one credible-set/LCA owner per sequence;
`mem_assign` counts the actual memory owner for every event.
`mix` and `mi_enc_scale` expose likelihood alignment and staged
MI-to-Encoder gradient.
`frontier=K/visited/branches` reports the mean active expert count, unique
path-union nodes read, and locally scored branch count per event.
`wake_actions=A/R/M/Q` reports ASSIMILATE, RETRIEVE, MEMORIZE, and QUEUE_SPLIT;
`sleep_actions` is a separate list of committed topology/memory transactions.
`novelty` and `max_sim` expose why hard writes did or did not occur.
`prune=off/on` exposes the warm-up.

Generate `h_tree.pt` with:

```bash
python MultiAttentionEncoder/AttentionEncoder/AttenEncoderMain_v1.py \
  --thp_json ... --tree_csv ... --checkpoint ... \
  --output Data/tree_20/h_tree.pt --node_only
```

Checkpoints with `encoder_config.kind == "attention_memory"` are rejected;
retrain with `CausalPrefixEncoder`. Split-grown leaves without an Attention
counterpart keep Memory's own node-embedding init (sleep fit / random).

The prefix encoder sees only `events[:event_index]`.

## Parameter-independent sequence cache

Training prepares three causal caches once per input sequence and reuses them
across events and epochs:

- event time features `[log(1 + t_k), log(1 + delta_t_k)]`;
- Hawkes strict-history decay statistics `[K, D, M]`;
- Hawkes interval-integral statistics `[K, D, M]`.

The Hawkes caches depend only on fixed event times/types, event vocabulary, and
the fixed decay basis. They remain exact while semantic, episodic, working,
router, query, and encoder parameters change. Cached event NLL therefore keeps
the same value and mu/W gradient as direct history scanning. GRU prefix
embeddings, routing, retrieval, and effective parameters are deliberately not
cached because their learned inputs change during online updates.

When training on CUDA, the raw dataset and these parameter-independent caches
are now moved to CUDA before cache construction and remain resident there for
the complete `train()` call. The cache log includes `device=cuda`. CSV parsing,
dynamic controller decisions, topology transactions, and checkpoint I/O remain
host-side by design.

`CausalPrefixEncoder.forward_all_prefix()` runs one GRU over a sequence and
right-shifts its outputs so row `k` remains exactly conditioned on
`events[:k]`. Wake precomputes these frozen prefix states and their routes once,
then retains the original sequential working-memory/retrieval/write loop.
Global training batches every event in a sequence through Tree routing,
episodic retrieval, effective Hawkes parameters, and leaf likelihood. Its
likelihood mixture and optimizer-step normalization are unchanged.

Sequence-level MI uses an exact route-value vector-Jacobian product: the batch
MI derivative is injected into the same routing graph used by the global
prediction objective. Router/node gradients remain full-strength, while only
the MI gradient entering Encoder follows the existing warm-up/scale setting.
Metric tensors stay on device until the end of a sequence or global batch.
Hard leaf sampling and hard Memory actions still require event-level host
synchronization because they causally determine subsequent memory state.

Episodic windows retain their full Hawkes prefix for bounded Light gain checks
and Deep structural evidence. At training start the dataset cache construction
is reported separately:

```text
[Cache] sequences=65 events=4160 built=65 device=cuda time=...
```

Every dataset item must contain:

```python
{
    "times": FloatTensor[K],
    "types": LongTensor[K],
    "T": optional_scalar_tensor,
}
```

## Command-line training

```bash
python -m Train.Train \
  --data-path data.csv \
  --epochs 20 \
  --cold-start-epochs 5 \
  --checkpoint checkpoints/memory_tree.pt \
  --num-basis 2 \
  --decays 0.5 1.5
```

After training completes, `Train.py` automatically writes two diagnostics
artifacts beside the checkpoint:

- `<checkpoint_stem>_training_metrics.json`: compact per-epoch scalar log;
- `<checkpoint_stem>_training_curves.png`: one 4-by-2 figure covering
  prediction, routing specialization, Router supervision, assignment
  concentration, memory decisions, frontier/topology, and phase runtime.

Use `--training-metrics-path` and `--training-plot-path` to select explicit
locations, or `--no-training-plots` to disable both artifacts. Resumed runs
plot the complete restored history plus the newly trained epochs.

Resume a saved wake/sleep trainer for additional epochs:

```bash
python -m Train.Train \
  --data-path data.csv \
  --resume checkpoints/memory_tree.pt \
  --epochs 10 \
  --checkpoint checkpoints/memory_tree_resumed.pt
```

Format-version 3 checkpoints restore dynamic topology and memory contents,
optimizer state/groups by stable parameter name, dormant split-module state,
split queues, history, completed epoch, and RNG state.

## Inference

```python
from Train.Inference import MemoryTreeInference

inference = MemoryTreeInference.from_checkpoint(
    "checkpoints/memory_tree.pt",
)
online_result = inference.run_sequence(sequence)
next_event = inference.predict_next_event(sequence)
```

At inference time:

- no sleep objective is optimized;
- no topology Split/Merge/Prune is committed;
- routing, episodic retrieval, usage updates, and working-memory adaptation are
  active;
- optional memory writes are delayed until the configured future horizon has
  actually arrived, so online inference does not inspect future events;
- `predict_next_event()` returns the current intensity type distribution and a
  deterministic locally constant-rate expected time. It is not an exact
  Hawkes sample.

Command line:

```bash
python -m Train.Inference \
  --checkpoint checkpoints/memory_tree.pt \
  --times 0.1 0.4 0.9 \
  --types 0 1 0
```

Use `--no-write` for evaluation that must not mutate episodic memory.

## Tests

```bash
/Users/zimoshen/opt/miniconda3/envs/memTree/bin/python -m unittest discover -v
```
