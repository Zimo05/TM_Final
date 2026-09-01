#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_PARENT="$(cd "$PROJECT_ROOT/.." && pwd)"
cd "$PROJECT_ROOT"

MEMTREE_PYTHON="/Users/zimoshen/opt/miniconda3/envs/memTree/bin/python"
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -x "$MEMTREE_PYTHON" ]]; then
  PYTHON_BIN="$MEMTREE_PYTHON"
else
  PYTHON_BIN="python"
fi
DEVICE_IDS="${DEVICES:-0}"

DATA_ROOT="${DATA_ROOT:-$PROJECT_PARENT/HawkesMemory_wfy/Data/tree_17}"
DATA_PATH="${DATA_PATH:-$DATA_ROOT/17Cluster/THP_17.json}"
SUMMARY_CSV="${SUMMARY_CSV:-$DATA_ROOT/sequence_summary.csv}"
TREE_CSV="${TREE_CSV:-$DATA_ROOT/tree_node_sequences.csv}"
HAWKES_DATA="${HAWKES_DATA:-$DATA_ROOT/hawkes_dataset_17.csv}"

THP_OUTPUT_DIR="$DATA_ROOT/thp_checkpoints"
THP_CHECKPOINT="$THP_OUTPUT_DIR/checkpoint_best.pt"
THP_TRAIN_LOG="$DATA_ROOT/thp_train.log"
ENCODED_OUTPUT="$DATA_ROOT/17Cluster/thp_encoded_17.pt"
ATTENTION_WEIGHTS="$DATA_ROOT/encoder_weights_17.pt"
H_TREE_OUTPUT="${H_TREE_OUTPUT:-$DATA_ROOT/h_tree_one_circle.pt}"

RUN_NAME="${RUN_NAME:-dws_17_08281630}"
EPOCHS="${EPOCHS:-10}"
SPLIT_SEED="${SPLIT_SEED:-42}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-$DATA_ROOT/splits/memory_seed${SPLIT_SEED}.json}"
BASE_CONTROLLER_CHECKPOINT="${BASE_CONTROLLER_CHECKPOINT:-$PROJECT_ROOT/Memory/Checkpoints/dws_17_controller_v4_best.pt}"
CONTROLLER_VERSION="${CONTROLLER_VERSION:-6}"
CONTROLLER_HEADS="${CONTROLLER_HEADS:-adapt,retrieve,write}"
MEMORY_CHECKPOINT="$PROJECT_ROOT/Memory/Checkpoints/${RUN_NAME}_last.pt"
MEMORY_BEST_CHECKPOINT="$PROJECT_ROOT/Memory/Checkpoints/${RUN_NAME}_best.pt"
VALIDATION_HISTORY="$PROJECT_ROOT/Memory/Checkpoints/${RUN_NAME}_validation_history.json"
CONTROLLER_DIAGNOSTICS="$PROJECT_ROOT/Memory/Checkpoints/${RUN_NAME}_controller_diagnostics.json"
RECALIBRATED_CHECKPOINT="$PROJECT_ROOT/Memory/Checkpoints/${RUN_NAME}_recalibrated.pt"
ROLLOUT_CALIBRATED_CHECKPOINT="$PROJECT_ROOT/Memory/Checkpoints/${RUN_NAME}_rollout_calibrated.pt"
QUICK_EVAL_DIR="$PROJECT_ROOT/Memory/Eval/${RUN_NAME}/quick"
FULL_EVAL_DIR="$PROJECT_ROOT/Memory/Eval/${RUN_NAME}/full"
MEMORY_LOG="$PROJECT_ROOT/Memory/Logs/DWS/17/${RUN_NAME}.log"
MEMORY_PID_FILE="$PROJECT_ROOT/Memory/Logs/DWS/17/${RUN_NAME}.pid"
# Put repository-level routing packages before the legacy Memory namespace;
# otherwise Memory/Routing_Retrieval (a compatibility namespace) can shadow
# the actual server/local implementation under the project root.
MEMORY_PYTHONPATH="$PROJECT_ROOT:$PROJECT_PARENT:$PROJECT_ROOT/Memory"

mkdir -p \
  "$THP_OUTPUT_DIR" \
  "$(dirname "$ENCODED_OUTPUT")" \
  "$(dirname "$MEMORY_CHECKPOINT")" \
  "$(dirname "$MEMORY_LOG")"

usage() {
  cat <<'EOF'
Usage:
  ./run_HM.sh train-thp        Train THP from scratch (foreground)
  ./run_HM.sh encode           Encode DWS sequences with trained THP (foreground)
  ./run_HM.sh train-attention  Train Attention Encoder (foreground)
  ./run_HM.sh build-h-tree     Generate Data/tree_17/h_tree_17.pt (foreground)
  ./run_HM.sh build-strict-baseline Build leakage-free upstream artifacts in strict_seed${SPLIT_SEED}
  ./run_HM.sh memory           Start Memory Tree training with nohup (background)
  ./run_HM.sh prepare-memory-split  Create the fixed DWS17 train/validation/test split
  ./run_HM.sh memory-controller     Train Controller v4 with integrated Router/Sleep
  ./run_HM.sh controller-finetune   Warm-start strict Controller-only v5/v6 training
  ./run_HM.sh controller-write-rank-finetune  Run Write-only ranking fine-tuning
  ./run_HM.sh inspect-checkpoint    Show Controller/Router/Sleep checkpoint identity
  ./run_HM.sh recalibrate-controller  Jointly recalibrate Controller thresholds
  ./run_HM.sh calibrate-write-rollout Calibrate Write using validation rollouts
  ./run_HM.sh evaluate-controller-quick Evaluate two test sequences per DWS17 cluster
  ./run_HM.sh evaluate-controller-full  Evaluate the complete DWS17 test split
  ./run_HM.sh all              Run the first four stages, then start Memory in background
  ./run_HM.sh status           Show Memory background-process status
  ./run_HM.sh logs             Follow the Memory training log
  ./run_HM.sh stop             Stop the Memory background process

Optional environment overrides:
  RUN_NAME=name EPOCHS=10 PYTHON=/path/to/python DEVICES=0 ./run_HM.sh <action>
  BASE_CONTROLLER_CHECKPOINT=/path/model.pt CONTROLLER_VERSION=6 ./run_HM.sh controller-finetune
EOF
}

require_file() {
  local path="$1"
  local producer="${2:-}"
  if [[ ! -s "$path" ]]; then
    echo "[Error] Required file is missing or empty: $path" >&2
    if [[ -n "$producer" ]]; then
      echo "[Hint] Run: ./run_HM.sh $producer" >&2
    fi
    exit 1
  fi
}

require_python_runtime() {
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[Error] Python executable was not found: $PYTHON_BIN" >&2
    echo "[Hint] Set PYTHON=/path/to/the/memTree/python executable." >&2
    exit 1
  fi
  if ! "$PYTHON_BIN" -c 'import torch' >/dev/null 2>&1; then
    echo "[Error] PyTorch is unavailable in: $PYTHON_BIN" >&2
    echo "[Hint] Activate the memTree environment or set PYTHON explicitly." >&2
    exit 1
  fi
}

require_memory_runtime() {
  require_python_runtime
  if ! PYTHONPATH="$MEMORY_PYTHONPATH" "$PYTHON_BIN" -c \
      'import LatentHawkesTree' >/dev/null 2>&1; then
    echo "[Error] LatentHawkesTree and its routing package are not importable." >&2
    echo "[Hint] Expected Routing_Retrieval_Investigation or Routing_Retrieval." >&2
    echo "[Hint] Checked PYTHONPATH: $MEMORY_PYTHONPATH" >&2
    exit 1
  fi
}

memory_cli_help() {
  PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$MEMORY_PYTHONPATH" \
    "$PYTHON_BIN" -u -B -m Train.Train --help 2>&1
}

run_encoder_stage() {
  local action="$1"
  local split_manifest_arg=""
  local split_data_arg=""
  if [[ "${STRICT_UPSTREAM:-0}" == "1" ]]; then
    split_manifest_arg="$SPLIT_MANIFEST"
    split_data_arg="$HAWKES_DATA"
  fi
  require_python_runtime
  env \
    PYTHON="$PYTHON_BIN" \
    DEVICES="$DEVICE_IDS" \
    DATA_PATH="$DATA_PATH" \
    OUTPUT_DIR="$THP_OUTPUT_DIR" \
    TRAIN_LOG="$THP_TRAIN_LOG" \
    CHECKPOINT="$THP_CHECKPOINT" \
    ENCODED_OUTPUT="$ENCODED_OUTPUT" \
    SUMMARY_CSV="$SUMMARY_CSV" \
    TREE_CSV="$TREE_CSV" \
    FINAL_OUTPUT="$H_TREE_OUTPUT" \
    ATTENTION_WEIGHTS="$ATTENTION_WEIGHTS" \
    SPLIT_MANIFEST="$split_manifest_arg" \
    SPLIT_DATA_PATH="$split_data_arg" \
    bash "$PROJECT_ROOT/MultiAttentionEncoder/run.sh" "$action"
}

build_strict_baseline() (
  local strict_root="$DATA_ROOT/strict_seed${SPLIT_SEED}"
  SPLIT_MANIFEST="$strict_root/split_manifest.json"
  THP_OUTPUT_DIR="$strict_root/thp_checkpoints"
  THP_CHECKPOINT="$THP_OUTPUT_DIR/checkpoint_best.pt"
  THP_TRAIN_LOG="$strict_root/thp_train.log"
  ENCODED_OUTPUT="$strict_root/thp_encoded.pt"
  ATTENTION_WEIGHTS="$strict_root/encoder_weights.pt"
  H_TREE_OUTPUT="$strict_root/h_tree.pt"
  STRICT_UPSTREAM=1
  mkdir -p "$strict_root" "$THP_OUTPUT_DIR"
  echo "[Strict baseline] isolated output: $strict_root"
  prepare_memory_split
  train_thp
  encode_sequences
  train_attention
  build_h_tree
  echo "[Done] strict_inductive H-tree: $H_TREE_OUTPUT"
)

train_thp() {
  require_file "$DATA_PATH"
  echo "[HM 1/5] Training THP on DWS 17-cluster data"
  run_encoder_stage train
  require_file "$THP_CHECKPOINT" "train-thp"
  echo "[Done] THP checkpoint: $THP_CHECKPOINT"
}

encode_sequences() {
  require_file "$DATA_PATH"
  require_file "$THP_CHECKPOINT" "train-thp"
  echo "[HM 2/5] Encoding DWS sequences"
  run_encoder_stage encode
  require_file "$ENCODED_OUTPUT" "encode"
  echo "[Done] Encoded sequences: $ENCODED_OUTPUT"
}

train_attention() {
  require_file "$DATA_PATH"
  require_file "$SUMMARY_CSV"
  require_file "$THP_CHECKPOINT" "train-thp"
  echo "[HM 3/5] Training Attention Encoder"
  run_encoder_stage train_attention
  require_file "$ATTENTION_WEIGHTS" "train-attention"
  echo "[Done] Attention weights: $ATTENTION_WEIGHTS"
}

build_h_tree() {
  require_file "$DATA_PATH"
  require_file "$SUMMARY_CSV"
  require_file "$THP_CHECKPOINT" "train-thp"
  require_file "$ATTENTION_WEIGHTS" "train-attention"
  echo "[HM 4/5] Building 17-cluster H-tree"
  run_encoder_stage final_encode
  require_file "$H_TREE_OUTPUT" "build-h-tree"
  echo "[Done] H-tree: $H_TREE_OUTPUT"
}

memory_is_running() {
  [[ -s "$MEMORY_PID_FILE" ]] || return 1
  local pid
  pid="$(<"$MEMORY_PID_FILE")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

start_memory() {
  local controller_mode="${1:-0}"
  require_memory_runtime
  require_file "$HAWKES_DATA"
  require_file "$SUMMARY_CSV"
  require_file "$H_TREE_OUTPUT" "build-h-tree"
  if [[ "$controller_mode" != "0" ]]; then
    require_file "$SPLIT_MANIFEST" "prepare-memory-split"
  fi
  if [[ "$controller_mode" == "2" || "$controller_mode" == "3" ]]; then
    require_file "$BASE_CONTROLLER_CHECKPOINT"
  fi

  if memory_is_running; then
    echo "[Error] Memory training is already running with PID $(<"$MEMORY_PID_FILE")." >&2
    echo "[Hint] Use './run_HM.sh status' or './run_HM.sh logs'." >&2
    exit 1
  fi
  rm -f "$MEMORY_PID_FILE"

  # The three artifact paths were added after the original server CLI. Keep
  # them when the checked-out CLI supports them, but do not make an otherwise
  # compatible training launch fail just because the server copy is older.
  local cli_help
  if ! cli_help="$(memory_cli_help)"; then
    echo "[Error] The Memory Train CLI could not be loaded for --help." >&2
    echo "[Hint] Check PYTHONPATH and the selected PYTHON executable." >&2
    exit 1
  fi
  local cli_output_args=()
  local unsupported_output_args=()
  if [[ "$cli_help" == *"--best-checkpoint"* ]]; then
    cli_output_args+=(--best-checkpoint "$MEMORY_BEST_CHECKPOINT")
  else
    unsupported_output_args+=(--best-checkpoint)
  fi
  if [[ "$cli_help" == *"--validation-history-path"* ]]; then
    cli_output_args+=(--validation-history-path "$VALIDATION_HISTORY")
  else
    unsupported_output_args+=(--validation-history-path)
  fi
  if [[ "$cli_help" == *"--controller-diagnostics-path"* ]]; then
    cli_output_args+=(--controller-diagnostics-path "$CONTROLLER_DIAGNOSTICS")
  else
    unsupported_output_args+=(--controller-diagnostics-path)
  fi
  if [[ "${#unsupported_output_args[@]}" -gt 0 ]]; then
    echo "[Warning] Older Train CLI detected; skipping unsupported optional outputs: ${unsupported_output_args[*]}" >&2
    echo "[Hint] Sync Memory/Train/TrainingCLI.py to enable best-checkpoint and diagnostics artifacts." >&2
  fi

  local training_epochs=50
  local split_args=()
  local controller_args=()
  if [[ "$controller_mode" != "0" ]]; then
    training_epochs="$EPOCHS"
    split_args=(--split-manifest "$SPLIT_MANIFEST" --split train)
    if [[ "$controller_mode" == "1" ]]; then
      controller_args=(--controller-v4-fresh)
    else
      controller_args=(
        --controller-base-checkpoint "$BASE_CONTROLLER_CHECKPOINT"
        --controller-target-version "$CONTROLLER_VERSION"
        --controller-heads "$CONTROLLER_HEADS"
      )
      if [[ "$controller_mode" == "3" ]]; then
        controller_args+=(--controller-write-ranking)
      fi
    fi
  fi

  echo "[HM 5/5] Starting integrated Memory training: RUN_NAME=$RUN_NAME"
  # The current CLI accepts this historical Deep-Sleep option as an alias;
  # it also keeps the script compatible with the server's older CLI.
  nohup env \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$MEMORY_PYTHONPATH" \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES="$DEVICE_IDS" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    "$PYTHON_BIN" -u -B -m Train.Train \
      --data-path "$HAWKES_DATA" \
      "${split_args[@]}" \
      --h-tree "$H_TREE_OUTPUT" \
      --sequence-summary "$SUMMARY_CSV" \
      --checkpoint "$MEMORY_CHECKPOINT" \
      "${cli_output_args[@]}" \
      "${controller_args[@]}" \
      --epochs "$training_epochs" \
      --cold-start-epochs 10 \
      --z-dim 50 \
      --node-dim 128 \
      --memory-key-dim 64 \
      --tree-init-depth 0 \
      --num-basis 2 \
      --decays 0.5 1.5 \
      --semantic-blend 0 \
      --residual-init-scale 0.08 \
      --residual-init-rank 4 \
      --residual-init-grad-clip 0 \
      --leaf-symmetry-scale 0 \
      --alignment-epochs 5 \
      --alignment-batch-size 16 \
      --alignment-lr 0.001 \
      --alignment-weight-decay 0.00001 \
      --alignment-temperature 1.0 \
      --alignment-grad-clip 5.0 \
      --prune-warmup-epochs 12 \
      --merge-min-replay 12 \
      --frontier-min-experts 2 \
      --frontier-budget 7 \
      --frontier-routing-temperature 1.10 \
      --frontier-exploration 0 \
      --frontier-confidence-weight 0.60 \
      --frontier-compute-cost 0.005 \
      --frontier-posterior-temperature 0.85 \
      --frontier-credible-mass 0.30 \
      --frontier-owner-confidence 0.50 \
      --max-writes-per-sequence 8 \
      --route-mix-weight 0 \
      --route-posterior-weight 0 \
      --route-distill-weight 0.25 \
      --route-mi-weight 0.15 \
      --route-balance-weight 0.10 \
      --route-energy-temperature 1.0 \
      --route-encoder-warmup-epochs 0 \
      --route-encoder-grad-scale 0.08 \
      --route-encoder-reliability-decay 0.80 \
      --route-teacher-temperature 0.85 \
      --route-balance-batch-size 64 \
      --wake-wavefront-batch-size 64 \
      --light-replay-budget 128 \
      --deep-min-interval 3 \
      --deep-computation-cost 0.05 \
      --deep-prior-probability 0.10 \
      --deep-prior-weight 0.01 \
      --deep-evidence-budget 32 \
      --topology-inertia-strength 0.03 \
      --topology-inertia-tau 3.0 \
      --device cuda \
    >"$MEMORY_LOG" 2>&1 &

  local pid=$!
  printf '%s\n' "$pid" >"$MEMORY_PID_FILE"
  echo "[Started] PID=$pid"
  echo "[Log] $MEMORY_LOG"
  echo "[Follow] ./run_HM.sh logs"
}

prepare_memory_split() {
  require_memory_runtime
  require_file "$HAWKES_DATA"
  mkdir -p "$(dirname "$SPLIT_MANIFEST")"
  if [[ -e "$SPLIT_MANIFEST" ]]; then
    echo "[Exists] Reusing fixed split: $SPLIT_MANIFEST"
    return
  fi
  env PYTHONPATH="$MEMORY_PYTHONPATH" "$PYTHON_BIN" -m DataSplit \
    --data-path "$HAWKES_DATA" \
    --output "$SPLIT_MANIFEST" \
    --seed "$SPLIT_SEED"
}

inspect_checkpoint() {
  require_memory_runtime
  require_file "$MEMORY_CHECKPOINT"
  env PYTHONPATH="$MEMORY_PYTHONPATH" "$PYTHON_BIN" -c \
    "import torch; p=torch.load(r'$MEMORY_CHECKPOINT',map_location='cpu',weights_only=False); print({'epoch':p.get('epoch'),'format_version':p.get('format_version'),'router':p.get('model_config',{}).get('router_kind'),'controller_version':p.get('controller_state',{}).get('controller_version'),'sleep_keys':[k for k in ('deep_sleep_gate_state_dict','topology_selector_state_dict','sleep_state') if k in p]})"
}

recalibrate_controller() {
  require_memory_runtime
  require_file "$MEMORY_BEST_CHECKPOINT"
  require_file "$SPLIT_MANIFEST" "prepare-memory-split"
  env PYTHONPATH="$MEMORY_PYTHONPATH" "$PYTHON_BIN" -m RecalibrateController \
    --checkpoint "$MEMORY_BEST_CHECKPOINT" --data-path "$HAWKES_DATA" \
    --split-manifest "$SPLIT_MANIFEST" --output "$RECALIBRATED_CHECKPOINT" \
    --seed "$SPLIT_SEED" --device cuda
}

calibrate_write_rollout() {
  require_memory_runtime
  require_file "$MEMORY_BEST_CHECKPOINT"
  require_file "$SPLIT_MANIFEST" "prepare-memory-split"
  env PYTHONPATH="$MEMORY_PYTHONPATH" "$PYTHON_BIN" -m CalibrateWriteRollout \
    --checkpoint "$MEMORY_BEST_CHECKPOINT" --data-path "$HAWKES_DATA" \
    --split-manifest "$SPLIT_MANIFEST" --output "$ROLLOUT_CALIBRATED_CHECKPOINT" \
    --seed "$SPLIT_SEED" --device cuda
}

evaluate_controller() {
  local mode="${1:-quick}"
  local checkpoint="${EVAL_CHECKPOINT:-$MEMORY_BEST_CHECKPOINT}"
  local output="$FULL_EVAL_DIR"
  local extra=()
  [[ "$mode" == "quick" ]] && output="$QUICK_EVAL_DIR" && extra=(--quick-per-cluster 2)
  require_memory_runtime
  require_file "$checkpoint"
  require_file "$SPLIT_MANIFEST" "prepare-memory-split"
  env PYTHONPATH="$MEMORY_PYTHONPATH" "$PYTHON_BIN" -m Evaluate \
    --checkpoint "$checkpoint" --data-path "$HAWKES_DATA" \
    --split-manifest "$SPLIT_MANIFEST" --output-dir "$output" \
    --protocol both --seed "$SPLIT_SEED" "${extra[@]}"
}

show_status() {
  if memory_is_running; then
    local pid
    pid="$(<"$MEMORY_PID_FILE")"
    echo "[Running] Memory training PID=$pid"
    ps -fp "$pid"
    echo
    tail -n 17 "$MEMORY_LOG" 2>/dev/null || true
  else
    echo "[Stopped] No live Memory training process was found."
    if [[ -s "$MEMORY_LOG" ]]; then
      echo "[Last log lines]"
      tail -n 17 "$MEMORY_LOG"
    fi
    return 1
  fi
}

follow_logs() {
  if [[ ! -f "$MEMORY_LOG" ]]; then
    echo "[Error] Memory log does not exist yet: $MEMORY_LOG" >&2
    exit 1
  fi
  tail -f "$MEMORY_LOG"
}

stop_memory() {
  if ! memory_is_running; then
    echo "[Stopped] No live Memory training process was found."
    rm -f "$MEMORY_PID_FILE"
    return 0
  fi
  local pid
  pid="$(<"$MEMORY_PID_FILE")"
  kill "$pid"
  rm -f "$MEMORY_PID_FILE"
  echo "[Stopped] Sent SIGTERM to Memory training PID=$pid"
}

ACTION="${1:-help}"
case "$ACTION" in
  train-thp|train_thp)
    train_thp
    ;;
  encode)
    encode_sequences
    ;;
  train-attention|train_attention|attention)
    train_attention
    ;;
  build-h-tree|build_h_tree|h-tree|htree)
    build_h_tree
    ;;
  build-strict-baseline|build_strict_baseline)
    build_strict_baseline
    ;;
  memory)
    start_memory
    ;;
  prepare-memory-split)
    prepare_memory_split
    ;;
  memory-controller)
    start_memory 1
    ;;
  controller-finetune)
    start_memory 2
    ;;
  controller-write-rank-finetune)
    start_memory 3
    ;;
  inspect-checkpoint)
    inspect_checkpoint
    ;;
  recalibrate-controller)
    recalibrate_controller
    ;;
  calibrate-write-rollout)
    calibrate_write_rollout
    ;;
  evaluate-controller-quick)
    evaluate_controller quick
    ;;
  evaluate-controller-full)
    evaluate_controller full
    ;;
  all)
    train_thp
    encode_sequences
    train_attention
    build_h_tree
    start_memory
    ;;
  status)
    show_status
    ;;
  logs)
    follow_logs
    ;;
  stop)
    stop_memory
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "[Error] Unknown action: $ACTION" >&2
    usage >&2
    exit 2
    ;;
esac
