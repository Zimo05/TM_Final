#!/usr/bin/env bash
set -euo pipefail

# Train/encode the THP MultiAttenEncoder on the 13-cluster dataset.
#
# Usage from the repository root:
#   bash MultiAttentionEncoder/run.sh train
#   bash MultiAttentionEncoder/run.sh encode
#   bash MultiAttentionEncoder/run.sh train_attention
#   bash MultiAttentionEncoder/run.sh final_encode
#
# Optional overrides:
#   PYTHON=/path/to/python DEVICES=0,1,2,3 bash MultiAttentionEncoder/run.sh train

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
THP_DIR="$SCRIPT_DIR/THP"

ACTION="${1:-train}"
PYTHON_BIN="${PYTHON:-python}"
DEVICE_IDS="${DEVICES:-0,1,2,3}"
DEVICE_TYPE="${DEVICE_TYPE:-cuda}"

# Keep the historical defaults for the DWS runner, while allowing CL (and
# CPU-only smoke tests) to select its own batch/worker/epoch settings without
# duplicating this pipeline.
THP_BATCH_SIZE="${THP_BATCH_SIZE:-64}"
THP_NUM_WORKERS="${THP_NUM_WORKERS:-4}"
THP_EPOCHS="${THP_EPOCHS:-100}"
THP_DATA_PARALLEL="${THP_DATA_PARALLEL:-1}"
ATTENTION_BATCH_SIZE="${ATTENTION_BATCH_SIZE:-64}"
ATTENTION_EPOCHS="${ATTENTION_EPOCHS:-50}"

DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/Data/tree_13/13Cluster/THP_13.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/Data/tree_13/thp_checkpoints}"
TRAIN_LOG="${TRAIN_LOG:-$PROJECT_ROOT/Data/tree_13/thp_train.log}"
BEST_CHECKPOINT="${CHECKPOINT:-$OUTPUT_DIR/checkpoint_best.pt}"
ENCODED_OUTPUT="${ENCODED_OUTPUT:-$PROJECT_ROOT/Data/tree_13/13Cluster/thp_encoded_13.pt}"
SUMMARY_CSV="${SUMMARY_CSV:-$PROJECT_ROOT/Data/tree_13/sequence_summary.csv}"
TREE_CSV="${TREE_CSV:-$PROJECT_ROOT/Data/tree_13/tree_node_sequences.csv}"
FINAL_OUTPUT="${FINAL_OUTPUT:-$PROJECT_ROOT/Data/tree_13/h_tree_13.pt}"
ATTENTION_WEIGHTS="${ATTENTION_WEIGHTS:-$PROJECT_ROOT/Data/tree_13/encoder_weights_13.pt}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-}"
SPLIT_DATA_PATH="${SPLIT_DATA_PATH:-}"
STRICT_SPLIT_ARGS=()
if [[ -n "$SPLIT_MANIFEST" ]]; then
  if [[ -z "$SPLIT_DATA_PATH" ]]; then
    echo "SPLIT_DATA_PATH is required when SPLIT_MANIFEST is set." >&2
    exit 2
  fi
  STRICT_SPLIT_ARGS=(--split-manifest "$SPLIT_MANIFEST" --split-data-path "$SPLIT_DATA_PATH")
fi

case "$ACTION" in
  train)
    mkdir -p "$OUTPUT_DIR"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$DEVICE_IDS" \
      "$PYTHON_BIN" "$THP_DIR/Main.py" \
        -data "$DATA_PATH" \
        -batch_size "$THP_BATCH_SIZE" \
        -num_workers "$THP_NUM_WORKERS" \
        -data_parallel "$THP_DATA_PARALLEL" \
        -n_head 4 \
        -n_layers 3 \
        -d_model 128 \
        -d_rnn 128 \
        -d_inner_hid 256 \
        -d_k 32 \
        -d_v 32 \
        -dropout 0.3 \
        -lr 0.001 \
        -weight_decay 0.0001 \
        -smooth 0 \
        -epoch "$THP_EPOCHS" \
        -event_loss_weight 0 \
        -time_loss_weight 0 \
        -class_weight 0 \
        -warmup_steps 100 \
        -patience 12 \
        -min_delta 0 \
        -log "$TRAIN_LOG" \
        -save_dir "$OUTPUT_DIR" \
        ${STRICT_SPLIT_ARGS[@]+"${STRICT_SPLIT_ARGS[@]}"}
    ;;

  encode)
    if [[ ! -f "$BEST_CHECKPOINT" ]]; then
      echo "Checkpoint not found: $BEST_CHECKPOINT" >&2
      echo "Run the train action first." >&2
      exit 1
    fi

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$DEVICE_IDS" \
      "$PYTHON_BIN" "$THP_DIR/EncodeMain.py" \
        --data "$DATA_PATH" \
        --checkpoint "$BEST_CHECKPOINT" \
        --output "$ENCODED_OUTPUT" \
        --batch-size "$THP_BATCH_SIZE" \
        --num-workers "$THP_NUM_WORKERS" \
        --device "$DEVICE_TYPE"
    ;;

  train_attention)
    if [[ ! -f "$BEST_CHECKPOINT" ]]; then
      echo "Checkpoint not found: $BEST_CHECKPOINT" >&2
      echo "Run the train action first." >&2
      exit 1
    fi

    "$PYTHON_BIN" "$SCRIPT_DIR/Process_input.py" \
      "$SUMMARY_CSV" \
      "$TREE_CSV"

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$DEVICE_IDS" \
      "$PYTHON_BIN" "$SCRIPT_DIR/AttentionEncoder/Train.py" \
        --thp_json "$DATA_PATH" \
        --tree_csv "$TREE_CSV" \
        --summary_csv "$SUMMARY_CSV" \
        --checkpoint "$BEST_CHECKPOINT" \
        --weights_out "$ATTENTION_WEIGHTS" \
        --d_model 128 \
        --num_heads 4 \
        --d_rnn 128 \
        --d_inner_hid 256 \
        --d_k 32 \
        --d_v 32 \
        --n_layers 3 \
        --dropout 0.3 \
        --epochs "$ATTENTION_EPOCHS" \
        --batch_size "$ATTENTION_BATCH_SIZE" \
        --lr 0.001 \
        --weight_decay 0.0001 \
        --route_weight 1.0 \
        --path_weight 0.0 \
        --recon_weight 0.5 \
        --grad_clip 1.0 \
        --seed 42 \
        --train_ratio 0.8 \
        --dev_ratio 0.1 \
        --patience 10 \
        --min_delta 0.0001 \
        --device "$DEVICE_TYPE" \
        ${STRICT_SPLIT_ARGS[@]+"${STRICT_SPLIT_ARGS[@]}"}
    ;;

  final_encode)
    if [[ ! -f "$BEST_CHECKPOINT" ]]; then
      echo "Checkpoint not found: $BEST_CHECKPOINT" >&2
      echo "Run the train action first." >&2
      exit 1
    fi

    "$PYTHON_BIN" "$SCRIPT_DIR/Process_input.py" \
      "$SUMMARY_CSV" \
      "$TREE_CSV"

    WEIGHTS_ARGS=()
    if [[ -f "$ATTENTION_WEIGHTS" ]]; then
      WEIGHTS_ARGS=(--weights "$ATTENTION_WEIGHTS")
    else
      echo "[Warn] Attention weights not found: $ATTENTION_WEIGHTS" >&2
      echo "[Warn] Run '$0 train_attention' first; continuing with random initialization." >&2
    fi

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$DEVICE_IDS" \
      "$PYTHON_BIN" "$SCRIPT_DIR/AttentionEncoder/AttenEncoderMain_v1.py" \
        --thp_json "$DATA_PATH" \
        --tree_csv "$TREE_CSV" \
        --summary_csv "$SUMMARY_CSV" \
        --checkpoint "$BEST_CHECKPOINT" \
        --output "$FINAL_OUTPUT" \
        --d_model 128 \
        --num_heads 4 \
        --d_rnn 128 \
        --d_inner_hid 256 \
        --d_k 32 \
        --d_v 32 \
        --n_layers 3 \
        --dropout 0.3 \
        --batch_size "$ATTENTION_BATCH_SIZE" \
        --device "$DEVICE_TYPE" \
        --node_only \
        ${STRICT_SPLIT_ARGS[@]+"${STRICT_SPLIT_ARGS[@]}"} \
        ${WEIGHTS_ARGS[@]+"${WEIGHTS_ARGS[@]}"}
    ;;

  *)
    echo "Usage: $0 {train|encode|train_attention|final_encode}" >&2
    exit 2
    ;;
esac
