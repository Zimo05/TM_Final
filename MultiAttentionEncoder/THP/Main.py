"""
THP Training Script

Trains the MultiAttenEncoder on event sequence data (JSON format) and saves
a checkpoint that can be used by EncodeMain.py / DataTreeConstructor.py.

Usage:
    python Main.py -data /path/to/THP_8.json -epoch 100 -batch_size 16 -log log.txt

Output:
    checkpoint.pt  — trained model weights, loadable by THPEncoding.build_model()
"""

import argparse
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure the THP directory is on sys.path so flat imports work
_THP_DIR = str(Path(__file__).resolve().parent)
if _THP_DIR not in sys.path:
    sys.path.insert(0, _THP_DIR)
_ENCODER_ROOT = str(Path(__file__).resolve().parents[1])
if _ENCODER_ROOT not in sys.path:
    sys.path.insert(0, _ENCODER_ROOT)

import Utils
from Dataset import get_dataloader
from TransformerModel.Models import MultiAttenEncoder
from tqdm import tqdm
from SplitManifest import build_data_provenance, load_strict_manifest


# ---------------------------------------------------------------------------
# Data loading (JSON → train / dev / test split)
# ---------------------------------------------------------------------------

def load_json_data(json_path: str) -> Tuple[List, int]:
    """Load event sequences from a JSON file.

    Expected JSON format (same as THP_8.json):
        {"key": [{"time_since_start": ..., "time_since_last_event": ..., "type_event": ...}, ...], ...}

    Returns
    -------
    event_streams : list of list of dict
    num_types : int
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        streams = list(raw.values())
    elif isinstance(raw, list):
        streams = raw
    else:
        raise TypeError(f"Unsupported JSON root type: {type(raw).__name__}")

    # Compute num_types: max type_event + 1  (type_event is 0-indexed)
    num_types = max(
        int(ev["type_event"]) for s in streams for ev in s
    ) + 1

    return streams, num_types


def split_data(streams: List, train_ratio=0.8, dev_ratio=0.1, seed=42):
    """Randomly split event streams into train / dev / test."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(streams))
    n_train = int(len(streams) * train_ratio)
    n_dev = int(len(streams) * dev_ratio)

    train_idx = idx[:n_train]
    dev_idx = idx[n_train:n_train + n_dev]
    test_idx = idx[n_train + n_dev:]

    train_data = [streams[i] for i in train_idx]
    dev_data = [streams[i] for i in dev_idx]
    test_data = [streams[i] for i in test_idx]
    return train_data, dev_data, test_data


def prepare_dataloader(opt):
    """Load JSON data and return dataloaders + num_types."""
    print(f"[Info] Loading data from {opt.data} ...")
    all_streams, num_types = load_json_data(opt.data)
    print(f"[Info] Loaded {len(all_streams)} sequences, num_types={num_types}")
    opt.data_provenance = None
    if opt.split_manifest is None:
        train_data, dev_data, test_data = split_data(
            all_streams,
            train_ratio=opt.train_ratio,
            dev_ratio=opt.dev_ratio,
            seed=opt.seed,
        )
    else:
        if opt.split_data_path is None:
            raise ValueError("--split-data-path is required with --split-manifest")
        with open(opt.data, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError(
                "strict mode requires keyed THP JSON so source IDs are explicit"
            )

        def sort_key(key):
            parts = str(key).split("_")
            try:
                return (int(parts[0]), int(parts[1]) if len(parts) >= 2 else 0)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"cannot map THP JSON key {key!r} to a global source ID"
                ) from exc

        ordered_keys = sorted(raw, key=sort_key)
        source_to_stream = {
            source_id: raw[key] for source_id, key in enumerate(ordered_keys)
        }
        manifest = load_strict_manifest(
            opt.split_manifest,
            data_path=opt.split_data_path,
            available_source_ids=source_to_stream,
        )
        splits = manifest["splits"]
        train_data = [source_to_stream[index] for index in splits["train"]]
        dev_data = [source_to_stream[index] for index in splits["validation"]]
        test_data = [source_to_stream[index] for index in splits["test"]]
        opt.data_provenance = build_data_provenance(manifest)
    print(f"[Info] Split: train={len(train_data)}, dev={len(dev_data)}, test={len(test_data)}")

    # Keep Main.py compatible with older Dataset.py copies that predate the
    # configurable num_workers argument.
    loader_kwargs = {}
    if "num_workers" in inspect.signature(get_dataloader).parameters:
        loader_kwargs["num_workers"] = opt.num_workers
    else:
        print(
            "[Warn] Dataset.get_dataloader does not support num_workers; "
            "using its built-in worker setting."
        )

    trainloader = get_dataloader(
        train_data, opt.batch_size, shuffle=True, **loader_kwargs
    )
    devloader = get_dataloader(
        dev_data, opt.batch_size, shuffle=False, **loader_kwargs
    )
    testloader = get_dataloader(
        test_data, opt.batch_size, shuffle=False, **loader_kwargs
    )
    return trainloader, devloader, testloader, num_types, train_data


def compute_class_weights(train_data, num_types):
    """Inverse-frequency class weights (normalized to mean 1) for type loss.

    train_data uses 0-indexed type_event; loss operates on the same 0-indexed
    space, so the returned weight tensor has length num_types.
    """
    counts = np.zeros(num_types, dtype=np.float64)
    for stream in train_data:
        for ev in stream:
            counts[int(ev["type_event"])] += 1
    counts = np.clip(counts, 1.0, None)
    inv = 1.0 / counts
    inv = inv / inv.mean()  # normalize so average weight is 1.0
    return torch.tensor(inv, dtype=torch.float32)


def macro_f1(confusion):
    confusion = confusion.astype(np.float64)
    true_positive = np.diag(confusion)
    precision = true_positive / np.maximum(confusion.sum(axis=0), 1.0)
    recall = true_positive / np.maximum(confusion.sum(axis=1), 1.0)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(f1.mean())


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def unwrap_model(model):
    """Return the underlying model when DataParallel is enabled."""
    return model.module if isinstance(model, nn.DataParallel) else model


def train_epoch(model, training_data, optimizer, scheduler, pred_loss_func, opt):
    """Epoch operation in training phase."""
    model.train()

    total_event_ll = 0
    total_time_se = 0
    total_correct = 0       # correctly predicted next-event types
    total_top3 = 0
    total_num_pred = 0      # total predictable next-events (for acc / rmse)
    base_model = unwrap_model(model)
    confusion = np.zeros((base_model.num_types, base_model.num_types), dtype=np.int64)

    for batch in tqdm(training_data, mininterval=2,
                      desc="  - (Training)   ", leave=False):
        event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

        optimizer.zero_grad()

        enc_out, prediction = model(event_type, event_time, time_gap)

        # --- Loss components (all normalized to per-token scale) ---
        # 1. Negative log-likelihood of next events
        # 2. Event type prediction loss  (normalized by #predictable tokens)
        pred_loss, correct_num, valid_num, top3_num, batch_confusion = Utils.type_loss(
            prediction[0], event_type, pred_loss_func
        )
        if opt.event_loss_weight > 0:
            event_ll, non_event_ll = Utils.log_likelihood(
                base_model, enc_out, event_time, event_type
            )
            event_loss = -torch.sum(event_ll - non_event_ll) / valid_num.clamp_min(1)
        else:
            event_ll = enc_out.new_zeros(event_type.size(0))
            non_event_ll = enc_out.new_zeros(event_type.size(0))
            event_loss = enc_out.new_zeros(())
        pred_loss = pred_loss / valid_num.clamp_min(1)

        # 3. Time gap prediction loss  (normalized by #predictable tokens)
        if opt.time_loss_weight > 0:
            se = Utils.time_loss(prediction[1], time_gap, event_type)
            time_loss = se / valid_num.clamp_min(1)
        else:
            se = enc_out.new_zeros(())
            time_loss = enc_out.new_zeros(())

        # Weighted sum: NLL is an auxiliary term, type prediction is the goal.
        loss = opt.event_loss_weight * event_loss + pred_loss \
            + opt.time_loss_weight * time_loss

        loss.backward()

        # Gradient clipping to prevent exploding gradients
        if opt.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)

        optimizer.step()
        scheduler.step()  # step per batch for warmup + cosine decay

        # Logging
        total_event_ll += torch.sum(event_ll - non_event_ll).item()
        total_time_se += se.item()
        total_correct += correct_num.item()
        total_top3 += top3_num.item()
        total_num_pred += valid_num.item()
        confusion += batch_confusion.detach().cpu().numpy()

    rmse = np.sqrt(total_time_se / max(total_num_pred, 1))
    return (
        total_event_ll / max(total_num_pred, 1),
        total_correct / max(total_num_pred, 1),
        rmse,
        total_top3 / max(total_num_pred, 1),
        macro_f1(confusion),
    )


def eval_epoch(model, validation_data, pred_loss_func, opt):
    """Epoch operation in evaluation phase."""
    model.eval()

    total_event_ll = 0
    total_time_se = 0
    total_correct = 0
    total_top3 = 0
    total_num_pred = 0
    base_model = unwrap_model(model)
    confusion = np.zeros((base_model.num_types, base_model.num_types), dtype=np.int64)

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc="  - (Validation) ", leave=False):
            event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

            enc_out, prediction = model(event_type, event_time, time_gap)

            if opt.event_loss_weight > 0:
                event_ll, non_event_ll = Utils.log_likelihood(
                    base_model, enc_out, event_time, event_type
                )
            else:
                event_ll = enc_out.new_zeros(event_type.size(0))
                non_event_ll = enc_out.new_zeros(event_type.size(0))
            _, correct_num, valid_num, top3_num, batch_confusion = Utils.type_loss(
                prediction[0], event_type, pred_loss_func
            )
            if opt.time_loss_weight > 0:
                se = Utils.time_loss(prediction[1], time_gap, event_type)
            else:
                se = enc_out.new_zeros(())

            total_event_ll += torch.sum(event_ll - non_event_ll).item()
            total_time_se += se.item()
            total_correct += correct_num.item()
            total_top3 += top3_num.item()
            total_num_pred += valid_num.item()
            confusion += batch_confusion.detach().cpu().numpy()

    rmse = np.sqrt(total_time_se / max(total_num_pred, 1))
    return (
        total_event_ll / max(total_num_pred, 1),
        total_correct / max(total_num_pred, 1),
        rmse,
        total_top3 / max(total_num_pred, 1),
        macro_f1(confusion),
    )


def model_config_from_opt(opt, num_types):
    return {
        "num_types": num_types,
        "d_model": opt.d_model,
        "d_rnn": opt.d_rnn,
        "d_inner": opt.d_inner_hid,
        "n_layers": opt.n_layers,
        "n_head": opt.n_head,
        "d_k": opt.d_k,
        "d_v": opt.d_v,
        "dropout": opt.dropout,
        "use_time_gap": True,
    }


def save_checkpoint(path, model, model_config, epoch, metrics, data_provenance=None):
    base_model = unwrap_model(model)
    torch.save(
        {
            # Keep checkpoints loadable without DataParallel's ``module.`` prefix.
            "state_dict": base_model.state_dict(),
            "model_config": model_config,
            "epoch": int(epoch),
            # Python scalars remain compatible with PyTorch 2.6+'s default
            # weights_only=True safe loader; NumPy scalars do not.
            "metrics": {key: float(value) for key, value in metrics.items()},
            "data_provenance": data_provenance,
        },
        path,
    )


def train(model, training_data, validation_data, optimizer, scheduler,
          pred_loss_func, model_config, opt):
    """Training loop with checkpoint saving."""
    best_accuracy = -float("inf")
    best_event_ll = -float("inf")
    best_path = os.path.join(opt.save_dir, "checkpoint_best.pt")
    epochs_without_improvement = 0
    last_epoch = 0

    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        last_epoch = epoch
        print(f"\n[ Epoch {epoch} ]")

        start = time.time()
        train_event, train_type, train_time, train_top3, train_f1 = train_epoch(
            model, training_data, optimizer, scheduler, pred_loss_func, opt
        )
        print(
            f"  - (Training)    loglikelihood: {train_event: 8.5f}, "
            f"accuracy: {train_type: 8.5f}, top3: {train_top3: 8.5f}, "
            f"macro-F1: {train_f1: 8.5f}, RMSE: {train_time: 8.5f}, "
            f"elapse: {(time.time() - start) / 60:3.3f} min"
        )

        start = time.time()
        valid_event, valid_type, valid_time, valid_top3, valid_f1 = eval_epoch(
            model, validation_data, pred_loss_func, opt
        )
        print(
            f"  - (Validation)  loglikelihood: {valid_event: 8.5f}, "
            f"accuracy: {valid_type: 8.5f}, top3: {valid_top3: 8.5f}, "
            f"macro-F1: {valid_f1: 8.5f}, RMSE: {valid_time: 8.5f}, "
            f"elapse: {(time.time() - start) / 60:3.3f} min"
        )

        # Log to file
        with open(opt.log, "a") as f:
            f.write(
                f"{epoch}, {valid_event: 8.5f}, {valid_type: 8.5f}, "
                f"{valid_top3: 8.5f}, {valid_f1: 8.5f}, {valid_time: 8.5f}\n"
            )

        # Accuracy is the primary goal. Likelihood breaks ties.
        improved = valid_type > best_accuracy + opt.min_delta
        if improved:
            best_accuracy = valid_type
            best_event_ll = valid_event
            epochs_without_improvement = 0
            os.makedirs(opt.save_dir, exist_ok=True)
            save_checkpoint(
                best_path,
                model,
                model_config,
                epoch,
                {
                    "validation_loglikelihood": valid_event,
                    "validation_accuracy": valid_type,
                    "validation_top3": valid_top3,
                    "validation_macro_f1": valid_f1,
                    "validation_rmse": valid_time,
                },
                data_provenance=opt.data_provenance,
            )
            print(f"  - [Checkpoint] Saved best model → {best_path}")
        else:
            epochs_without_improvement += 1

        if opt.patience > 0 and epochs_without_improvement >= opt.patience:
            print(
                f"  - [EarlyStopping] No validation accuracy improvement "
                f"> {opt.min_delta:g} for {opt.patience} epochs. "
                f"Best accuracy={best_accuracy:.5f}; stopping at epoch {epoch}."
            )
            break

    # --- Save final checkpoint ---
    final_path = os.path.join(opt.save_dir, "checkpoint_final.pt")
    os.makedirs(opt.save_dir, exist_ok=True)
    save_checkpoint(
        final_path,
        model,
        model_config,
        last_epoch,
        {
            "validation_loglikelihood": valid_event,
            "validation_accuracy": valid_type,
            "validation_top3": valid_top3,
            "validation_macro_f1": valid_f1,
            "validation_rmse": valid_time,
        },
        data_provenance=opt.data_provenance,
    )
    print(f"\n[Checkpoint] Final model saved → {final_path}")
    return best_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train MultiAttenEncoder on event sequences.")

    # Data
    parser.add_argument("-data", required=True,
                        help="Path to JSON file (e.g. THP_8.json)")
    parser.add_argument("-train_ratio", type=float, default=0.8)
    parser.add_argument("-dev_ratio", type=float, default=0.1)
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("--split-manifest", type=Path, default=None,
                        help="Strict shared train/validation/test manifest.")
    parser.add_argument("--split-data-path", type=Path, default=None,
                        help="Source CSV whose SHA-256 is recorded by the manifest.")

    # Training
    parser.add_argument("-epoch", type=int, default=100)
    parser.add_argument("-batch_size", type=int, default=16)
    parser.add_argument("-num_workers", type=int, default=0)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-weight_decay", type=float, default=1e-4,
                        help="AdamW weight decay coefficient.")
    parser.add_argument("-smooth", type=float, default=0.0,
                        help="Label smoothing epsilon (0 = cross-entropy)")
    parser.add_argument("-grad_clip", type=float, default=5.0,
                        help="Gradient clipping value")
    parser.add_argument("-warmup_steps", type=int, default=100,
                        help="Number of warmup steps for scheduler")
    parser.add_argument("-patience", type=int, default=15,
                        help="Early-stop after this many epochs without validation "
                             "accuracy improvement; 0 disables early stopping.")
    parser.add_argument("-min_delta", type=float, default=1e-4,
                        help="Minimum validation accuracy increase counted as improvement.")
    parser.add_argument("-event_loss_weight", type=float, default=0.0,
                        help="Weight on the Hawkes NLL term (auxiliary). "
                             "Type prediction loss has weight 1.0.")
    parser.add_argument("-time_loss_weight", type=float, default=0.0,
                        help="Weight on the time-gap MSE term (auxiliary).")
    parser.add_argument("-class_weight", type=int, default=0,
                        help="1 = use inverse-frequency class weights for the "
                             "type loss (handles imbalance); 0 = disable.")
    parser.add_argument("-data_parallel", type=int, default=0,
                        help="1 = distribute each batch across all visible CUDA GPUs; "
                             "0 = use one GPU.")

    # Model architecture
    parser.add_argument("-d_model", type=int, default=128)
    parser.add_argument("-d_rnn", type=int, default=128)
    parser.add_argument("-d_inner_hid", type=int, default=256)
    parser.add_argument("-d_k", type=int, default=32)
    parser.add_argument("-d_v", type=int, default=32)
    parser.add_argument("-n_head", type=int, default=4)
    parser.add_argument("-n_layers", type=int, default=4)
    parser.add_argument("-dropout", type=float, default=0.2)

    # Output
    parser.add_argument("-log", type=str, default="log.txt")
    parser.add_argument("-save_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoint.pt files")

    opt = parser.parse_args()

    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)

    # Device
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {opt.device}")

    # Log header
    with open(opt.log, "w") as f:
        f.write("Epoch, Log-likelihood, Accuracy, Top-3, Macro-F1, RMSE\n")

    print(f"[Info] Parameters: {opt}")

    # --- Data ---
    trainloader, devloader, testloader, num_types, train_data = prepare_dataloader(opt)

    # --- Model ---
    model = MultiAttenEncoder(
        num_types=num_types,
        d_model=opt.d_model,
        d_rnn=opt.d_rnn,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
        use_time_gap=True,
    )
    model.to(opt.device)

    if opt.data_parallel:
        visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if visible_gpus > 1:
            model = nn.DataParallel(model)
            print(f"[Info] DataParallel enabled on {visible_gpus} GPUs")
        else:
            print(
                f"[Warn] DataParallel requested, but only {visible_gpus} CUDA GPU "
                "is visible; continuing on one device."
            )

    # --- Optimizer & scheduler ---
    optimizer = optim.AdamW(
        model.parameters(),
        lr=opt.lr,
        weight_decay=opt.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-5,
    )

    # Linear warmup + CosineAnnealingLR
    warmup_steps = opt.warmup_steps
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(max(1, opt.epoch * len(trainloader) - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Loss function ---
    # Class weights to counter heavy imbalance (e.g. type7/type6 dominating).
    weight = None
    if opt.class_weight:
        weight = compute_class_weights(train_data, num_types).to(opt.device)
        print(f"[Info] Using inverse-freq class weights: "
              f"{weight.detach().cpu().numpy().round(3)}")

    if opt.smooth > 0:
        pred_loss_func = Utils.LabelSmoothingLoss(
            opt.smooth, num_types, ignore_index=-1, weight=weight
        )
    else:
        pred_loss_func = nn.CrossEntropyLoss(
            ignore_index=-1, reduction="none", weight=weight
        )

    # --- Parameter count ---
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Info] Number of parameters: {num_params}")

    # --- Train ---
    model_config = model_config_from_opt(opt, num_types)
    best_path = train(
        model,
        trainloader,
        devloader,
        optimizer,
        scheduler,
        pred_loss_func,
        model_config,
        opt,
    )

    # --- Test evaluation with the validation-selected checkpoint ---
    checkpoint = torch.load(best_path, map_location=opt.device)
    unwrap_model(model).load_state_dict(checkpoint["state_dict"])
    print(f"\n[Info] Loaded best validation-accuracy checkpoint: {best_path}")
    print("[Info] Evaluating on test set ...")
    test_event, test_type, test_time, test_top3, test_f1 = eval_epoch(
        model, testloader, pred_loss_func, opt
    )
    print(
        f"  - (Test)  loglikelihood: {test_event: 8.5f}, "
        f"accuracy: {test_type: 8.5f}, top3: {test_top3: 8.5f}, "
        f"macro-F1: {test_f1: 8.5f}, RMSE: {test_time: 8.5f}"
    )


if __name__ == "__main__":
    main()
