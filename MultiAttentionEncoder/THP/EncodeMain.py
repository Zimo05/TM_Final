"""
event_type: [B, L]
event_time: [B, L]
enc_output: [B, L, d_model]

"""
import json
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple, Optional

import torch
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent))
from TransformerModel import Constants
from Dataset import get_dataloader
from TransformerModel.Models import MultiAttenEncoder


EventStream = List[Dict[str, float]]


class THPEncoding:
    def __init__(
        self,
        data: str,
        batch_size: int = 16,
        num_workers: int = 0,
        d_model: int = 64,
        d_rnn: int = 256,
        d_inner_hid: int = 128,
        d_k: int = 16,
        d_v: int = 16,
        n_head: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        allow_random_init: bool = False,
    ):
        self.data = Path(data)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.d_model = d_model
        self.d_rnn = d_rnn
        self.d_inner_hid = d_inner_hid
        self.d_k = d_k
        self.d_v = d_v
        self.n_head = n_head
        self.n_layers = n_layers
        self.dropout = dropout
        self.checkpoint = checkpoint
        self.allow_random_init = allow_random_init

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def load_json_data(self, path: Path) -> Tuple[List[str], List["EventStream"], int]:
        with path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if isinstance(raw_data, dict):
            sequence_ids = list(raw_data.keys())
            event_streams = list(raw_data.values())
        elif isinstance(raw_data, list):
            sequence_ids = [str(i) for i in range(len(raw_data))]
            event_streams = raw_data
        else:
            raise TypeError(f"Unsupported input type: {type(raw_data).__name__}")

        if not event_streams:
            raise ValueError("The input file does not contain any event sequences.")

        num_types = max(
            int(event["type_event"])
            for stream in event_streams
            for event in stream
        ) + 1

        return sequence_ids, event_streams, num_types
    
    @staticmethod
    def masked_mean_pool(enc_out: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        enc_out: [B, L, d_model]
        pad_mask: [B, L], True=valid, False=pad
        return: [B, d_model]
        """
        if enc_out.ndim != 3:
            raise ValueError(f"enc_out must have shape [B, L, d_model], got {tuple(enc_out.shape)}")
        if pad_mask.ndim != 2:
            raise ValueError(f"pad_mask must have shape [B, L], got {tuple(pad_mask.shape)}")
        if enc_out.size(0) != pad_mask.size(0) or enc_out.size(1) != pad_mask.size(1):
            raise ValueError(
                f"enc_out and pad_mask shape mismatch: "
                f"enc_out={tuple(enc_out.shape)}, pad_mask={tuple(pad_mask.shape)}"
            )

        lengths = pad_mask.sum(dim=1, keepdim=True).clamp_min(1)  # [B, 1]
        seq_emb = (enc_out * pad_mask.unsqueeze(-1)).sum(dim=1) / lengths
        return seq_emb

    def build_model(self, num_types: int) -> "MultiAttenEncoder":
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")

        checkpoint = None
        checkpoint_config = None
        if self.checkpoint:
            checkpoint = torch.load(self.checkpoint, map_location=self.device)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                checkpoint_config = checkpoint.get("model_config")
        elif not self.allow_random_init:
            raise ValueError(
                "A trained THP checkpoint is required for encoding. "
                "Set allow_random_init=True only for an explicit random-init experiment."
            )

        if checkpoint_config:
            checkpoint_num_types = checkpoint_config.get("num_types")
            if checkpoint_num_types is not None and num_types > checkpoint_num_types:
                raise ValueError(
                    f"Checkpoint supports event types 0..{checkpoint_num_types - 1}, "
                    f"but the input data contains type {num_types - 1}."
                )
            if checkpoint_num_types is not None:
                num_types = checkpoint_num_types
            self.d_model = checkpoint_config["d_model"]
            self.d_rnn = checkpoint_config["d_rnn"]
            self.d_inner_hid = checkpoint_config["d_inner"]
            self.n_layers = checkpoint_config["n_layers"]
            self.n_head = checkpoint_config["n_head"]
            self.d_k = checkpoint_config["d_k"]
            self.d_v = checkpoint_config["d_v"]
            self.dropout = checkpoint_config["dropout"]
            use_time_gap = checkpoint_config.get("use_time_gap", False)
        else:
            # Legacy state_dict checkpoints predate time-gap encoding.
            if checkpoint is not None:
                legacy_state = (
                    checkpoint["state_dict"]
                    if isinstance(checkpoint, dict) and "state_dict" in checkpoint
                    else checkpoint
                )
                embedding_weight = legacy_state.get("encoder.event_emb.weight")
                if embedding_weight is not None:
                    checkpoint_num_types = embedding_weight.size(0) - 1
                    if num_types > checkpoint_num_types:
                        raise ValueError(
                            f"Checkpoint supports event types 0..{checkpoint_num_types - 1}, "
                            f"but the input data contains type {num_types - 1}."
                        )
                    num_types = checkpoint_num_types
            use_time_gap = not self.checkpoint

        model = MultiAttenEncoder(
            num_types=num_types,
            d_model=self.d_model,
            d_rnn=self.d_rnn,
            d_inner=self.d_inner_hid,
            n_layers=self.n_layers,
            n_head=self.n_head,
            d_k=self.d_k,
            d_v=self.d_v,
            dropout=self.dropout,
            use_time_gap=use_time_gap,
        )
        model.to(self.device)

        if checkpoint is not None:
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            model.load_state_dict(checkpoint)
            print(f"[Info] Loaded checkpoint from {self.checkpoint}")

        model.eval()
        return model

    def encode_sequences(
        self,
        model: "MultiAttenEncoder",
        dataloader,
        sequence_ids: Sequence[str],
    ) -> Dict[str, torch.Tensor]:
        encoded_sequences: Dict[str, torch.Tensor] = {}
        offset = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, mininterval=2, desc="  - (Encoding)   ", leave=False):
                event_time, time_gap, event_type = map(lambda x: x.to(self.device), batch)

                # enc_out: [B, L, d_model]
                model_out = model(event_type, event_time, time_gap)

                # Compatible with:
                # MultiAttenEncoder: model(...) -> enc_out
                # Transformer:       model(...) -> (enc_out, prediction)
                enc_out = model_out[0] if isinstance(model_out, tuple) else model_out

                pad_mask = event_type.ne(Constants.PAD)
                valid_lengths = pad_mask.sum(dim=1).tolist()

                seq_emb = self.masked_mean_pool(enc_out, pad_mask)

                for batch_index in range(seq_emb.size(0)):
                    sequence_id = sequence_ids[offset + batch_index]
                    encoded_sequences[sequence_id] = seq_emb[batch_index].detach().cpu()

                offset += len(valid_lengths)

        return encoded_sequences

    def save_encoded(self, encoded_sequences: Dict[str, torch.Tensor], output_path: Path) -> None:
        """Save encoded sequences to disk.

        Tensors are saved as a single .pt file; sequence IDs are saved as a JSON index.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save all tensors stacked + the id list, so we can recover the mapping
        ids = list(encoded_sequences.keys())
        tensors = torch.stack([encoded_sequences[seq_id] for seq_id in ids], dim=0)  # [N, d_model]

        save_dict = {
            "embeddings": tensors,
            "sequence_ids": ids,
        }
        torch.save(save_dict, output_path)
        print(f"[Info] Saved {len(ids)} encoded sequences to {output_path} "
              f"(shape: {tuple(tensors.shape)})")

    def run_encoding(
        self, output_path: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        sequence_ids, event_streams, num_types = self.load_json_data(self.data)
        # Length batching reorders variable-length sequences. Encoding must
        # preserve JSON order so sequence IDs stay aligned with embeddings.
        dataloader = get_dataloader(
            event_streams,
            self.batch_size,
            shuffle=False,
            use_length_batch=False,
            num_workers=self.num_workers,
        )

        model = self.build_model(num_types)
        model_num_types = model.num_types
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            print(f"[Info] DataParallel encoding on {torch.cuda.device_count()} GPUs")
        print("Encoding configuration:")
        print(f"  data={self.data}")
        print(f"  batch_size={self.batch_size}")
        print(f"  d_model={self.d_model}, d_rnn={self.d_rnn}, d_inner_hid={self.d_inner_hid}")
        print(f"  n_head={self.n_head}, n_layers={self.n_layers}")
        print(f"  d_k={self.d_k}, d_v={self.d_v}, dropout={self.dropout}")
        print(f"  checkpoint={self.checkpoint}")
        print(f"  device={self.device}")
        print(f"  sequences={len(sequence_ids)}, num_types={model_num_types}")
        encoded_sequences = self.encode_sequences(model, dataloader, sequence_ids)

        first_id = next(iter(encoded_sequences))
        print(
            f"[Info] Encoded {len(encoded_sequences)} sequences. "
            f"Example: {first_id} -> {tuple(encoded_sequences[first_id].shape)}"
        )

        if output_path is not None:
            self.save_encoded(encoded_sequences, Path(output_path))

        return encoded_sequences
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Encode event sequences with a trained THP model.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    encoder = THPEncoding(
        data=args.data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    encoder.run_encoding(output_path=args.output)
