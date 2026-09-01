import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch

from HawkesBackbone import HawkesFamily
from LatentHawkesTree import HawkesTree


class ConstructMemoryTree:
    """Memory Tree 构建入口，目前负责单 Hawkes Backbone 的冷启动训练。"""

    def __init__(
        self,
        data_path: str,
        checkpoint_path: str = "checkpoints/hawkes_backbone_init.pt",
        num_basis: int = 2,
        decays: Optional[List[float]] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        stability_weight: float = 1e-3,
        grad_clip: float = 5.0,
        seed: int = 0,
        device: Optional[str] = None,
        z_dim: int = 50,
        node_dim: int = 64,
        memory_key_dim: int = 64,
        tree_init_depth: int = 1,
    ):
        self.data_path = Path(data_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.num_basis = num_basis
        self.decays = decays if decays is not None else [0.5, 1.5]
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.stability_weight = stability_weight
        self.grad_clip = grad_clip
        self.seed = seed
        self.z_dim = z_dim
        self.node_dim = node_dim
        self.memory_key_dim = memory_key_dim
        self.tree_init_depth = tree_init_depth

        if len(self.decays) != self.num_basis:
            raise ValueError(
                f"decays 数量必须等于 num_basis: "
                f"{len(self.decays)} != {self.num_basis}"
            )
        if any(decay <= 0 for decay in self.decays):
            raise ValueError("所有 decay 必须大于 0")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.data: List[Dict[str, torch.Tensor]] = []
        self.event_types: List[int] = []
        self.type_to_index: Dict[int, int] = {}
        self.hawkes_backbone: Optional[HawkesFamily] = None
        self.hawkes_tree: Optional[HawkesTree] = None

        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)

        print(f"Using device: {self.device}")

    @staticmethod
    def _parse_values(value, cast):
        """解析 CSV 中逗号分隔的数值，兼容可选的方括号。"""
        text = str(value).strip().strip("[]")
        return [cast(item.strip()) for item in text.split(",") if item.strip()]

    @classmethod
    def _parse_sequence(cls, times_value, types_value):
        times = cls._parse_values(times_value, float)
        event_types = cls._parse_values(types_value, int)

        if not times or not event_types:
            return None
        if len(times) != len(event_types):
            raise ValueError(
                f"事件时间和类型数量不一致: {len(times)} != {len(event_types)}"
            )
        if any(
            current < previous
            for previous, current in zip(times, times[1:])
        ):
            raise ValueError("事件时间必须按非递减顺序排列")
        if times[0] < 0:
            raise ValueError("事件时间不能为负数")

        return {"times": times, "types": event_types}

    def load_sequences(self) -> List[Dict[str, torch.Tensor]]:
        """加载数据，并将原始事件标签映射为 [0, D-1]。"""
        if not self.data_path.is_file():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")

        dataframe = pd.read_csv(self.data_path)
        required_columns = {"event_times", "event_types"}
        missing_columns = required_columns.difference(dataframe.columns)
        if missing_columns:
            raise ValueError(
                f"CSV 缺少必要列: {sorted(missing_columns)}"
            )

        print(f"DataFrame shape: {dataframe.shape}")
        print(f"Columns: {dataframe.columns.tolist()}")

        raw_data = []
        for row_index, row in dataframe.iterrows():
            try:
                sequence = self._parse_sequence(
                    row["event_times"],
                    row["event_types"],
                )
                if sequence is not None:
                    sequence["source_index"] = int(row_index)
                    if (
                        "cluster" in dataframe.columns
                        and not pd.isna(row["cluster"])
                    ):
                        sequence["cluster_id"] = int(row["cluster"])
                    raw_data.append(sequence)
            except (TypeError, ValueError) as error:
                print(f"跳过第 {row_index} 行: {error}")

        if not raw_data:
            raise ValueError("数据集中没有有效事件序列")

        self.event_types = sorted({
            event_type
            for sequence in raw_data
            for event_type in sequence["types"]
        })
        self.type_to_index = {
            event_type: index
            for index, event_type in enumerate(self.event_types)
        }

        # 数据保存在 CPU；训练时逐条送入设备，避免大型数据集占满显存。
        self.data = []
        for sequence in raw_data:
            times = torch.tensor(sequence["times"], dtype=torch.float32)
            types = torch.tensor(
                [
                    self.type_to_index[event_type]
                    for event_type in sequence["types"]
                ],
                dtype=torch.long,
            )
            loaded = {
                "times": times,
                "types": types,
                "T": times[-1],
                "source_index": torch.tensor(
                    sequence["source_index"],
                    dtype=torch.long,
                ),
            }
            if "cluster_id" in sequence:
                loaded["cluster_id"] = torch.tensor(
                    sequence["cluster_id"],
                    dtype=torch.long,
                )
            self.data.append(loaded)

        print(f"加载了 {len(self.data)} 个序列")
        print(f"检测到 {len(self.event_types)} 种事件类型: {self.event_types}")
        if self.event_types != list(range(len(self.event_types))):
            print(f"事件类型映射: {self.type_to_index}")
        print(f"第一个序列长度: {len(self.data[0]['times'])}")

        return self.data

    def _build_hawkes_backbone(self) -> HawkesFamily:
        if not self.event_types:
            raise RuntimeError("请先调用 load_sequences()")

        self.hawkes_backbone = HawkesFamily(
            num_types=len(self.event_types),
            num_basis=self.num_basis,
            init_mu=0.1,
            init_W=0.01,
            decays=torch.tensor(self.decays, dtype=torch.float32),
        ).to(self.device)
        tree = self._build_memory_tree()
        tree.initialize_semantics_from_hawkes(self.hawkes_backbone)
        return self.hawkes_backbone

    def _build_memory_tree(self) -> HawkesTree:
        """Instantiate the semantic tree and its episodic/working memories."""
        if not self.event_types:
            raise RuntimeError("请先调用 load_sequences()")
        self.hawkes_tree = HawkesTree(
            z_dim=self.z_dim,
            node_dim=self.node_dim,
            num_event_types=len(self.event_types),
            num_basis=self.num_basis,
            init_depth=self.tree_init_depth,
            memory_key_dim=self.memory_key_dim,
        ).to(self.device)
        return self.hawkes_tree

    def memory_forward(
        self,
        z_t: torch.Tensor,
        working_delta: Optional[torch.Tensor] = None,
        update_memory_state: bool = True,
    ):
        """Run active-frontier routing, path-union retrieval, and fusion."""
        if self.hawkes_backbone is None:
            self._build_hawkes_backbone()
        if self.hawkes_tree is None:
            self._build_memory_tree()
        return self.hawkes_tree(
            z_t.to(self.device),
            working_delta=working_delta,
            decays=self.hawkes_backbone.decays,
            update_memory_state=update_memory_state,
        )

    def memory_event_nll(
        self,
        sequence: Dict[str, torch.Tensor],
        event_index: int,
        z_t: torch.Tensor,
        working_delta: Optional[torch.Tensor] = None,
        update_memory_state: bool = True,
    ):
        """Compute one event likelihood with memory-composed parameters."""
        if z_t.ndim == 1:
            z_t = z_t.unsqueeze(0)
        if z_t.shape[0] != 1:
            raise ValueError("memory_event_nll expects one sequence/event at a time")

        memory_output = self.memory_forward(
            z_t=z_t,
            working_delta=working_delta,
            update_memory_state=update_memory_state,
        )
        effective_params = memory_output["effective_params"].select(0)
        loss = self.hawkes_backbone.event_NLL(
            sequence=sequence,
            params=effective_params,
            k=event_index,
        )
        return loss, memory_output

    def cold_start_hawkes(
        self,
        num_epochs: int = 10,
    ) -> Path:
        """
        使用全部序列训练单 Hawkes 模型，并生成 HawkesCheckPoint。

        返回:
            保存后的 checkpoint 路径。
        """
        if num_epochs <= 0:
            raise ValueError("num_epochs 必须大于 0")

        if not self.data:
            self.load_sequences()

        model = self._build_hawkes_backbone()
        result = model.cold_start(
            dataset=self.data,
            num_epochs=num_epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            stability_weight=self.stability_weight,
            grad_clip=self.grad_clip,
            checkpoint_path=str(self.checkpoint_path),
            seed=self.seed,
            metadata={
                "event_types": self.event_types,
                "type_to_index": self.type_to_index,
                "data_path": str(self.data_path),
            },
        )
        if self.hawkes_tree is None:
            raise RuntimeError("memory tree was not constructed with the backbone")
        self.hawkes_tree.initialize_semantics_from_hawkes(model)
        return Path(result["checkpoint_path"])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Memory Tree 的 Hawkes Backbone 冷启动训练"
    )
    parser.add_argument(
        "--data-path",
        default="/home/xinye/Hawkes-Memory/Data/tree_15/hawkes_dataset_15.csv",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="checkpoints/hawkes_backbone_init.pt",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    constructor = ConstructMemoryTree(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    constructor.cold_start_hawkes(num_epochs=args.epochs)
