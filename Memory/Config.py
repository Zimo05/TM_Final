"""Topology data shared by the Memory tree."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TreeNode:
    """
    Pure topology node.

    Trainable tensors are NOT stored here directly.
    They are stored in HawkesTree.ParameterDict / ModuleDict.
    Episodic entries are owned by TreeEpisodicMemory, keyed by node_id.
    """
    node_id: str
    parent: Optional[str] = None
    left: Optional[str] = None
    right: Optional[str] = None
    depth: int = 0

    # The queue stores canonical MemoryResiduals.MemoryItem instances.  Use an
    # object annotation here to keep the topology module free of memory-layer
    # imports and their circular dependency.
    split_queue: list[object] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None
