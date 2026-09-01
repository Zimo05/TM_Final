"""Small construction helpers for attaching the investigation to a tree."""

from __future__ import annotations

from typing import Optional

import torch.nn as nn

from .configuration import FrontierRoutingConfig
from .frontier_model import FrontierRoutingRetrieval


def build_frontier_routing_retrieval(
    tree: nn.Module,
    *,
    frontier_budget: int = 4,
    config: Optional[FrontierRoutingConfig] = None,
) -> FrontierRoutingRetrieval:
    """Build the experimental adapter without mutating the wrapped tree."""
    if config is None:
        config = FrontierRoutingConfig(
            frontier_budget=frontier_budget,
        )
    elif frontier_budget != 4:
        raise ValueError(
            "pass frontier_budget through config when config is provided"
        )
    return FrontierRoutingRetrieval(tree=tree, config=config)
