"""Minimal runnable demonstration of the isolated construction."""

from __future__ import annotations

import torch

from LatentHawkesTree import HawkesTree
from routing_retrieval_investigation import (
    FrontierRoutingConfig,
    FrontierRoutingRetrieval,
)


def main() -> None:
    torch.manual_seed(7)
    tree = HawkesTree(
        z_dim=8,
        node_dim=16,
        num_event_types=8,
        num_basis=2,
        init_depth=3,
        memory_key_dim=8,
    )
    model = FrontierRoutingRetrieval(
        tree,
        FrontierRoutingConfig(frontier_budget=4),
    )
    z_t = torch.randn(2, tree.z_dim)
    output = model(
        z_t,
        working_delta=torch.zeros(tree.param_dim),
        update_memory_state=False,
        update_search_state=False,
    )
    for batch_index, sample in enumerate(output.samples):
        print(
            f"sample={batch_index} "
            f"frontier={sample.node_ids} "
            f"mass={sample.mass.detach().tolist()} "
            f"visited={len(sample.visited_node_ids)}"
        )
    print("effective theta:", tuple(output.effective_params.theta.shape))


if __name__ == "__main__":
    main()
