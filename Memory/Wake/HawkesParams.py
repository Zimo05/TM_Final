from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class HawkesParams:
    """
    Unconstrained Hawkes parameters.

    mu_tilde: [D] or [..., D]
    W_tilde:  [D, D, M] or [..., D, D, M]
    """
    mu_tilde: torch.Tensor
    W_tilde: torch.Tensor

    @property
    def raw_mu(self) -> torch.Tensor:
        return self.mu_tilde

    @property
    def raw_W(self) -> torch.Tensor:
        return self.W_tilde

    def mu(self) -> torch.Tensor:
        return F.softplus(self.mu_tilde)

    def W(self) -> torch.Tensor:
        return F.softplus(self.W_tilde)

    def detach(self) -> "HawkesParams":
        return HawkesParams(
            self.mu_tilde.detach(),
            self.W_tilde.detach(),
        )

    def clone_detached(self, requires_grad: bool = False) -> "HawkesParams":
        mu = self.mu_tilde.detach().clone()
        W = self.W_tilde.detach().clone()
        if requires_grad:
            mu.requires_grad_(True)
            W.requires_grad_(True)
        return HawkesParams(mu, W)


def lowrank_project_hawkes_residual(
    delta: HawkesParams,
    rank: int,
) -> HawkesParams:
    """Apply the shared Memory residual projection ``P_r``.

    The base-intensity residual is retained exactly.  Each Hawkes excitation
    matrix ``W[:, :, m]`` is independently truncated to rank ``r``.  Keeping
    this operation here makes cold-start residual signatures and online
    episodic-memory writes use the same residual geometry.
    """
    if rank < 0:
        raise ValueError("rank must be non-negative")
    W = delta.W_tilde
    if W.ndim not in (3, 4) or W.size(-3) != W.size(-2):
        raise ValueError(
            "W_tilde must have shape [D, D, M] or [Q, D, D, M]"
        )

    # ``torch.linalg.svd`` supports leading batch dimensions.  Keeping the
    # candidate dimension in the operation is important for Wake: every
    # residual candidate has an independent gradient, but all candidates can
    # share the same GPU kernel launch.
    projected = []
    for basis_index in range(W.size(-1)):
        matrix = W[..., basis_index]
        U, singular_values, Vh = torch.linalg.svd(
            matrix,
            full_matrices=False,
        )
        retained_rank = min(int(rank), int(singular_values.size(-1)))
        if retained_rank == 0:
            matrix_projected = torch.zeros_like(matrix)
        else:
            matrix_projected = (
                U[..., :, :retained_rank]
                * singular_values[..., :retained_rank].unsqueeze(-2)
            ) @ Vh[..., :retained_rank, :]
        projected.append(matrix_projected)

    return HawkesParams(
        delta.mu_tilde,
        torch.stack(projected, dim=-1),
    )
