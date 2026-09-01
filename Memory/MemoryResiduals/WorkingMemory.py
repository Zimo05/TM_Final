from typing import Optional

import torch
from torch import Tensor


def _assert_finite_without_cuda_sync(value: Tensor, message: str) -> None:
    """Validate CUDA tensors without materializing a Python boolean.

    ``bool(torch.isfinite(x).all())`` introduces a device-to-host barrier.
    PyTorch's asynchronous device assertion preserves the same guard on CUDA
    while allowing the event stream to remain queued.  CPU execution keeps the
    eager exception type used by unit tests and debugging.
    """
    finite = torch.isfinite(value).all()
    if value.device.type == "cuda":
        torch._assert_async(finite, message)
    elif not bool(finite):
        raise FloatingPointError(message)


class WorkingMemoryAdapter:
    """
    Sequence-specific transient adapter.

    It is not a persistent model parameter.
    It is reset at the beginning of each sequence.
    """

    def __init__(
        self,
        param_dim: int,
        rho: float = 0.8,
        eta: float = 1e-2,
        clip_grad_norm: Optional[float] = 1.0,
        device: str = "cpu",
    ):
        self.param_dim = param_dim
        self.rho = rho
        self.eta = eta
        self.clip_grad_norm = clip_grad_norm
        self.device = device

        self.delta = torch.zeros(param_dim, device=device)

    def reset(self) -> None:
        self.delta.zero_()

    def to(self, device: torch.device | str) -> "WorkingMemoryAdapter":
        self.device = str(device)
        self.delta = self.delta.to(device)
        return self

    def make_trainable_delta(self) -> Tensor:
        """
        Use this delta in the current forward computation.
        """
        return self.delta.detach().clone().requires_grad_(True)

    def new_batch_state(self, batch_size: int) -> Tensor:
        """Return independent zero-initialized working states ``[B, P]``."""
        if batch_size <= 0:
            raise ValueError("working-memory batch_size must be positive")
        return self.delta.new_zeros(batch_size, self.param_dim)

    def make_trainable_rows(
        self,
        state: Tensor,
        row_indices: Tensor,
    ) -> Tensor:
        """Detach active sequence rows for one wavefront position."""
        if state.ndim != 2 or state.size(1) != self.param_dim:
            raise ValueError("working-memory state must have shape [B, P]")
        if (
            row_indices.ndim != 1
            or row_indices.dtype != torch.long
            or row_indices.device != state.device
        ):
            raise ValueError("row_indices must be device-aligned long [B_active]")
        return (
            state.index_select(0, row_indices)
            .detach()
            .clone()
            .requires_grad_(True)
        )

    @torch.no_grad()
    def update(
        self,
        loss: Tensor,
        delta_used: Tensor,
        adaptation_probability: Optional[Tensor] = None,
    ) -> None:
        """
        delta_used must be the exact tensor used to compute loss.
        """
        grad = torch.autograd.grad(
            loss,
            delta_used,
            retain_graph=False,
            create_graph=False,
        )[0]

        self.update_from_gradient(
            grad,
            adaptation_probability=adaptation_probability,
        )

    @torch.no_grad()
    def update_from_gradient(
        self,
        grad: Tensor,
        adaptation_probability: Optional[Tensor] = None,
    ) -> None:
        """Apply ``rho * delta - eta * p(A) * grad`` to working state."""
        if grad.shape != (self.param_dim,):
            raise ValueError(
                f"working-memory gradient must have shape [{self.param_dim}]"
            )
        grad = grad.detach().to(self.delta)
        _assert_finite_without_cuda_sync(
            grad,
            "working-memory gradient contains NaN or Inf",
        )
        if adaptation_probability is not None:
            probability = torch.as_tensor(
                adaptation_probability,
                device=grad.device,
                dtype=grad.dtype,
            ).detach()
            if probability.numel() != 1:
                raise ValueError("adaptation_probability must be scalar")
            _assert_finite_without_cuda_sync(
                probability,
                "adaptation_probability contains NaN or Inf",
            )
            grad = grad * probability.clamp(0.0, 1.0)

        if self.clip_grad_norm is not None:
            grad_norm = grad.double().norm().clamp_min(1e-300)
            scale = torch.clamp(
                grad_norm.new_tensor(self.clip_grad_norm) / grad_norm,
                max=1.0,
            )
            grad = grad * scale.to(device=grad.device, dtype=grad.dtype)

        self.delta.mul_(self.rho)
        self.delta.add_(grad, alpha=-self.eta)
        _assert_finite_without_cuda_sync(
            self.delta,
            "working-memory update produced NaN or Inf",
        )

    @torch.no_grad()
    def update_batch_rows(
        self,
        state: Tensor,
        row_indices: Tensor,
        grad: Tensor,
        adaptation_probability: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply the scalar WM recurrence independently to active rows.

        Clipping is row-wise, so changing the minibatch size cannot change any
        sequence's working-memory dynamics.
        """
        if state.ndim != 2 or state.size(1) != self.param_dim:
            raise ValueError("working-memory state must have shape [B, P]")
        if (
            row_indices.ndim != 1
            or row_indices.dtype != torch.long
            or row_indices.device != state.device
        ):
            raise ValueError("row_indices must be device-aligned long [B_active]")
        if grad.shape != (row_indices.numel(), self.param_dim):
            raise ValueError(
                "working-memory gradient must have shape [B_active, P]"
            )
        grad = grad.detach().to(state)
        _assert_finite_without_cuda_sync(
            grad,
            "batched working-memory gradient contains NaN or Inf",
        )
        if adaptation_probability is not None:
            probability = torch.as_tensor(
                adaptation_probability,
                device=state.device,
                dtype=state.dtype,
            ).detach().reshape(-1)
            if probability.shape != (row_indices.numel(),):
                raise ValueError(
                    "adaptation_probability must have shape [B_active]"
                )
            _assert_finite_without_cuda_sync(
                probability,
                "batched adaptation_probability contains NaN or Inf",
            )
            grad = grad * probability.clamp(0.0, 1.0).unsqueeze(-1)

        if self.clip_grad_norm is not None:
            grad_norm = grad.double().norm(dim=-1).clamp_min(1e-300)
            scale = torch.clamp(
                grad_norm.new_full(
                    grad_norm.shape,
                    self.clip_grad_norm,
                )
                / grad_norm,
                max=1.0,
            )
            grad = grad * scale.to(device=grad.device, dtype=grad.dtype)[
                :, None
            ]

        previous = state.index_select(0, row_indices)
        next_state = self.rho * previous - self.eta * grad
        _assert_finite_without_cuda_sync(
            next_state,
            "batched working-memory update produced NaN or Inf",
        )
        state.index_copy_(0, row_indices, next_state)
        return next_state
