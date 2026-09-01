from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence

import torch.nn.functional as F
import torch.nn as nn
import torch

EPS = 1e-8
EVENT_TIME_FEATURES_KEY = "_event_time_features"
HAWKES_HISTORY_STATS_KEY = "_hawkes_history_stats"
HAWKES_INTERVAL_STATS_KEY = "_hawkes_interval_stats"
HAWKES_CACHE_SIGNATURE_KEY = "_hawkes_cache_signature"

def inv_softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(x).clamp_min(EPS))

class HawkesFamily(nn.Module):
    """
    Multivariate Hawkes parameters with exponential basis kernels.
    
    Args:
        num_types: Number of event types D
        num_basis: Number of basis kernels M
        init_mu: Initial value for base intensity
        init_W: Initial value for impact matrix
        decays: Decay rates for basis kernels [M]
    """
    def __init__(
        self,
        num_types: int,
        num_basis: int,
        init_mu: float = 0.1,
        init_W: float = 0.01,
        decays: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_types = num_types
        self.num_basis = num_basis

        # 修复1: 使用正确的参数名 raw_mu 和 raw_W
        raw_mu = inv_softplus(torch.full((num_types,), init_mu))
        raw_W = inv_softplus(torch.full((num_types, num_types, num_basis), init_W))

        self.raw_mu = nn.Parameter(raw_mu)  # 修复：保存为属性
        self.raw_W = nn.Parameter(raw_W)    # 修复：保存为属性

        if decays is None:
            decays = torch.linspace(0.1, 1.0, num_basis)
        self.register_buffer('decays', decays)
        self._cache_signature = (
            int(num_types),
            tuple(float(value) for value in decays.detach().cpu().reshape(-1)),
        )

    @classmethod
    def from_cold_start_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> tuple["HawkesFamily", Dict]:
        """Restore a backbone written by :meth:`cold_start`.

        Returning the payload as well as the model keeps validation/training
        metadata available to callers without attaching it to the module.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Hawkes cold-start checkpoint not found: {checkpoint_path}"
            )
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(payload, dict):
            raise TypeError("Hawkes cold-start checkpoint must contain a dict")

        required = {"model_state_dict", "num_types", "num_basis", "decays"}
        missing = required.difference(payload)
        if missing:
            raise KeyError(
                "Hawkes cold-start checkpoint is missing keys: "
                f"{sorted(missing)}"
            )

        decays = torch.as_tensor(payload["decays"], dtype=torch.float32)
        model = cls(
            num_types=int(payload["num_types"]),
            num_basis=int(payload["num_basis"]),
            decays=decays,
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.to(torch.device(device))
        return model, payload
    
    def mu(self):
        """Shape: [D]"""
        return F.softplus(self.raw_mu)
    
    def W(self):
        """Shape: [D, D, M]"""
        return F.softplus(self.raw_W)

    @property
    def cache_signature(self):
        return self._cache_signature

    @staticmethod
    def _positive_parameter(parameters, name: str) -> torch.Tensor:
        """Accept both legacy mu()/W() methods and effective tensor fields."""
        value = getattr(parameters, name)
        return value() if callable(value) else value

    def _parameter_decays(self, parameters) -> torch.Tensor:
        decays = getattr(parameters, "decays", None)
        return self.decays if decays is None else decays
    
    def unpack_sequence(self, sequence):
        """
        Unpack sequence dictionary.
        
        Args:
            sequence: {
                "times": Tensor[P],
                "types": Tensor[P],
                "T": optional scalar
            }
        Returns:
            times: Tensor[P]
            types: Tensor[P] (long)
            T: scalar or None
        """
        times = sequence["times"]
        types = sequence["types"].long()
        
        if "T" in sequence:
            T = sequence["T"]
        else:
            T = None
        
        return times, types, T

    @torch.no_grad()
    def prepare_sequence_cache(
        self,
        sequence: Mapping[str, torch.Tensor],
        *,
        inplace: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Cache all parameter-independent event-sequence features.

        The cached Hawkes tensors depend only on times, types, fixed decays,
        and the event vocabulary. They remain valid while mu/W and the Memory
        Tree change, so they can be reused across events and epochs.
        """
        target = (
            sequence
            if inplace and isinstance(sequence, MutableMapping)
            else dict(sequence)
        )
        times, types, _ = self.unpack_sequence(target)
        types = types.long()
        if times.ndim != 1 or types.ndim != 1 or times.shape != types.shape:
            raise ValueError("times/types must be aligned one-dimensional tensors")
        if times.numel() and bool((times[1:] < times[:-1]).any().item()):
            raise ValueError("event times must be non-decreasing")
        if types.numel() and (
            int(types.min()) < 0 or int(types.max()) >= self.num_types
        ):
            raise ValueError("event types are outside the model vocabulary")

        event_count = int(times.numel())
        expected_hawkes_shape = (
            event_count,
            self.num_types,
            self.num_basis,
        )
        cache_is_current = (
            target.get(HAWKES_CACHE_SIGNATURE_KEY) == self._cache_signature
            and isinstance(target.get(HAWKES_HISTORY_STATS_KEY), torch.Tensor)
            and target[HAWKES_HISTORY_STATS_KEY].shape == expected_hawkes_shape
            and isinstance(target.get(HAWKES_INTERVAL_STATS_KEY), torch.Tensor)
            and target[HAWKES_INTERVAL_STATS_KEY].shape == expected_hawkes_shape
        )
        time_cache_is_current = (
            isinstance(target.get(EVENT_TIME_FEATURES_KEY), torch.Tensor)
            and target[EVENT_TIME_FEATURES_KEY].shape == (event_count, 2)
        )

        if not time_cache_is_current:
            previous = (
                torch.cat([times.new_zeros(1), times[:-1]])
                if event_count
                else times
            )
            delta_t = (times - previous).clamp_min(0.0)
            target[EVENT_TIME_FEATURES_KEY] = torch.stack(
                [torch.log1p(times), torch.log1p(delta_t)],
                dim=-1,
            )

        if not cache_is_current:
            decays = self.decays.to(device=times.device, dtype=times.dtype)
            one_hot_types = torch.nn.functional.one_hot(
                types,
                num_classes=self.num_types,
            ).to(dtype=times.dtype)

            # [target k, source j]. Intensity uses the original strict-time
            # rule, so earlier array positions at the same timestamp do not
            # excite one another.
            time_delta = times[:, None] - times[None, :]
            positions = torch.arange(event_count, device=times.device)
            source_before = positions[None, :] < positions[:, None]
            strict_history = source_before & time_delta.gt(0.0)
            history_kernels = torch.exp(
                -time_delta.clamp_min(0.0).unsqueeze(-1)
                * decays.reshape(1, 1, -1)
            ) * strict_history.unsqueeze(-1)
            history_stats = torch.einsum(
                "kjm,jc->kcm",
                history_kernels,
                one_hot_types,
            )

            # For interval [t_{k-1}, t_k], every source j < k starts
            # contributing at max(t_{k-1}, t_j). This reproduces
            # interval_integral exactly, including repeated timestamps.
            previous_times = (
                torch.cat([times.new_zeros(1), times[:-1]])
                if event_count
                else times
            )
            lower = torch.maximum(
                previous_times[:, None],
                times[None, :],
            )
            interval_valid = source_before & lower.lt(times[:, None])
            left_delta = (lower - times[None, :]).clamp_min(0.0)
            right_delta = time_delta.clamp_min(0.0)
            interval_kernels = (
                torch.exp(
                    -left_delta.unsqueeze(-1) * decays.reshape(1, 1, -1)
                )
                - torch.exp(
                    -right_delta.unsqueeze(-1) * decays.reshape(1, 1, -1)
                )
            ) / decays.reshape(1, 1, -1)
            interval_kernels = (
                interval_kernels * interval_valid.unsqueeze(-1)
            )
            interval_stats = torch.einsum(
                "kjm,jc->kcm",
                interval_kernels,
                one_hot_types,
            )

            target[HAWKES_HISTORY_STATS_KEY] = history_stats
            target[HAWKES_INTERVAL_STATS_KEY] = interval_stats
            target[HAWKES_CACHE_SIGNATURE_KEY] = self._cache_signature

        return dict(target)

    def _cached_event_statistics(
        self,
        sequence: Mapping[str, torch.Tensor],
        k: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if sequence.get(HAWKES_CACHE_SIGNATURE_KEY) != self._cache_signature:
            return None
        history = sequence.get(HAWKES_HISTORY_STATS_KEY)
        interval = sequence.get(HAWKES_INTERVAL_STATS_KEY)
        if not isinstance(history, torch.Tensor) or not isinstance(
            interval, torch.Tensor
        ):
            return None
        expected = (len(sequence["times"]), self.num_types, self.num_basis)
        if history.shape != expected or interval.shape != expected:
            return None
        return (
            history[k].to(device=device, dtype=dtype),
            interval[k].to(device=device, dtype=dtype),
        )

    def intensity_at_cached_event(self, sequence, k, parameters):
        """Return event intensity, using cached decay statistics when present."""
        times, types, _ = self.unpack_sequence(sequence)
        device = parameters.raw_mu.device
        dtype = parameters.raw_mu.dtype
        cached = self._cached_event_statistics(
            sequence,
            k,
            device=device,
            dtype=dtype,
        )
        if cached is None:
            history = {
                "times": times[:k],
                "types": types[:k],
            }
            return self.intensity_at_event(history, times[k], parameters)

        history_stats, _ = cached
        mu = self._positive_parameter(parameters, "mu").to(
            device=device,
            dtype=dtype,
        )
        W = self._positive_parameter(parameters, "W").to(
            device=device,
            dtype=dtype,
        )
        contribution = (
            W * history_stats.unsqueeze(0)
        ).sum(dim=(1, 2))
        return (mu + contribution).clamp_min(EPS)
    
    def intensity_at_event(self, history, t, parameters):
        """
        Args:
            history: {
                "times": Tensor[P],
                "types": Tensor[P]
            }
            Prefix events before current time t.
            
            t: Scalar tensor or float.
            
        Returns:
            lambda_t: Tensor [D]
        """
        hist_event, hist_type, _ = self.unpack_sequence(history)
        device = parameters.raw_mu.device
        dtype = hist_event.dtype

        hist_event = hist_event.to(device)
        hist_type = hist_type.to(device)
        mu = self._positive_parameter(parameters, "mu").to(device)
        W = self._positive_parameter(parameters, "W").to(device)
        decays = self._parameter_decays(parameters).to(device)
        t = torch.as_tensor(t, device=device, dtype=dtype)

        lambda_t = mu.clone()

        if hist_event.numel() == 0:
            return lambda_t.clamp_min(EPS) # 没有历史强度也要把baseline拉起来减少不稳定
        
        mask = hist_event < t
        past_times = hist_event[mask]
        past_types = hist_type[mask]

        if past_times.numel() == 0:
            return lambda_t.clamp_min(EPS) # 没有历史强度也要把baseline拉起来减少不稳定
        

        dt = t - past_times  # [P]
        exp_kernels = torch.exp(-dt[:, None] * decays[None, :])  # [P, M]

        # W[:, past_types, :] has shape [D, P, M]
        # contribution[d] = sum_p sum_m W[d, c_p, m] * basis[p, m]
        contribution = (W[:, past_types, :] * exp_kernels[None, :, :]).sum(dim=(1, 2))
        lambda_t += contribution
        return lambda_t.clamp_min(EPS)

    def interval_integral(self, t_prev, t, history, parameters):
        """
        For intervel: [a,b]=[t_prev, t]
        Calculate: sum_{d=1}^D ∫_{t_prev}^{t} lambda_d(tau | H_tau) d tau.
        Args:
            t_prev:
                Left endpoint of interval.

            t:
                Right endpoint of interval.

            history:
                Events available before the right endpoint t.
                Usually sequence[:k] when computing the kth event loss.

            params:
                HawkesParams object.

        return:
            integral: Scalar tensor.

        """

        hist_event, hist_type, _ = self.unpack_sequence(history)
        device = parameters.raw_mu.device
        dtype = parameters.raw_mu.dtype

        hist_times = hist_event.to(device=device, dtype=dtype)
        t_prev = torch.as_tensor(t_prev, device=device, dtype=dtype)
        t = torch.as_tensor(t, device=device, dtype=dtype)
        hist_types = hist_type.to(device=device)
        mu = self._positive_parameter(parameters, "mu").to(device)
        W = self._positive_parameter(parameters, "W").to(device)
        decays = self._parameter_decays(parameters).to(device)
        
        
        duration = (t - t_prev).clamp_min(0.0)
        baseline_integral = mu.sum() * duration

        if hist_times.numel() == 0:
            return baseline_integral

        lower = torch.maximum(t_prev.expand_as(hist_times), hist_times)  # [P]
        # Keep only events whose effect overlaps with [t_prev, t]
        valid = lower < t

        past_times = hist_times[valid]
        past_types = hist_types[valid]
        lower = lower[valid]

        # left_exp[p, m] = exp(-delta_m * (lower_p - t_p))
        left_exp = torch.exp(-(lower - past_times)[:, None] * decays[None, :])  # [P, M]

        # right_exp[p, m] = exp(-delta_m * (t - t_p))
        right_exp = torch.exp(-(t - past_times)[:, None] * decays[None, :])     # [P, M]

        # ∫ exp(-delta_m(tau - t_p)) d tau
        basis_integral = (left_exp - right_exp) / decays[None, :]               # [P, M]

        # W[:, past_types, :] -> [D, P, M]
        excitation_integral = (
            W[:, past_types, :] * basis_integral[None, :, :]
        ).sum()

        return baseline_integral + excitation_integral
    
    def event_NLL(self, sequence, params, k):
        times, types, _ = self.unpack_sequence(sequence)

        device = params.raw_mu.device
        dtype = params.raw_mu.dtype

        times = times.to(device=device, dtype=dtype)
        types = types.to(device=device)

        t_k = times[k]
        c_k = types[k]

        cached = self._cached_event_statistics(
            sequence,
            k,
            device=device,
            dtype=dtype,
        )
        if cached is None:
            # lambda_d(t_k | H_{t_k}) for all d
            lambda_t = self.intensity_at_cached_event(sequence, k, params)
            mu = None
            W = None
        else:
            history_stats, _ = cached
            mu = self._positive_parameter(params, "mu").to(
                device=device,
                dtype=dtype,
            )
            W = self._positive_parameter(params, "W").to(
                device=device,
                dtype=dtype,
            )
            lambda_t = (
                mu + (W * history_stats.unsqueeze(0)).sum(dim=(1, 2))
            ).clamp_min(EPS)

        # -log lambda_{c_k}(t_k)
        log_intensity_loss = -torch.log(lambda_t[c_k].clamp_min(EPS))

        # interval [t_{k-1}, t_k]
        if k == 0:
            t_prev = times.new_tensor(0.0)
        else:
            t_prev = times[k - 1]

        if cached is None:
            history = {
                "times": times[:k],
                "types": types[:k],
            }
            integral_loss = self.interval_integral(
                t_prev,
                t_k,
                history,
                params,
            )
        else:
            _, interval_stats = cached
            duration = (t_k - t_prev).clamp_min(0.0)
            integral_loss = (
                mu.sum() * duration
                + (W * interval_stats.unsqueeze(0)).sum()
            )

        return log_intensity_loss + integral_loss
    
    def sequence_NLL(self, sequence, params, include_tail: bool = True) -> torch.Tensor:
        """Vectorized negative log-likelihood for one complete sequence.

        This is algebraically equivalent to summing :meth:`event_NLL` over all
        events and then adding the optional tail integral.  Computing all
        pairwise history contributions in one tensor avoids launching hundreds
        of tiny GPU operations per sequence during backbone cold start.
        """
        times, types, T = self.unpack_sequence(sequence)

        device = params.raw_mu.device
        dtype = params.raw_mu.dtype

        times = times.to(device=device, dtype=dtype)
        types = types.to(device=device)

        K = times.numel()

        # Empty sequence: only baseline no-event likelihood over [0, T]
        if K == 0:
            if T is None:
                return params.raw_mu.sum() * 0.0

            empty_history = {
                "times": times,
                "types": types,
            }
            return self.interval_integral(
                times.new_tensor(0.0),
                torch.as_tensor(T, device=device, dtype=dtype),
                empty_history,
                params,
            )

        mu = self._positive_parameter(params, "mu").to(device=device, dtype=dtype)
        W = self._positive_parameter(params, "W").to(device=device, dtype=dtype)
        decays = self._parameter_decays(params).to(device=device, dtype=dtype)

        # Pair (k, j) represents the contribution of source event j to target
        # event k. The original implementation uses a strict time mask, so
        # simultaneous events do not excite one another even when j < k.
        time_delta = times[:, None] - times[None, :]                 # [K, K]
        history_mask = time_delta.gt(0)
        kernels = torch.exp(
            -time_delta.clamp_min(0).unsqueeze(-1) * decays
        ) * history_mask.unsqueeze(-1)                              # [K, K, M]
        pair_weights = W[types[:, None], types[None, :], :]         # [K, K, M]
        target_intensity = mu[types] + (pair_weights * kernels).sum(dim=(1, 2))
        log_intensity_loss = -torch.log(target_intensity.clamp_min(EPS)).sum()

        # The interval terms telescope over [0, observation_end]. Each source
        # event contributes its exponential-kernel integral from its event time
        # to the observation end. ``-expm1`` is stable for short horizons.
        observation_end = times[-1]
        if include_tail and T is not None:
            observation_end = torch.maximum(
                observation_end,
                torch.as_tensor(T, device=device, dtype=dtype),
            )
        baseline_integral = mu.sum() * observation_end
        horizon = (observation_end - times).clamp_min(0)             # [K]
        basis_integral = -torch.expm1(-horizon[:, None] * decays) / decays
        source_weights = W[:, types, :].sum(dim=0)                  # [K, M]
        excitation_integral = (source_weights * basis_integral).sum()

        return log_intensity_loss + baseline_integral + excitation_integral

    def stability_regularizer(
        self,
        params,
        tau_stab: float = 0.99,
        power_iterations: int = 20,
    ) -> torch.Tensor:
        """
        A[d, d'] = sum_m W[d, d', m] / decay_m

        Penalize spectral radius larger than tau_stab.

        Args:
            params:
                HawkesParams object.

            tau_stab:
                Stability threshold. Usually < 1.

        Returns:
            reg:
                Scalar tensor.
        """
        W = self._positive_parameter(params, "W")
        decays = self._parameter_decays(params)

        # Integrated excitation matrix A: [D, D]
        A = (W / decays[None, None, :]).sum(dim=-1)

        # A is non-negative, so its spectral radius is the Perron eigenvalue.
        # Power iteration avoids the unstable eigvals backward pass when A has
        # repeated eigenvalues (which happens with the uniform initialization).
        vector = torch.ones(
            A.shape[0],
            device=A.device,
            dtype=A.dtype,
        )
        vector = vector / vector.sum().clamp_min(EPS)

        for _ in range(power_iterations):
            vector = A @ vector
            vector = vector / vector.sum().clamp_min(EPS)

        # With sum(vector) == 1, sum(A @ vector) converges to rho(A).
        rho = (A @ vector).sum()

        reg = torch.relu(rho - tau_stab).pow(2)

        return reg

    def integrated_excitation_matrix(self) -> torch.Tensor:
        """Return A[d, d'] = sum_m W[d, d', m] / decay_m."""
        return (self.W() / self.decays[None, None, :]).sum(dim=-1)

    @staticmethod
    def _move_sequence(
        sequence: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        return {
            key: value.to(device)
            for key, value in sequence.items()
        }

    def _dataset_nll_per_event(
        self,
        dataset: Sequence[Dict[str, torch.Tensor]],
    ) -> float:
        """Evaluate mean NLL per event without constructing gradients."""
        if not dataset:
            return float("nan")

        device = self.raw_mu.device
        total_nll = 0.0
        total_events = 0

        self.eval()
        with torch.no_grad():
            for cpu_sequence in dataset:
                sequence = self._move_sequence(cpu_sequence, device)
                nll = self.sequence_NLL(sequence, self)
                if not torch.isfinite(nll):
                    raise FloatingPointError("验证阶段出现非有限 NLL")

                total_nll += float(nll.cpu())
                total_events += int(sequence["times"].numel())

        return total_nll / max(total_events, 1)

    def cold_start(
        self,
        dataset: Sequence[Dict[str, torch.Tensor]],
        num_epochs: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        stability_weight: float = 1e-3,
        excitation_l1_weight: float = 1e-5,
        tau_stab: float = 0.99,
        grad_clip: float = 5.0,
        validation_fraction: float = 0.1,
        patience: int = 8,
        min_delta: float = 1e-4,
        checkpoint_path: str = "checkpoints/hawkes_backbone_init.pt",
        seed: int = 0,
        metadata: Optional[Dict] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Train a single Hawkes model before constructing the memory tree.

        The routine provides a minimal robust training structure:
        deterministic train/validation split, event-normalized NLL, AdamW,
        stability and sparsity regularization, gradient clipping, finite-value
        checks, early stopping, and restoration of the best validation model.
        """
        if not dataset:
            raise ValueError("cold_start 需要至少一个有效序列")
        if num_epochs <= 0:
            raise ValueError("num_epochs 必须大于 0")
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("validation_fraction 必须位于 [0, 1)")
        if patience <= 0:
            raise ValueError("patience 必须大于 0")

        device = self.raw_mu.device
        for sequence_index, sequence in enumerate(dataset):
            if "times" not in sequence or "types" not in sequence:
                raise ValueError(
                    f"序列 {sequence_index} 缺少 times 或 types"
                )
            times = sequence["times"]
            types = sequence["types"]
            if times.numel() != types.numel():
                raise ValueError(
                    f"序列 {sequence_index} 的 times/types 长度不一致"
                )
            if types.numel() and (
                int(types.min()) < 0 or int(types.max()) >= self.num_types
            ):
                raise ValueError(
                    f"序列 {sequence_index} 的事件类型超出 "
                    f"[0, {self.num_types - 1}]"
                )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        order = torch.randperm(len(dataset), generator=generator).tolist()

        if len(dataset) > 1 and validation_fraction > 0:
            validation_size = max(
                1,
                int(round(len(dataset) * validation_fraction)),
            )
            validation_size = min(validation_size, len(dataset) - 1)
        else:
            validation_size = 0

        validation_indices = order[:validation_size]
        train_indices = order[validation_size:]
        train_data = [dataset[index] for index in train_indices]
        validation_data = [dataset[index] for index in validation_indices]

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        train_history: List[float] = []
        validation_history: List[float] = []
        best_metric = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        best_state = None

        for epoch in range(1, num_epochs + 1):
            self.train()
            epoch_nll = 0.0
            epoch_events = 0
            max_gradient_norm = 0.0

            epoch_order = torch.randperm(
                len(train_data),
                generator=generator,
            ).tolist()

            for sequence_index in epoch_order:
                sequence = self._move_sequence(
                    train_data[sequence_index],
                    device,
                )
                event_count = int(sequence["times"].numel())

                optimizer.zero_grad(set_to_none=True)
                nll = self.sequence_NLL(sequence, self)
                nll_per_event = nll / max(event_count, 1)
                stability = self.stability_regularizer(
                    self,
                    tau_stab=tau_stab,
                )
                excitation_l1 = self.W().mean()
                loss = (
                    nll_per_event
                    + stability_weight * stability
                    + excitation_l1_weight * excitation_l1
                )

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "训练阶段出现非有限 loss；请检查时间尺度和 decay"
                    )

                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    self.parameters(),
                    grad_clip,
                    error_if_nonfinite=True,
                )
                max_gradient_norm = max(
                    max_gradient_norm,
                    float(gradient_norm.detach().cpu()),
                )
                optimizer.step()

                if not all(
                    torch.isfinite(parameter).all()
                    for parameter in self.parameters()
                ):
                    raise FloatingPointError("参数更新后出现 NaN 或 Inf")

                epoch_nll += float(nll.detach().cpu())
                epoch_events += event_count

            train_nll = epoch_nll / max(epoch_events, 1)
            validation_nll = (
                self._dataset_nll_per_event(validation_data)
                if validation_data
                else train_nll
            )
            train_history.append(train_nll)
            validation_history.append(validation_nll)

            if verbose:
                print(
                    f"[Hawkes Cold Start][{epoch:03d}/{num_epochs:03d}] "
                    f"train_nll/event={train_nll:.6f} "
                    f"val_nll/event={validation_nll:.6f} "
                    f"max_grad={max_gradient_norm:.4f}"
                )

            if validation_nll < best_metric - min_delta:
                best_metric = validation_nll
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.state_dict().items()
                }
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    if verbose:
                        print(
                            f"[Hawkes Cold Start] early stopping at epoch "
                            f"{epoch}; best epoch={best_epoch}"
                        )
                    break

        if best_state is None:
            raise RuntimeError("冷启动训练未产生有效模型")

        self.load_state_dict(best_state)
        self.to(device)

        with torch.no_grad():
            excitation_matrix = self.integrated_excitation_matrix()
            # Evaluation only: eigvals is safe because no backward is required.
            spectral_radius = float(
                torch.linalg.eigvals(excitation_matrix)
                .abs()
                .max()
                .real
                .cpu()
            )
            minimum_intensity = float(self.mu().min().cpu())

        result = {
            "best_epoch": best_epoch,
            "best_validation_nll": best_metric,
            "train_history": train_history,
            "validation_history": validation_history,
            "spectral_radius": spectral_radius,
            "minimum_baseline_intensity": minimum_intensity,
            "excitation_matrix": excitation_matrix.detach().cpu(),
            "train_size": len(train_data),
            "validation_size": len(validation_data),
        }

        checkpoint = {
            "format_version": 1,
            "model_class": type(self).__name__,
            "model_state_dict": {
                name: value.detach().cpu()
                for name, value in self.state_dict().items()
            },
            "num_types": self.num_types,
            "num_basis": self.num_basis,
            "decays": self.decays.detach().cpu(),
            "training_config": {
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "stability_weight": stability_weight,
                "excitation_l1_weight": excitation_l1_weight,
                "tau_stab": tau_stab,
                "grad_clip": grad_clip,
                "validation_fraction": validation_fraction,
                "patience": patience,
                "min_delta": min_delta,
                "seed": seed,
            },
            "training_result": result,
            "metadata": metadata or {},
        }

        output_path = Path(checkpoint_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(output_path)

        result["checkpoint_path"] = str(output_path.resolve())
        if verbose:
            print(
                f"[Hawkes Cold Start] checkpoint saved to "
                f"{result['checkpoint_path']}"
            )
            print(
                f"[Hawkes Cold Start] spectral_radius={spectral_radius:.6f}, "
                f"min_mu={minimum_intensity:.6e}"
            )

        return result
