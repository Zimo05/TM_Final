import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

_THP_DIR = str(Path(__file__).resolve().parent)
if _THP_DIR not in sys.path:
    sys.path.insert(0, _THP_DIR)


def positive_beta(raw_beta):
    """Map the learned unconstrained intensity scale to a positive value."""
    return F.softplus(raw_beta) + 1e-6


def intensity(logits, beta):
    """Positive event intensity with a learned softplus scale."""
    return F.softplus(beta * logits) / beta


def log_likelihood(model, data, time, types, num_samples=16):
    """Log-likelihood of next events conditioned only on previous history.

    The hidden state at position i parameterizes intensities on
    (time_i, time_{i+1}]. The survival integral sums intensities over every
    event type; using only the observed type gives an invalid point-process
    likelihood.
    """
    if types.size(1) < 2:
        zeros = data.new_zeros(types.size(0))
        return zeros, zeros

    valid_mask = types[:, 1:].ne(0)
    time_gap = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
    base_logits = model.linear(data[:, :-1, :])
    beta = positive_beta(model.beta)

    event_logits = base_logits + model.alpha * time_gap.unsqueeze(-1)
    event_intensity = intensity(event_logits, beta)
    target = (types[:, 1:] - 1).clamp_min(0)
    target_intensity = event_intensity.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    event_ll = torch.log(target_intensity.clamp_min(1e-9))
    event_ll = (event_ll * valid_mask).sum(dim=-1)

    # Deterministic midpoint quadrature avoids Monte Carlo noise while
    # integrating the total intensity across each inter-event interval.
    sample_fractions = (
        torch.arange(num_samples, device=data.device, dtype=data.dtype) + 0.5
    ) / num_samples
    sampled_elapsed = time_gap.unsqueeze(-1) * sample_fractions
    sampled_logits = base_logits.unsqueeze(2) + model.alpha * sampled_elapsed.unsqueeze(-1)
    total_intensity = intensity(sampled_logits, beta).sum(dim=-1).mean(dim=2)
    non_event_ll = (total_intensity * time_gap * valid_mask).sum(dim=-1)

    return event_ll, non_event_ll


def type_loss(prediction, types, loss_func):
    """ Event prediction loss, cross entropy or label smoothing.

    Returns
    -------
    loss        : summed loss over all valid (non-pad) tokens
    correct_num : number of correctly predicted valid tokens
    valid_num   : number of valid (non-pad) tokens  (for normalization)
    top3_num    : number whose target is in the top-3 predictions
    confusion   : [num_types, num_types] confusion matrix
    """

    # convert [1,2,3] based types to [0,1,2]; also convert padding events to -1
    truth = types[:, 1:] - 1
    prediction = prediction[:, :-1, :]

    # mask of valid (non-pad) target positions: pad was 0 -> truth == -1
    valid_mask = truth.ne(-1)
    valid_num = valid_mask.sum()

    pred_type = torch.max(prediction, dim=-1)[1]
    # only count correctness on valid positions
    correct_num = torch.sum((pred_type == truth) & valid_mask)
    top_k = min(3, prediction.size(-1))
    top_types = prediction.topk(top_k, dim=-1).indices
    top3_num = ((top_types == truth.unsqueeze(-1)).any(dim=-1) & valid_mask).sum()

    valid_truth = truth[valid_mask]
    valid_pred = pred_type[valid_mask]
    num_types = prediction.size(-1)
    confusion = torch.bincount(
        valid_truth * num_types + valid_pred,
        minlength=num_types * num_types,
    ).reshape(num_types, num_types)

    # compute cross entropy loss
    if isinstance(loss_func, LabelSmoothingLoss):
        loss = loss_func(prediction, truth)
    else:
        loss = loss_func(prediction.transpose(1, 2), truth)

    loss = torch.sum(loss)
    return loss, correct_num, valid_num, top3_num, confusion


def time_loss(prediction, time_gap, types):
    """ Time prediction loss. """

    prediction = prediction.squeeze(-1)
    true = time_gap[:, 1:]
    prediction = prediction[:, :-1]
    valid_mask = types[:, 1:].ne(0)

    # event time gap prediction
    diff = prediction - true
    se = torch.sum(diff * diff * valid_mask)
    return se


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-1, weight=None):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        self.eps = label_smoothing
        self.num_classes = tgt_vocab_size
        self.ignore_index = ignore_index
        # per-class weight tensor of shape [num_classes] or None
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, output, target):
        """
        output (FloatTensor): (batch_size) x n_classes
        target (LongTensor): batch_size
        """

        non_pad_mask = target.ne(self.ignore_index).float()

        # Clone to avoid mutating the caller's tensor
        target = target.clone()
        target[target.eq(self.ignore_index)] = 0
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        # Standard label smoothing: smoothed = one_hot * (1 - eps) + eps / num_classes
        # This ensures the smoothed distribution sums to 1
        one_hot = one_hot * (1 - self.eps) + self.eps / self.num_classes

        log_prb = F.log_softmax(output, dim=-1)
        loss = -(one_hot * log_prb).sum(dim=-1)

        # Match CrossEntropyLoss semantics: weight by the target class, not
        # every class in the smoothed distribution.
        if self.weight is not None:
            loss = loss * self.weight[target]

        loss = loss * non_pad_mask
        return loss
