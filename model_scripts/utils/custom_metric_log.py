# per_task_metrics.py
import torch
import torchmetrics


class PerTaskMAE(torchmetrics.Metric):
    """MAE for a single task index, compatible with Chemprop's MPNN._evaluate_batch."""

    def __init__(self, task_idx: int, alias: str | None = None):
        super().__init__()
        self.task_idx = int(task_idx)
        self._alias = alias or f"MAE_task{task_idx}"

        # states Lightning will reset/aggregate across epochs/devices
        self.add_state("sum_abs", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("weight_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")

    @property
    def alias(self) -> str:
        # used in: self.log(f"{label}/{m.alias}", m, ...)
        return self._alias

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        lt_mask: torch.Tensor | None = None,
        gt_mask: torch.Tensor | None = None,
    ):
        """
        preds:   [batch, tasks] or [batch, tasks, u]
        targets: [batch, tasks]
        mask:    [batch, tasks] boolean
        weights: [batch] or [batch, 1]
        (lt_mask / gt_mask unused for plain MAE)
        """
        # If Chemprop is using an extra "n_targets" dim, flatten it like MPNN does before metrics
        if preds.dim() == 3:
            preds = preds[..., 0]

        # select this task's column
        preds_t = preds[:, self.task_idx]
        targets_t = targets[:, self.task_idx]

        # task-specific mask
        if mask is None:
            mask_t = torch.ones_like(targets_t, dtype=torch.bool)
        else:
            mask_t = mask[:, self.task_idx]

        # per-sample weights
        if weights is None:
            w = torch.ones_like(targets_t, dtype=torch.float)
        else:
            w = weights.view(-1).to(targets_t.dtype)

        # apply mask
        preds_t = preds_t[mask_t]
        targets_t = targets_t[mask_t]
        w = w[mask_t]

        if preds_t.numel() == 0:
            return  # nothing to update for this batch/task

        self.sum_abs += torch.sum(w*(preds_t - targets_t).abs())
        self.weight_sum += torch.sum(w)

    def compute(self) -> torch.Tensor:
        return self.sum_abs / self.weight_sum.clamp_min(1e-12)
