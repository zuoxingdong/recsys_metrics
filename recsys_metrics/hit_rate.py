from typing import Optional

from torch import Tensor

from recsys_metrics.utils import _check_ranking_inputs, _reduce_tensor


def hit_rate(preds: Tensor, target: Tensor, k: Optional[int] = None, reduction: Optional[str] = 'mean') -> Tensor:
    preds, target, k = _check_ranking_inputs(preds, target, k=k, batched=True)

    _, topk_idx = preds.topk(k, dim=-1)
    relevance = target.take_along_dim(topk_idx, dim=-1)
    return _reduce_tensor((relevance.sum(dim=-1) > 0).float(), reduction=reduction)
