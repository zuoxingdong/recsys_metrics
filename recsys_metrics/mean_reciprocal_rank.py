from typing import Optional

from torch import Tensor

from recsys_metrics.utils import _check_ranking_inputs, _reduce_tensor


def mean_reciprocal_rank(preds: Tensor, target: Tensor, k: Optional[int] = None, reduction: Optional[str] = 'mean') -> Tensor:
    preds, target, k = _check_ranking_inputs(preds, target, k=k, batched=True)

    _, topk_idx = preds.topk(k, dim=-1)
    relevance = target.take_along_dim(topk_idx, dim=-1)
    first_relevant_positions = relevance.argmax(dim=-1) + 1
    valid_mask = (relevance.sum(dim=-1) > 0)
    # we do NOT need `div_no_nan` since the denominator is always non-zero! saved a bit more wall-clock time
    return _reduce_tensor(valid_mask/first_relevant_positions, reduction=reduction)
