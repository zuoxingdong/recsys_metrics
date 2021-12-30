from typing import Optional

from torch import Tensor

from recsys_metrics.utils import _check_ranking_inputs, _reduce_tensor, div_no_nan


def recall(preds: Tensor, target: Tensor, k: Optional[int] = None, reduction: Optional[str] = 'mean') -> Tensor:
    preds, target, k = _check_ranking_inputs(preds, target, k=k, batched=True)

    _, topk_idx = preds.topk(k, dim=-1)
    relevance = target.take_along_dim(topk_idx, dim=-1).sum(dim=-1).float()
    return _reduce_tensor(div_no_nan(relevance, target.sum(dim=-1)), reduction=reduction)
