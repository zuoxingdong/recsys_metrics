from typing import Optional

import torch
from torch import Tensor

from recsys_metrics.utils import _check_ranking_inputs, _reduce_tensor, div_no_nan


def mean_average_precision(preds: Tensor, target: Tensor, k: Optional[int] = None, reduction: Optional[str] = 'mean') -> Tensor:
    preds, target, k = _check_ranking_inputs(preds, target, k=k, batched=True)

    _, topk_idx = preds.topk(k, dim=-1)
    relevance = target.take_along_dim(topk_idx, dim=-1)
    cum_relevance = relevance.cumsum(dim=-1)
    rank_positions = torch.arange(1, k + 1, dtype=torch.float32, device=target.device)
    ap = (relevance*cum_relevance) / rank_positions
    return _reduce_tensor(div_no_nan(ap.sum(dim=-1), relevance.sum(dim=-1)), reduction=reduction)
