from typing import Optional

import torch
from torch import Tensor

from recsys_metrics.utils import _check_beyond_accuracy_inputs


def catalog_coverage(preds: Tensor, indexes: Tensor, k: Optional[int] = None) -> Tensor:
    preds, indexes, k = _check_beyond_accuracy_inputs(preds, indexes, k=k)

    _, topk_idx = preds.topk(k, dim=-1)
    sorted = False  # turn off sorting speeds up ~40% and we do NOT need sorted result for `numel`
    nunique_topk = indexes.take_along_dim(topk_idx, dim=-1).unique(sorted=sorted).numel()
    nunique_all = indexes.unique(sorted=sorted).numel()
    return torch.tensor(nunique_topk/nunique_all, dtype=torch.float32, device=preds.device)
