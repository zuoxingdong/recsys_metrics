from typing import Optional

import torch
from torch import Tensor
from torch.distributions import Categorical

from recsys_metrics.utils import _check_beyond_accuracy_inputs


def _logn(n, x):
    return torch.log2(x) / torch.log2(n)


def distributional_coverage(preds: Tensor, indexes: Tensor, k: Optional[int] = None, normalize: Optional[bool] = True) -> Tensor:
    preds, indexes, k = _check_beyond_accuracy_inputs(preds, indexes, k=k)

    _, topk_idx = preds.topk(k, dim=-1)
    sorted = False  # turn off sorting speeds up ~40% and we do NOT need sorted result for `numel`
    _, cnt = indexes.take_along_dim(topk_idx, dim=-1).unique(sorted=sorted, return_counts=True)
    n_items = indexes.unique(sorted=sorted).numel()

    entropy = Categorical(probs=cnt/n_items).entropy()
    if normalize:
        beta = _logn(
            torch.tensor(n_items, dtype=torch.float32, device=preds.device), 
            torch.tensor(torch.e, dtype=torch.float32, device=preds.device)
        )
        entropy *= beta
    return entropy
