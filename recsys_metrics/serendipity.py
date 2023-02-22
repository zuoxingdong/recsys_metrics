from typing import Optional

import torch
from torch import Tensor

from recsys_metrics.utils import _check_ranking_inputs, _check_preds_unexpectedness_inputs, _reduce_tensor


# Ref: https://eugeneyan.com/writing/serendipity-and-accuracy-in-recommender-systems/#serendipity
def serendipity(preds: Tensor, target: Tensor, unexpectedness: Tensor, k: Optional[int] = None, reduction: Optional[str] = 'mean') -> Tensor:
    preds, target, k = _check_ranking_inputs(preds, target, k=k, batched=True)
    preds, unexpectedness, k = _check_preds_unexpectedness_inputs(preds, unexpectedness, k=k)

    _, topk_idx = preds.topk(k, dim=-1)
    relevance = target.take_along_dim(topk_idx, dim=-1)
    unexpected = unexpectedness.take_along_dim(topk_idx, dim=-1)
    return _reduce_tensor((relevance * unexpected).float().mean(dim=-1), reduction=reduction)


def category_unexpectedness(item_category: Tensor, hist_seq_category: Tensor, window_size: Optional[int] = None) -> Tensor:
    if item_category.device != hist_seq_category.device:
        raise ValueError('`item_category` and `hist_seq_category` are must on the same device')
    
    if hist_seq_category.ndim != item_category.ndim + 1:
        raise ValueError('`hist_seq_category` must have one more dimension than `item_category`')

    if item_category.shape[:item_category.ndim] != hist_seq_category.shape[:item_category.ndim]:
        raise ValueError('`item_category` and `hist_seq_category` must have the same shape, except for the last dimension of `hist_seq_category`')

    if not item_category.numel() or not item_category.size():  # already checked tensor shape consistency, so it's sufficient to only check one tensor
        raise ValueError('`item_category` and `hist_seq_category` must be non-empty and non-scalar tensors')
    
    if item_category.dtype is not torch.long:
        raise ValueError('`item_category` must be a tensor of long integers')

    if hist_seq_category.dtype is not torch.long:
        raise ValueError('`hist_seq_category` must be a tensor of long integers')

    if window_size is None:
        window_size = hist_seq_category.shape[-1]
    if not (isinstance(window_size, int) and window_size > 0):
        raise ValueError(f'`window_size` has to be a positive integer or None')
    
    hist_seq_category = hist_seq_category[..., -window_size:]
    is_unexpected = (item_category[..., None] != hist_seq_category).all(dim=-1)
    return is_unexpected
