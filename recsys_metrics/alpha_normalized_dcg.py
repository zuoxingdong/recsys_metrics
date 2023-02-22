from typing import Optional

import torch
from torch import Tensor

from recsys_metrics.utils import _check_ranking_inputs, _reduce_tensor, div_no_nan


def _alpha_dcg(target: Tensor, alpha: int) -> Tensor:
    batch_size, k, n_topics = target.shape
    rank_positions = torch.arange(1, k + 1, dtype=torch.float32, device=target.device).tile((batch_size, 1))
    
    cumsum_topics = target.cumsum(dim=1).roll(1, 1)
    cumsum_topics[:, 0, :].zero_()
    
    gain = target * (1. - alpha)**cumsum_topics
    return (gain.sum(-1) / torch.log2(rank_positions + 1)).sum(dim=-1)


def alpha_normalized_dcg(preds: Tensor, target: Tensor, alpha: int, k: Optional[int] = None, reduction: Optional[str] = 'mean', ret_alpha_dcg: Optional[bool] = False) -> Tensor:
    preds, target, k = _check_ranking_inputs(
        preds, 
        target, 
        k=k, 
        batched=True, 
        skip_target_shape_check=True,  # allow target has shape [batch_size, list_size, topic_size]
    )

    _, topk_idx = preds.topk(k, dim=-1)
    sorted_target = target.take_along_dim(topk_idx[..., None], dim=-2)
    
    _, ideal_idx = target.sum(dim=-1).topk(k=k, dim=-1)
    ideal_target = target.take_along_dim(ideal_idx[..., None], dim=-2)

    if ret_alpha_dcg:
        return _reduce_tensor(_alpha_dcg(sorted_target, alpha=alpha), reduction=reduction)
    else:
        return _reduce_tensor(div_no_nan(_alpha_dcg(sorted_target, alpha=alpha), _alpha_dcg(ideal_target, alpha=alpha)), reduction=reduction)
