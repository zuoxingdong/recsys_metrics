from typing import Optional

import torch
from torch import Tensor

from recsys_metrics.utils import _check_preds_popularities_inputs


# EPC: measures the ability to recommend long-tail items in top positions
# Eq. (7) in https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1089.1342&rep=rep1&type=pdf
def expected_popularity_complement(preds: Tensor, popularities: Tensor, k: Optional[int] = None) -> Tensor:
    preds, popularities, k = _check_preds_popularities_inputs(preds, popularities, k=k)
    
    _, topk_idx = preds.topk(k, dim=-1)
    topk_pops = popularities.take_along_dim(topk_idx, dim=-1)

    batch_size, _ = preds.shape
    rank_positions = torch.arange(1, k + 1, dtype=torch.float32, device=preds.device).tile((batch_size, 1))
    log_discount = torch.log2(rank_positions + 1)

    numerator = ((1. - topk_pops)/log_discount).sum()  # `sum` -> scalar
    denominator = (1. / log_discount).sum()  # `sum` -> scalar
    return numerator / denominator
