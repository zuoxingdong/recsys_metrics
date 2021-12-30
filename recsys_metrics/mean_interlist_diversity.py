from typing import Optional

from torch import Tensor

from recsys_metrics.utils import _check_beyond_accuracy_inputs


# Efficient implementation: see https://ojs.aaai.org/index.php/AAAI/article/view/17886/17691
def mean_interlist_diversity(preds: Tensor, indexes: Tensor, k: Optional[int] = None) -> Tensor:
    preds, indexes, k = _check_beyond_accuracy_inputs(preds, indexes, k=k)

    _, topk_idx = preds.topk(k, dim=-1)
    sorted = False  # turn off sorting speeds up ~40% and we do NOT need sorted result for `numel`
    _, topk_cnt = indexes.take_along_dim(topk_idx, dim=-1).unique(sorted=sorted, return_counts=True)
    
    squared_cnt = (topk_cnt**2).sum(dim=-1)  # `tensor` dtype, no need to convert other raw dtypes below
    n_users = len(preds)
    n_pairs = n_users**2 - n_users
    return (n_users**2 - squared_cnt/k) / n_pairs
