from typing import Optional

from torch import Tensor

from recsys_metrics.precision import precision
from recsys_metrics.recall import recall
from recsys_metrics.mean_average_precision import mean_average_precision
from recsys_metrics.mean_reciprocal_rank import mean_reciprocal_rank
from recsys_metrics.hit_rate import hit_rate
from recsys_metrics.normalized_dcg import normalized_dcg


def rank_report(preds: Tensor, target: Tensor, k: Optional[int] = None, reduction: Optional[str] = 'mean', to_item: Optional[bool] = True) -> dict:
    report = {
        'precision': precision(preds=preds, target=target, k=k, reduction=reduction),
        'recall': recall(preds=preds, target=target, k=k, reduction=reduction),
        'mean_average_precision': mean_average_precision(preds=preds, target=target, k=k, reduction=reduction),
        'mean_reciprocal_rank': mean_reciprocal_rank(preds=preds, target=target, k=k, reduction=reduction),
        'hit_rate': hit_rate(preds=preds, target=target, k=k, reduction=reduction),
        'normalized_dcg': normalized_dcg(preds=preds, target=target, k=k, reduction=reduction),
    }
    if to_item:
        report = {k: v.item() for k, v in report.items()}
    return report
