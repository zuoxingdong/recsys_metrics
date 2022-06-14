from typing import Optional

from torch import Tensor

from recsys_metrics.precision import precision
from recsys_metrics.recall import recall
from recsys_metrics.mean_average_precision import mean_average_precision
from recsys_metrics.mean_reciprocal_rank import mean_reciprocal_rank
from recsys_metrics.hit_rate import hit_rate
from recsys_metrics.normalized_dcg import normalized_dcg


dict_abbrev_names = {
    'prec': 'prec',
    'rec': 'rec',
    'map': 'map',
    'mrr': 'mrr',
    'hr': 'hr',
    'ndcg': 'ndcg'
}


dict_full_names = {
    'prec': 'precision',
    'rec': 'recall',
    'map': 'mean_average_precision',
    'mrr': 'mean_reciprocal_rank',
    'hr': 'hit_rate',
    'ndcg': 'normalized_dcg'
}


def rank_report(preds: Tensor, target: Tensor, k: Optional[int] = None, reduction: Optional[str] = 'mean', to_item: Optional[bool] = True, name_abbreviation: Optional[bool] = False) -> dict:
    dict2name = dict_abbrev_names if name_abbreviation else dict_full_names
    report = {
        dict2name['prec']: precision(preds=preds, target=target, k=k, reduction=reduction),
        dict2name['rec']: recall(preds=preds, target=target, k=k, reduction=reduction),
        dict2name['map']: mean_average_precision(preds=preds, target=target, k=k, reduction=reduction),
        dict2name['mrr']: mean_reciprocal_rank(preds=preds, target=target, k=k, reduction=reduction),
        dict2name['hr']: hit_rate(preds=preds, target=target, k=k, reduction=reduction),
        dict2name['ndcg']: normalized_dcg(preds=preds, target=target, k=k, reduction=reduction),
    }
    if to_item:
        report = {k: v.item() for k, v in report.items()}
    return report
