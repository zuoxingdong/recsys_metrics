from typing import Optional

from torch import Tensor

from recsys_metrics.catalog_coverage import catalog_coverage
from recsys_metrics.distributional_coverage import distributional_coverage
from recsys_metrics.mean_interlist_diversity import mean_interlist_diversity
from recsys_metrics.expected_popularity_complement import expected_popularity_complement


dict_abbrev_names = {
    'cat_cov': 'cat_cov',
    'dist_cov': 'dist_cov',
    'mil': 'mil',
    'epc': 'epc'
}


dict_full_names = {
    'cat_cov': 'catalog_coverage',
    'dist_cov': 'distributional_coverage',
    'mil': 'mean_interlist_diversity',
    'epc': 'expected_popularity_complement'
}


def beyond_accuracy_report(preds: Tensor, indexes: Tensor, popularities: Optional[Tensor] = None, k: Optional[int] = None, to_item: Optional[bool] = True, name_abbreviation: Optional[bool] = False) -> dict:
    dict2name = dict_abbrev_names if name_abbreviation else dict_full_names
    report = {
        dict2name['cat_cov']: catalog_coverage(preds=preds, indexes=indexes, k=k),
        dict2name['dist_cov']: distributional_coverage(preds=preds, indexes=indexes, k=k, normalize=True),
        dict2name['mil']: mean_interlist_diversity(preds=preds, indexes=indexes, k=k),
    }
    if popularities is not None:
        report[dict2name['epc']] = expected_popularity_complement(preds=preds, popularities=popularities, k=k)
    if to_item:
        report = {k: v.item() for k, v in report.items()}
    return report
