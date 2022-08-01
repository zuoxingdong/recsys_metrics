<div align="center">

<p align='center'><b>recsys_metrics</b></p>

**An efficient PyTorch implementation of the evaluation metrics in recommender systems.**

______________________________________________________________________

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#Installation">Installation</a> •
  <a href="#How-to-use">How to use</a> •
  <a href="#Benchmark">Benchmark</a> •
  <a href="#Citation">Citation</a> •
</p>

______________________________________________________________________

</div>

## Overview

### Why do we need `recsys_metrics`?
### Highlights
- Efficient (vectorized) implementations over mini-batches
- Standard RecSys metrics: precision, recall, map, mrr, hr, ndcg
- Beyond-accuracy metrics: coverage, diversity, novelty, serendipity, etc.
- All metrics support a top-k argument.

## Installation

You can install `recsys_metrics` from PyPI:

```bash
pip install recsys_metrics
```

Or you can also install the latest version from source:

```bash
pip install git+https://github.com/zuoxingdong/recsys_metrics.git@master
```

Note that we support Python 3.7+ only.

## How to use

Let us take Hit Rate (HR) to illustrate how to use this library:

```python
preds = torch.tensor([
    [.5, .3, .1],
    [.3, .4, .5]
])
target = torch.tensor([
    [0, 0, 1],
    [0, 1, 1]
])
hit_rate(preds, target, k=1, reduction='mean')

>> tensor(0.5000)
```
The first example in the batch does not have a hit (i.e. top-1 item is not a relevant item) and second example in the batch gets a hit (i.e. top-1 item is a relevant item). Thus, we have a hit-rate of 0.5.

The API of other metrics are of the same format.


## Benchmark

| Metrics | Single Example | Mini-Batch |
| :---:  | :---: | :---: |
| Precision | ![](assets/bench_precision_single.png) | ![](assets/bench_precision_batch.png) |
| Recall | ![](assets/bench_recall_single.png) | ![](assets/bench_recall_batch.png) |
| MAP | ![](assets/bench_map_single.png) | ![](assets/bench_map_batch.png) |
| MRR | ![](assets/bench_mrr_single.png) | ![](assets/bench_mrr_batch.png) |
| HR | ![](assets/bench_hr_single.png) | ![](assets/bench_hr_batch.png) |
| NDCG | ![](assets/bench_ndcg_single.png) | ![](assets/bench_ndcg_batch.png) |

## Citation

This work is inspired by [Torchmetrics](https://github.com/PyTorchLightning/metrics) from PyTorchLightning Team.

Please use this bibtex if you want to cite this repository in your publications:

    @misc{recsys_metrics,
          author = {Zuo, Xingdong},
          title = {recsys_metrics: An efficient PyTorch implementation of the evaluation metrics in recommender systems.},
          year = {2021},
          publisher = {GitHub},
          journal = {GitHub repository},
          howpublished = {\url{https://github.com/zuoxingdong/recsys_metrics}},
        }
