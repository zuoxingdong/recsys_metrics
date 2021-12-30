import torch

from recsys_metrics.recall import recall
from torchmetrics.functional import retrieval_recall

from benchmark.benchmark import benchmark, benchmark_batch, sanity_check


# ns = [128, 256, 512, 1024, 2048, 4096]
# ks = [1, 50, 100]
def bench_recall_single(ns, ks, number=10_000, seed=42):
    def init_callback(n, k):
        torch.manual_seed(seed)
        preds = torch.rand(n)
        target = torch.randint(0, 2, size=(n,))
        return preds, target

    def base_callback(n, k, init_out):
        preds, target = init_out
        return retrieval_recall(preds, target, k=k)

    def our_callback(n, k, init_out):
        preds, target = init_out
        return recall(preds, target, k=k, reduction='mean')

    sanity_check(ns[-1], ks[-1], init_callback, base_callback, our_callback)

    df, g = benchmark(ns, ks, init_callback, base_callback, our_callback, number=number)
    g.fig.subplots_adjust(top=.88)
    g.fig.suptitle('Benchmark - Single Example: Recall', fontsize=16)
    return df, g

# n=512,
# k=50,
# batch_sizes=[32, 64, 128, 256, 512],
def bench_recall_batch(n, k, batch_sizes, number=1_000, seed=42):
    def init_callback(n, k, batch_size):
        torch.manual_seed(seed)
        preds = torch.rand(batch_size, n)
        target = torch.randint(0, 2, size=(batch_size, n))
        return preds, target

    def base_callback(n, k, init_out):
        preds, target = init_out
        out = torch.stack([
            retrieval_recall(_preds, _target, k=k)
            for _preds, _target in zip(preds, target)
        ]).mean(0)
        return out

    def our_callback(n, k, init_out):
        preds, target = init_out
        return recall(preds, target, k=k, reduction='mean')

    df, g = benchmark_batch(
        n=n,
        k=k,
        batch_sizes=batch_sizes,
        init_callback=init_callback,
        base_callback=base_callback,
        our_callback=our_callback,
        number=number
    )
    g.fig.subplots_adjust(top=.88)
    g.fig.suptitle('Benchmark - Mini-Batch: Recall', fontsize=16)
    return df, g
