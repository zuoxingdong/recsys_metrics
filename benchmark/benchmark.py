import itertools
import timeit

import pandas as pd
import seaborn as sns
sns.set()


plot_kws = dict(
    kind='line',
    marker='.',
    markersize=15,
    markers=True,
    dashes=True
)


def benchmark(ns, ks, init_callback, base_callback, our_callback, number=10_000):
    result = []
    for n, k in itertools.product(ns, ks):
        init_out = init_callback(n, k)
        result.append({
            'n': n,
            'topk': k,
            'time_elapsed': timeit.timeit(lambda: base_callback(n, k, init_out), number=number),
            'version': 'torchmetrics'
        })
        result.append({
            'n': n,
            'topk': k,
            'time_elapsed': timeit.timeit(lambda: our_callback(n, k, init_out), number=number),
            'version': 'our'
        })
    df = pd.DataFrame(result)
    g = sns.relplot(
        data=df,
        x='n',
        y='time_elapsed',
        hue='version',
        col='topk',
        **plot_kws
    )
    return df, g


def benchmark_batch(n, k, batch_sizes, init_callback, base_callback, our_callback, number=10_000):
    result = []
    for batch_size in batch_sizes:
        init_out = init_callback(n, k, batch_size)
        result.append({
            'batch_size': batch_size,
            'time_elapsed': timeit.timeit(lambda: base_callback(n, k, init_out), number=number),
            'version': 'torchmetrics'
        })
        result.append({
            'batch_size': batch_size,
            'time_elapsed': timeit.timeit(lambda: our_callback(n, k, init_out), number=number),
            'version': 'our'
        })
    df = pd.DataFrame(result)
    g = sns.relplot(
        data=df,
        x='batch_size',
        y='time_elapsed',
        hue='version',
        kind='line',
        markers=True
    )
    return df, g


def sanity_check(n, k, init_callback, base_callback, our_callback):
    init_out = init_callback(n, k)
    y_base = base_callback(n, k, init_out)
    y_our = our_callback(n, k, init_out)
    assert y_our.allclose(y_base) 
