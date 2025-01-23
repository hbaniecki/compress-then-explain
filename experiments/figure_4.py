import time
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn_extra.cluster import KMedoids
from goodpoints import compress

def d_mmd(X, Y, gamma):
    """ 
    source: https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py
    MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2))
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

DATASETS = ['imdb', 'imagenet1k']

result = pd.DataFrame({'dataset': [], 'repeat': [], 'method': [], 'time': [], 'd_mmd': []})

repeats = 33
for data_name in DATASETS:
    print(f'==== dataset: {data_name} ====')
    X_test = pd.read_csv(f'data/{data_name}/{data_name}-valid.csv').drop("target", axis=1).values
    if data_name == "imagenet1k":
        X_test = np.log10(1+X_test)
        # reduce imagenet1k for KMedoids because of RAM issue
        id_reduce = np.random.choice(n, size=25000, replace=False)
        X_test_reduced = X_test[id_reduce]
    else:
        X_test_reduced = X_test
    n = X_test.shape[0]
    d = X_test.shape[1]
    sigma = np.sqrt(2 * d)
    gamma = 1 / (sigma**2)
    for repeat in range(repeats):
        start = time.time()
        id_compressed = compress.compresspp_kt(X_test, kernel_type=b"gaussian", k_params=np.array([sigma**2]), g=4, seed=repeat)
        time_compressed = time.time() - start
        X_compressed = X_test[id_compressed]
        n_compressed = len(id_compressed)

        print(f'= repeat: {repeat} | n_sample: {n_compressed}')
        np.random.seed(repeat)
        id_random = np.random.choice(n, size=n_compressed)
        X_random = X_test[id_random] 

        start = time.time()
        id_cluster = KMedoids(init='k-medoids++', n_clusters=n_compressed, random_state=repeat).fit(X_test_reduced).medoid_indices_
        time_clustered = time.time() - start
        X_clustered = X_test_reduced[id_cluster]

        result = pd.concat([result, pd.DataFrame({
                'dataset': [data_name]*3,
                'repeat': [repeat]*3,
                'method': ["compress", "sample", "cluster"],
                'time': [time_compressed, 0.001, time_clustered],
                'd_mmd': [d_mmd(X_test, X_compressed, gamma), d_mmd(X_test, X_random, gamma), d_mmd(X_test, X_clustered, gamma)],
            })])
    result.to_csv(f'metadata/cte_benchmark_compress_{data_name}.csv', index=False) 
    result.to_csv('metadata/cte_benchmark_compress_text_image.csv', index=False) 