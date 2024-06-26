import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn import metrics
from geomloss import SamplesLoss
from scipy.stats import entropy 
from sklearn_extra.cluster import KMedoids
from goodpoints import compress
from openxai.dataloader import ReturnLoaders

def d_mmd(X, Y, gamma):
    """ 
    source: https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py
    MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2))
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

def d_wd(X, Y):
    X, Y = torch.from_numpy(X), torch.from_numpy(Y)
    loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9, debias=False, backend='tensorized')
    return loss(X, Y).item()

def get_topk_values(x, k):
    x = np.array(x)
    ind = np.argpartition(x, -k)[-k:]
    return x[ind]

def d_tv(X, Y):
    tv_1d = []
    for j in range(X.shape[1]):
        _, edges = np.histogram(X[:,j], density=True, bins=int(np.sqrt(Y.shape[0])))
        X_density = np.histogram(X[:,j], density=False, bins=edges)[0] / X.shape[0]
        Y_density = np.histogram(Y[:,j], density=False, bins=edges)[0] / Y.shape[0]
        tv_1d += [np.sum(np.abs(X_density - Y_density)) / 2]
    return np.mean(get_topk_values(tv_1d, 3))

def d_kl(X, Y):
    kl_1d = []
    for j in range(X.shape[1]):
        _, edges = np.histogram(X[:,j], density=True, bins=int(np.sqrt(Y.shape[0])))
        X_density = np.histogram(X[:,j], density=False, bins=edges)[0] / X.shape[0]
        Y_density = np.histogram(Y[:,j], density=False, bins=edges)[0] / Y.shape[0]
        kl_1d += [entropy(Y_density, X_density)]
    return np.mean(get_topk_values(kl_1d, 3))

DATASETS = ['gaussian', 'compas', 'heloc', 'adult', 'gmsc']

result = pd.DataFrame({'dataset': [], 'repeat': [], 'method': [], 'time': [], 'd_mmd': [], 'd_wd': [], 'd_tv': [], 'd_kl': []})

repeats = 33
for data_name in DATASETS:
    print(f'==== dataset: {data_name} ====')
    loader_train, loader_test = ReturnLoaders(data_name=data_name, download=False, batch_size=128)
    _, X_test = loader_train.dataset.data, loader_test.dataset.data
    _, y_test = loader_train.dataset.targets.to_numpy(), loader_test.dataset.targets.to_numpy()
    n = X_test.shape[0]
    d = X_test.shape[1]
    sigma = np.sqrt(2 * d)
    gamma = 1 / (sigma**2)
    for repeat in tqdm(range(repeats)):
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
        kmedoids = KMedoids(init='k-medoids++', n_clusters=n_compressed, random_state=repeat).fit(X_test)
        time_clustered = time.time() - start
        X_clustered = kmedoids.cluster_centers_

        result = pd.concat([result, pd.DataFrame({
                'dataset': [data_name]*3,
                'repeat': [repeat]*3,
                'method': ["compress", "sample", "cluster"],
                'time': [time_compressed, 0.001, time_clustered],
                'd_mmd': [d_mmd(X_test, X_compressed, gamma), d_mmd(X_test, X_random, gamma), d_mmd(X_test, X_clustered, gamma)],
                'd_wd': [d_wd(X_test, X_compressed), d_wd(X_test, X_random), d_wd(X_test, X_clustered)],
                'd_tv': [d_tv(X_test, X_compressed), d_tv(X_test, X_random), d_tv(X_test, X_clustered)],
                'd_kl': [d_kl(X_test, X_compressed), d_kl(X_test, X_random), d_kl(X_test, X_clustered)]
            })])
    result.to_csv('metadata/cte_benchmark_compress.csv', index=False) 