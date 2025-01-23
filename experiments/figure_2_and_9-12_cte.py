import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from goodpoints import compress
from openxai.model import LoadModel
from openxai.dataloader import ReturnLoaders
model_name = 'ann'

import sage
import shap

def exp_sage(
        model, X, y, X_background=None, 
        loss="cross entropy", 
        estimator="permutation", 
        random_state=None,
        bar=False, verbose=False
    ):
    if X_background is None:
        X_background = X
    imputer = sage.MarginalImputer(model.predict_proba, X_background)
    if estimator == "kernel":
        explainer = sage.KernelEstimator(imputer, loss=loss, random_state=random_state)
    elif estimator == "permutation":
        explainer = sage.PermutationEstimator(imputer, loss=loss, random_state=random_state)
    sage_values = explainer(X, y, bar=bar, verbose=verbose)
    return sage_values.values

def exp_shap(
        model, X, X_background=None, 
        estimator="permutation", 
        random_state=None,
        bar=False
    ):
    if X_background is None:
        X_background = X
    if estimator == "kernel":
        explainer = shap.KernelExplainer(lambda x: model.predict_proba(x)[:, 1], X_background, seed=random_state)
    elif estimator == "permutation":
        masker = shap.maskers.Independent(X_background, max_samples=X_background.shape[0])
        explainer = shap.PermutationExplainer(lambda x: model.predict_proba(x)[:, 1], masker, seed=random_state)
    shap_values = explainer(X, silent=not bar)
    return shap_values.values

datasets = ['gaussian', 'compas', 'heloc', 'adult', 'gmsc']
explanations = ["shap", "sage"]
estimators = ["kernel", "permutation"]
methods = ["compress", "sample"]

verbose = True
start = 0
stop = 33
for data_name in datasets:
    results = {f'{method}_{explanation}_{estimator}': [] for method in methods 
            for explanation in explanations for estimator in estimators}
    times = pd.DataFrame({'dataset': [], 'repeat': [], 'method': [], 'time': []})
    if verbose:
        print(f'==== dataset: {data_name} ====')
    _, loader_test = ReturnLoaders(data_name=data_name, download=False, batch_size=128)
    X_test = loader_test.dataset.data
    y_test = loader_test.dataset.targets.to_numpy()
    model = LoadModel(data_name=data_name, ml_model=model_name, pretrained=True)
    model.eval()
    n = X_test.shape[0]
    d = X_test.shape[1]
    sigma = np.sqrt(2 * d)
    for repeat in tqdm(range(start, stop)):
        if repeat == start + 1 and verbose:
            print(f'original: {n} || compressed: {len(id_compressed)}/{n} || ratio: {len(id_compressed)/n:.2f}')
        if verbose:
            print(f'= repeat: {repeat}')
        for method in methods:
            if method == "compress":
                id_compressed = compress.compresspp_kt(X_test, kernel_type=b"gaussian", k_params=np.array([sigma**2]), g=4, seed=repeat)
                X_compressed = X_test[id_compressed]
                y_compressed = y_test[id_compressed]
            elif method == "sample":
                np.random.seed(repeat)
                id_random = np.random.choice(n, size=len(id_compressed))
                X_compressed = X_test[id_random]
                y_compressed = y_test[id_random]
            for explanation in explanations:
                for estimator in estimators:
                    if explanation == "shap":
                        start_time = time.time()
                        exp = exp_shap(model, X_test, X_compressed, estimator=estimator, bar=False, random_state=repeat)
                        end_time = time.time() - start_time
                    elif explanation == "sage":
                        start_time = time.time()
                        exp = exp_sage(model, X_test, y_test, X_compressed, estimator=estimator, bar=False, random_state=repeat)
                        end_time = time.time() - start_time
                    times = pd.concat([times, pd.DataFrame({
                            'dataset': [data_name],
                            'repeat': [repeat],
                            'method': [f'{method}_{explanation}_{estimator}'], 
                            'time': [end_time]
                        })])
                    results[f'{method}_{explanation}_{estimator}'] += [exp]

    np.save(f'metadata/{data_name}/cte_shap_sage_results_{start}_{stop}.npy', results) 
    times.to_csv(f'metadata/{data_name}/cte_shap_sage_times_{start}_{stop}.csv', index=False) 