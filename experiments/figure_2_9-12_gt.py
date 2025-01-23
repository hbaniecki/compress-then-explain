import time
import numpy as np
import pandas as pd

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
        if X_background.shape[0] > 128*20:
            np.random.seed(random_state)
            id_gt = np.random.choice(X_background.shape[0], replace=False, size=128*20)
            X_background = X_background[id_gt]
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
        if X_background.shape[0] > 128*20:
            np.random.seed(random_state)
            id_gt = np.random.choice(X_background.shape[0], replace=False, size=128*20)
            X_background = X_background[id_gt]
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

verbose = True
start = 0
stop = 3
for data_name in datasets:
    results = {f'{explanation}_{estimator}': [] for explanation in explanations for estimator in estimators}
    times = pd.DataFrame({'dataset': [], 'repeat': [], 'method': [], 'time': []})
    if verbose:
        print(f'==== dataset: {data_name} ====')
    _, loader_test = ReturnLoaders(data_name=data_name, download=False, batch_size=128)
    X_test = loader_test.dataset.data
    y_test = loader_test.dataset.targets.to_numpy()
    model = LoadModel(data_name=data_name, ml_model=model_name, pretrained=True)
    model.eval()
    for repeat in range(start, stop):
        if verbose:
            print(f'= repeat: {repeat}')
        for explanation in explanations:
            for estimator in estimators:
                if explanation == "shap":
                    start_time = time.time()
                    exp = exp_shap(model, X_test, estimator=estimator, bar=True, random_state=repeat)
                    end_time = time.time() - start_time
                elif explanation == "sage":
                    start_time = time.time()
                    exp = exp_sage(model, X_test, y_test, estimator=estimator, bar=True, random_state=repeat)
                    end_time = time.time() - start_time
                times = pd.concat([times, pd.DataFrame({
                        'dataset': [data_name],
                        'repeat': [repeat],
                        'method': [f'{explanation}_{estimator}'], 
                        'time': [end_time]
                    })])
                results[f'{explanation}_{estimator}'] += [exp]

    np.save(f'metadata/{data_name}/gt_shap_sage_results_{start}_{stop}.npy', results) 
    times.to_csv(f'metadata/{data_name}/gt_shap_sage_times_{start}_{stop}.csv', index=False) 