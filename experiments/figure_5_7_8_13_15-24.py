'''
python figure_6-8_13_15-24.py --dataset 3 --method sample --start 0 --stop 33 --output /metadata
'''

import argparse
parser = argparse.ArgumentParser(description='main')
parser.add_argument('--dataset', default=3, type=int, help='task id')
parser.add_argument('--method', default="compress", type=str, help='method name')
parser.add_argument('--start', default=0, type=int, help='starting repeat')
parser.add_argument('--stop', default=3, type=int, help='stop repeat')
parser.add_argument('--output', default="/metadata", type=str, help='output path')
args = parser.parse_args()

import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import cte
import time
import numpy as np
import pandas as pd
from goodpoints import compress
from sklearn_extra.cluster import KMedoids

EXPLANATIONS = [
    "expected_gradients",
    "feature_effects",
    "shap_permutation", 
    "sage_permutation_foreground", 
    "sage_permutation_background", 
    "shap_kernel", 
    "sage_kernel_foreground", 
    "sage_kernel_background", 
]

VERBOSE = True

METADATA = pd.DataFrame()

x, y, model = cte.utils.get_data_and_model_openml(args.dataset)
n_samples = x.shape[0]
n_features = x.shape[1]
n_classes = len(np.unique(y))
sigma = np.sqrt(2 * n_features)
for repeat in range(args.start, args.stop):
    if VERBOSE:
        print(f'===  REPEAT: {repeat}', flush=True)
    n_compressed = cte.utils.largest_power_of_2_in_sqrt(n_samples)
    if args.method == "compress":
        t_compress_start = time.time()
        id_compressed = compress.compresspp_kt(x, kernel_type=b"gaussian", k_params=np.array([sigma**2]), g=4, seed=repeat)
        t_compress_end = time.time()
        assert n_compressed == len(id_compressed)
    elif args.method == "cluster":
        t_compress_start = time.time()
        id_compressed = KMedoids(init='k-medoids++', n_clusters=n_compressed, random_state=repeat).fit(x).medoid_indices_
        t_compress_end = time.time()
    elif args.method.startswith("sample"):
        np.random.seed(repeat)
        t_compress_start = time.time()
        id_compressed = np.random.choice(n_samples, size=n_compressed * int(args.method[6:]))
        t_compress_end = time.time()
    for explanation in EXPLANATIONS:
        if  n_features < 32 and explanation.startswith("expected_gradients") or\
            n_features >= 32 and not explanation.startswith("expected_gradients"):
            # skip
            continue
        if VERBOSE:
            print(f'- data: {args.dataset} |  method: {args.method} | explanation: {explanation}', flush=True)
        t_explain_start = time.time()
        e = cte.explanations.compute_explanation(
            explanation=explanation, 
            model=model, 
            x=x, 
            y=y,
            x_compressed=x[id_compressed], 
            y_compressed=y[id_compressed],
            random_state=repeat
        )
        t_explain_end = time.time()
        METADATA = pd.concat([METADATA, pd.DataFrame({
            'dataset': [args.dataset],
            'method': [args.method],
            'explanation': [explanation],
            'repeat': [repeat],
            'time_compress': [t_compress_end - t_compress_start],
            'time_explain': [t_explain_end - t_explain_start]
        })])
        np.save(f'{args.output}/explanation_{args.dataset}_{args.method}_{explanation}_{repeat}.npy', e) 
    METADATA.to_csv(f'{args.output}/metadata_{args.dataset}_{args.method}_{args.start}_{args.stop-1}.csv', index=False)