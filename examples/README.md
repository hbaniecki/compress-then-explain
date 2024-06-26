## setup environment

1. `conda env create -f env.yml -n cte`
2. `conda activate cte`
3. `pip install git+https://github.com/AI4LIFE-GROUP/OpenXAI --no-dependencies`

## run examples

- [cte_shap.ipynb](./cte_shap.ipynb): Example of CTE with the [`shap` Python package](https://github.com/shap/shap) explaining a neural network trained on the `heloc` dataset.
- [cte_sage.ipynb](./cte_sage.ipynb): Example of CTE with the [`sage` Python package](https://github.com/iancovert/sage) explaining an XGBoost model trained on the `compas` dataset.
- [cte_expected_gradients.ipynb](./cte_expected_gradients.ipynb): Example of CTE with the [`captum` Python package](https://github.com/pytorch/captum) explaining a CNN model trained on the `CIFAR_10` dataset.
- [cte_feature_effects.ipynb](./cte_feature_effects.ipynb): Example of CTE with the [`alibi` Python package](https://github.com/SeldonIO/alibi) explaining an XGBoost model trained on the `grid_stability` dataset.