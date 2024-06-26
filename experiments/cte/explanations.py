import sage
import shap
import torch
import captum
import numpy as np


def compute_explanation(
        explanation, 
        model,
        x, 
        y,
        x_compressed, 
        y_compressed,
        random_state,
        n_jobs=None
    ):
    if explanation == "shap_kernel":
        torch.set_grad_enabled(False)
        e = explanation_shap(
            model=model,
            x_foreground=x,
            x_background=x_compressed,
            estimator="kernel",
            random_state=random_state,
            n_jobs=n_jobs
        )
    elif explanation == "shap_permutation":
        torch.set_grad_enabled(False)
        e = explanation_shap(
            model=model,
            x_foreground=x,
            x_background=x_compressed,
            estimator="permutation",
            random_state=random_state,
            n_jobs=n_jobs
        )
    elif explanation == "sage_kernel_foreground":
        torch.set_grad_enabled(False)
        e = explanation_sage(
            model=model, 
            x_foreground=x_compressed, 
            y_foreground=y_compressed, 
            x_background=x_compressed, 
            estimator="kernel", 
            random_state=random_state,
            n_jobs=n_jobs
        )
    elif explanation == "sage_kernel_background":
        torch.set_grad_enabled(False)
        e = explanation_sage(
            model=model, 
            x_foreground=x, 
            y_foreground=y, 
            x_background=x_compressed, 
            estimator="kernel", 
            random_state=random_state,
            n_jobs=n_jobs
        )
    elif explanation == "sage_permutation_foreground":
        torch.set_grad_enabled(False)
        e = explanation_sage(
            model=model, 
            x_foreground=x_compressed, 
            y_foreground=y_compressed, 
            x_background=x_compressed, 
            estimator="permutation", 
            random_state=random_state,
            n_jobs=n_jobs
        )
    elif explanation == "sage_permutation_background":
        torch.set_grad_enabled(False)
        e = explanation_sage(
            model=model, 
            x_foreground=x, 
            y_foreground=y, 
            x_background=x_compressed, 
            estimator="permutation", 
            random_state=random_state,
            n_jobs=n_jobs
        )
    elif explanation == "feature_effects":
        torch.set_grad_enabled(False)
        e = explanation_feature_effects(
            model=model,
            x_foreground=x_compressed,
            x_background=x
        )
    elif explanation == "expected_gradients":
        e = explanation_expected_gradients(
            model=model,
            x_foreground=x,
            x_background=x_compressed
        )
    return e


def explanation_shap(
        model, 
        x_foreground, 
        x_background, 
        estimator="permutation", 
        random_state=None,
        silent=True,
        n_jobs=None
    ):
    if hasattr(model, "predict_proba"):
        f = lambda x: model.predict_proba(x)[:, 1]
    else:
        f = model.predict
    if estimator == "kernel":
        explainer = shap.KernelExplainer(f, x_background, seed=random_state)
    elif estimator == "permutation":
        masker = shap.maskers.Independent(x_background, max_samples=x_background.shape[0])
        explainer = shap.PermutationExplainer(f, masker, seed=random_state)
    if n_jobs is None:
        shap_values = explainer(x_foreground, silent=silent)
        return shap_values.values
    else:
        import joblib
        BATCH_SIZE = 512
        batches = [x_foreground[(i*BATCH_SIZE):(i+1)*BATCH_SIZE] for i in range(int(1+x_foreground.shape[0]/BATCH_SIZE))]
        shap_values = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(explainer)(x) for x in batches)
        return np.concatenate([sv.values for sv in shap_values], axis=0)


def explanation_sage(
        model, 
        x_foreground, 
        y_foreground, 
        x_background, 
        estimator="permutation", 
        random_state=None,
        bar=False, 
        verbose=False,
        n_jobs=None
    ):
    if hasattr(model, "predict_proba"):
        imputer = sage.MarginalImputer(lambda x: model.predict_proba(x)[:, 1], x_background)
        y_foreground = (y_foreground == 1).astype(int)
        loss = "cross entropy"
        if len(np.unique(y_foreground)) == 1: # hotfix sage in an edgecase
            y_foreground[0] = 1 - y_foreground[0]
    else:
        imputer = sage.MarginalImputer(model.predict, x_background)
        loss = "mse"
    if estimator == "kernel":
        explainer = sage.KernelEstimator(imputer, loss=loss, random_state=random_state)
    elif estimator == "permutation":
        if n_jobs is None:
            explainer = sage.PermutationEstimator(imputer, loss=loss, random_state=random_state)
        else:
            explainer = sage.PermutationEstimator(imputer, loss=loss, random_state=random_state, n_jobs=n_jobs)
    if n_jobs is None:
        sage_values = explainer(x_foreground, y_foreground, bar=bar, verbose=verbose)
    else:
        sage_values = explainer(x_foreground, y_foreground, bar=bar, verbose=verbose, batch_size=2048)
    return sage_values.values


def explanation_feature_effects(
        model,
        x_foreground,
        x_background,
        grid_points=100
    ):
    sq_grid_points = int(np.sqrt(grid_points))
    assert sq_grid_points**2 == grid_points
    if hasattr(model, "predict_proba"):
        f = lambda x: model.predict_proba(x)[:, 1]
    else:
        f = model.predict
    n_samples = x_foreground.shape[0]
    n_features = x_foreground.shape[1]
    effects = {}
    for s in range(0, n_features):
        x_s = x_background[:, s]
        xs = np.repeat(np.linspace(x_s.min(), x_s.max(), grid_points), n_samples)
        x_tiled = np.tile(x_foreground, (grid_points, 1))
        x_tiled[:, s] = xs
        pred = f(x_tiled)
        effects[f'{s}'] = pred.reshape(grid_points, -1).mean(axis=1)
    for s1 in range(0, n_features):
        for s2 in range(s1, n_features):
            if s1 == s2:
                continue
            x_s1 = x_background[:, s1]
            x_s2 = x_background[:, s2]
            grid = np.array([[i, j] 
                             for i in np.linspace(x_s1.min(), x_s1.max(), sq_grid_points) 
                             for j in np.linspace(x_s2.min(), x_s2.max(), sq_grid_points)])
            xs = np.repeat(grid, n_samples, axis=0)
            x_tiled = np.tile(x_foreground, (grid_points, 1))
            x_tiled[:, [s1, s2]] = xs
            pred = f(x_tiled)
            effects[f'{s1}_{s2}'] = pred.reshape(grid_points, -1).mean(axis=1)    
    return np.array([v for _, v in effects.items()]).T


def explanation_expected_gradients(
        model,
        x_foreground,
        x_background,
    ):
    if str(type(model)) == "<class 'openxai.model.ArtificialNeuralNetwork'>":
        explainer = captum.attr.IntegratedGradients(model.network)
    else:
        explainer = captum.attr.IntegratedGradients(model)
    if hasattr(model, "predict_proba"):
        target = 1
    else:
        target = 0
    n_baselines = x_background.shape[0]
    baselines = torch.as_tensor(x_background, dtype=torch.float)
    inputs = torch.as_tensor(x_foreground, dtype=torch.float)
    for i in range(n_baselines):
        if i != 0:
            eg += explainer.attribute(
                    inputs=inputs, 
                    baselines=baselines[[i], :],
                    target=target,
                    n_steps=50
                )
        else:
            eg = explainer.attribute(
                    inputs=inputs, 
                    baselines=baselines[[i], :],
                    target=target,
                    n_steps=50
                )
    return eg / n_baselines