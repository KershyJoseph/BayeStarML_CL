#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 15:10:27 2024

@author: LamirelFamily
"""

import arviz as az
import numpy as np
import pymc as pm
import pandas as pd

RANDOM_SEED = 5732
rng = np.random.default_rng(RANDOM_SEED)


def median_clip(X_train, X_test):
    """
    Apply median-based clipping and feature expansion to training and test data.

    Centers each feature by its median (computed from the training set) and splits 
    deviations into positive and negative components. This doubles the number of 
    features and allows the model to capture asymmetric behavior above and below 
    the median.

    Parameters
    ----------
    X_train : array-like
        Training data array of shape (n_samples, n_features).
    X_test : array-like
        Test data array of shape (m_samples, n_features).

    Returns
    -------
    tuple of ndarray
        (X_train_comb, X_test_comb)
        - X_train_comb : ndarray
            Transformed training data of shape (n_samples, 2 × n_features).
        - X_test_comb : ndarray
            Transformed test data of shape (m_samples, 2 × n_features).
    """
    X_train = np.array(X_train)
    median = np.median(X_train, axis=0)
    X_train_u = (X_train - median).clip(max=0)
    X_train_l = (X_train - median).clip(min=0)
    X_test = np.array(X_test)
    X_test_u = (X_test - median).clip(max=0)
    X_test_l = (X_test - median).clip(min=0)
    
    X_train_comb = np.concatenate((X_train_u, X_train_l), axis=1)
    X_test_comb = np.concatenate((X_test_u, X_test_l), axis=1)
    
    return X_train_comb, X_test_comb

def stacking_continuous(
    X,
    X_test,
    lpd_point,
    tau_mu,
    tau_sigma,
    *,
    test=True,
):
    """
    Build a continuous-feature Bayesian stacking model in PyMC.

    Constructs a hierarchical Bayesian model that learns feature-dependent 
    stacking weights for combining predictions from multiple candidate models 
    (e.g., BART, HBNN, GP). The weights are learned based on pointwise LOO 
    log predictive densities (`lpd_point`) and depend on continuous input features 
    provided in `X`. Optionally computes test-set weights for new inputs.

    Parameters
    ----------
    X : ndarray
        Training feature matrix (N × D) used to model the dependence of stacking weights.
    X_test : ndarray
        Test feature matrix (N* × D) for evaluating stacking weights on unseen data.
    lpd_point : ndarray
        Matrix of pointwise LOO log predictive densities (N × K), 
        where K is the number of candidate models.
    tau_mu : float
        Prior scale for the mean of the feature coefficients (`beta`).
    tau_sigma : float
        Prior scale for the standard deviation of the feature coefficients (`beta`).
    test : bool, optional
        Whether to compute and store feature-dependent stacking weights 
        for the test set. Default is True.

    Returns
    -------
    pm.Model
        PyMC model defining the continuous-feature Bayesian stacking framework, 
        including priors over feature coefficients and softmax-normalized weights 
        for model combination.
    """
    N = X.shape[0]
    d = X.shape[1]  # Number of continuous features
    N_test = X_test.shape[0]
    K = lpd_point.shape[1]  # Number of candidate models
    X_test = np.nan_to_num(X_test, nan=0.0)
    #print(X_test)

    with pm.Model() as model:
        # Priors for continuous features
        mu = pm.Normal("mu", mu=0, sigma=tau_mu, shape=(K-1,))
        sigma = pm.HalfNormal("sigma", sigma=tau_sigma, shape=(K-1,))
        beta_con = pm.Normal("beta_con", mu=0, sigma=1, shape=(d, K-1))

        # Deterministic stacking weights
        beta = pm.Deterministic(
            "beta", (sigma * beta_con + mu).T
        )

        assert beta.eval().shape == (K - 1, d)

        # Calculate stacking weights for training set

        f = pm.Deterministic("f", pm.math.concatenate([pm.math.dot(X, beta.T), np.zeros((N, 1))], axis=1))

        assert f.eval().shape == (N, K)

        # Log-softmax for stacking weights (log probability in unconstrained space)
        log_w = pm.math.log_softmax(f, axis=1)
        w = pm.Deterministic("w", np.exp(log_w))


        # Log probability of LOO training scores weighted by stacking weights
        logp = pm.math.logsumexp(lpd_point + log_w, axis=1)
        
        pot = pm.Potential("logp", pm.math.sum(logp))


        if test:
            # Calculate stacking weights for the test set
            f_test = pm.Deterministic("f_test", pm.math.concatenate([pm.math.dot(X_test, beta.T), np.zeros((N_test, 1))], axis=1))
            w_test = pm.Deterministic("w_test", pm.math.softmax(f_test, axis=1))

    return model

def _to_2d_draws(arr, N_test=None, name="arr"):
    """
    Ensure posterior predictive samples have shape (S, N_test).

    Accepts arrays in common PyMC formats and converts them to a 2D array
    where S is the total number of draws and N_test is the number of
    prediction points.

    Supported input shapes
    ----------------------
    (S, N)             : already flattened draws
    (chain, draw, N)   : flattened to (chain*draw, N)
    (N,)               : treated as a single draw -> (1, N)

    Parameters
    ----------
    arr : array-like
        Posterior predictive samples.
    N_test : int, optional
        Expected number of prediction points (checked if provided).
    name : str, default "arr"
        Variable name used in error messages.

    Returns
    -------
    ndarray
        Array of shape (S, N_test).
    """
    a = np.asarray(arr)
    if a.ndim == 1:
        a = a[None, :]
    elif a.ndim == 3:
        # (chain, draw, N) -> (S, N)
        a = a.reshape(-1, a.shape[-1])
    elif a.ndim != 2:
        raise ValueError(f"{name} must have 1D, 2D, or 3D shape; got {a.shape}")
    if N_test is not None and a.shape[1] != N_test:
        raise ValueError(f"{name} second dim must be N_test={N_test}; got {a.shape}")
    return a

def run_stack(
    BART_post_pred, HBNN_post_pred, GP_post_pred,
    x_train, x_pred,
    lpd_BART, lpd_HBNN, lpd_GP,
    tau_mu=1.0, tau_sigma=0.5,
    draws=1000, chains=4,
    random_seed=42,
):
    """
    Fit a continuous-feature Bayesian stacking model and combine posterior
    predictive draws from BART, HBNN, and GP using posterior draws of the
    stacking weights.

    The stacking model is trained on pointwise log predictive densities
    (LPDs) from the base models and covariates derived from `x_train` and
    `x_pred`. Posterior draws of the test-set weights are then used to form
    draw-by-draw stacked predictions, propagating uncertainty from both the
    base model predictions and the stacking weights.

    Parameters
    ----------
    BART_post_pred, HBNN_post_pred, GP_post_pred : array-like
        Posterior predictive draws from the three base models. Each input may
        have shape (S, N_test), (chain, draw, N_test), or (N_test,).
    x_train : array-like
        Training covariates used to construct stacking features.
    x_pred : array-like
        Test covariates at which stacking weights are evaluated.
    lpd_BART, lpd_HBNN, lpd_GP : array-like
        Pointwise log predictive densities for the three base models on the
        training data, each of shape (N_train,).
    tau_mu : float, default 1.0
        Prior scale for the mean function in the stacking model.
    tau_sigma : float, default 0.5
        Prior scale for the variance function in the stacking model.
    draws : int, default 1000
        Number of posterior draws for the stacking model.
    chains : int, default 4
        Number of MCMC chains.
    random_seed : int, default 42
        Random seed used for posterior sampling and draw alignment.

    Returns
    -------
    trace : arviz.InferenceData
        Posterior samples from the stacking model, including the test-set
        stacking weights `w_test`.
    y_stack_draws : ndarray
        Stacked posterior predictive draws with shape (S_stack, N_test).
    w_draws : ndarray
        Posterior draws of the stacking weights with shape (S_w, N_test, 3).

    Notes
    -----
    Base-model posterior predictive arrays and stacking-weight draws may
    contain different numbers of samples (e.g., different chains or draws).
    All arrays are first reshaped to (S, N_test). A common number of samples
    `S_stack` is then chosen as the minimum across all sources, and draws are
    randomly sampled (with replacement if needed) so that predictions and
    weights can be paired draw-by-draw when forming the stacked predictions.

    Raises
    ------
    ValueError
        If the stacking model returns a number of weights different from the
        expected three base models.
    """

    # Build BHS inputs 
    X1, X2 = median_clip(x_train, x_pred)
    lpd_point = np.vstack((lpd_BART, lpd_HBNN, lpd_GP)).T  # (N_train, K=3)

    model = stacking_continuous(X1, X2, lpd_point, tau_mu, tau_sigma)

    # Sample BHS posterior to get weight draws
    with model:
        trace = pm.sample(
            draws=draws,
            chains=chains,
            random_seed=random_seed,
            target_accept=0.9,
            progressbar=True,
        )

    # Extract posterior draws of weights 
    # trace.posterior["w_test"] has dims (chain, draw, N_test, K)
    w = trace.posterior["w_test"].values
    # Flatten chain/draw -> S_w
    w_draws = w.reshape(-1, w.shape[-2], w.shape[-1])  # (S_w, N_test, K)

    N_test = w_draws.shape[1]
    K = w_draws.shape[2]
    if K != 3:
        raise ValueError(f"Expected K=3 base models; got K={K}")

    # Make sure base posterior predictive arrays are (S, N_test)
    bart = _to_2d_draws(BART_post_pred, N_test=N_test, name="BART_post_pred")
    hbnn = _to_2d_draws(HBNN_post_pred, N_test=N_test, name="HBNN_post_pred")
    gp   = _to_2d_draws(GP_post_pred,   N_test=N_test, name="GP_post_pred")

    # Align number of draws between weights and base predictions 
    # We can pair draws by sampling indices .
    rng = np.random.default_rng(random_seed)
    S_w = w_draws.shape[0]
    S_b = bart.shape[0]
    S_h = hbnn.shape[0]
    S_g = gp.shape[0]

    S_stack = min(S_w, S_b, S_h, S_g)  # safe default
    idx_w = rng.choice(S_w, size=S_stack, replace=False) if S_w >= S_stack else rng.choice(S_w, size=S_stack, replace=True)
    idx_b = rng.choice(S_b, size=S_stack, replace=False) if S_b >= S_stack else rng.choice(S_b, size=S_stack, replace=True)
    idx_h = rng.choice(S_h, size=S_stack, replace=False) if S_h >= S_stack else rng.choice(S_h, size=S_stack, replace=True)
    idx_g = rng.choice(S_g, size=S_stack, replace=False) if S_g >= S_stack else rng.choice(S_g, size=S_stack, replace=True)

    w_use = w_draws[idx_w, :, :]             # (S_stack, N_test, 3)
    bart_use = bart[idx_b, :]                # (S_stack, N_test)
    hbnn_use = hbnn[idx_h, :]                # (S_stack, N_test)
    gp_use   = gp[idx_g, :]                  # (S_stack, N_test)

    # Combine draw-by-draw to propagate weight uncertainty
    # y_stack[s,i] = sum_k w[s,i,k] * y_k[s,i]
    y_stack_draws = (
        w_use[:, :, 0] * bart_use
        + w_use[:, :, 1] * hbnn_use
        + w_use[:, :, 2] * gp_use
    )

    return trace, y_stack_draws, w_draws
