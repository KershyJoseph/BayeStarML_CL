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

def run_stack(BART_pred, HBNN_pred, GP_pred,
              x_train, x_pred, lpd_BART, lpd_HBNN, lpd_GP,
              tau_mu=1.0, tau_sigma=0.5):
    """
    Perform Bayesian hierarchical stacking of model predictions.

    Combines predictions from multiple probabilistic models (BART, HBNN, GP) 
    using a hierarchical Bayesian stacking model. The stacking weights are 
    learned from leave-one-out (LOO) log predictive densities and a 
    median-clipped feature representation of the inputs.

    Parameters
    ----------
    BART_pred : tuple of array-like
        BART predictions as (mean, std, lower, upper).
    HBNN_pred : tuple of array-like
        HBNN predictions as (mean, std, lower, upper).
    GP_pred : tuple of array-like
        GP predictions as (mean, std, lower, upper).
    x_train : array-like
        Training input features.
    x_pred : array-like
        Prediction input features.
    lpd_BART : array-like
        Pointwise log predictive densities for the BART model.
    lpd_HBNN : array-like
        Pointwise log predictive densities for the HBNN model.
    lpd_GP : array-like
        Pointwise log predictive densities for the GP model.
    tau_mu : float, optional
        Prior scale parameter for the mean weights. Default is 1.0.
    tau_sigma : float, optional
        Prior scale parameter for the weight variance. Default is 0.5.

    Returns
    -------
    tuple
        (comb_res, comb_er, comb_lo, comb_hi, pivoted_w)
        - comb_res : pandas.Series
            Combined predictive means from the stacked ensemble.
        - comb_er : pandas.Series
            Combined predictive uncertainties.
        - comb_lo : pandas.Series
            Combined lower credible interval bounds.
        - comb_hi : pandas.Series
            Combined upper credible interval bounds.
        - pivoted_w : pandas.DataFrame
            Learned stacking weights for each model across test points.
    """
    X1, X2 = median_clip(x_train, x_pred)
    lpd_point = np.vstack((lpd_BART, lpd_HBNN, lpd_GP)).T
    model = stacking_continuous(X1, X2, lpd_point, tau_mu, tau_sigma)
    
    with model as model:
        trace = pm.sample(1000, chains=4)
        trace.to_netcdf("BHS_trace.nc")
        summary_w = az.summary(trace, var_names=['w_test'])
        summary_w = summary_w.reset_index()
        print(summary_w)

        summary_w[['variable', 'index_1', 'index_2']] = summary_w['index'].str.extract(r'(\w+)\[(\d+), (\d+)\]')

        summary_w['index_1'] = pd.to_numeric(summary_w['index_1'])
        summary_w['index_2'] = pd.to_numeric(summary_w['index_2'])

        pivoted_w = summary_w.pivot(index='index_1', columns='index_2', values='mean')
        # Rename the columns for clarity
        pivoted_w.columns = ['w_test_0', 'w_test_1', 'w_test_2']

        # Reset the index so that you have a DataFrame with 98 rows
        pivoted_w = pivoted_w.reset_index(drop=True)
        # Now pivoted_w contains 98 rows with two columns (w_test_0 and w_test_1)
        
        comb_res = BART_pred[0] * pivoted_w['w_test_0'] + HBNN_pred[0] * pivoted_w['w_test_1'] + GP_pred[0] * pivoted_w['w_test_2']
        comb_er = BART_pred[1] * pivoted_w['w_test_0'] + HBNN_pred[1] * pivoted_w['w_test_1'] + GP_pred[1] * pivoted_w['w_test_2']
        comb_lo = BART_pred[2] * pivoted_w['w_test_0'] + HBNN_pred[2] * pivoted_w['w_test_1'] + GP_pred[2] * pivoted_w['w_test_2']
        comb_hi = BART_pred[3] * pivoted_w['w_test_0'] + HBNN_pred[3] * pivoted_w['w_test_1'] + GP_pred[3] * pivoted_w['w_test_2']
    
    return comb_res, comb_er, comb_lo, comb_hi, pivoted_w
