#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 15:20:01 2025

@author: LamirelFamily
"""

import numpy as np
import pytensor.tensor as tt
import pymc as pm
import arviz as az
from utils import find_pointwise_loo
from tqdm.auto import tqdm
from preprocess import denormalise_err, denormalise_val
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import os
from numpy.linalg import solve

RANDOM_SEED = 5732 

def sample_latent_given_obs(x_obs, sigma_obs_diag, chol_s, n_param):
    """
    Draw one posterior sample of latent inputs given noisy, possibly-missing observations.

    Assumes a zero-mean multivariate normal prior with covariance from a
    Cholesky factor and heteroskedastic Gaussian measurement noise. Handles
    missing observations by conditioning on the observed subset.

    Parameters
    ----------
    x_obs : array-like, shape (n_param,)
        Observed inputs; NaNs indicate missing values.
    sigma_obs_diag : array-like, shape (n_param,)
        1-sigma measurement uncertainties; NaNs mark entries to ignore.
    chol_s : array-like, shape (n_param*(n_param+1)/2,)
        Flattened lower-triangular Cholesky factor of the prior covariance.
    n_param : int
        Dimensionality of the input space.

    Returns
    -------
    x_sample : ndarray, shape (n_param,)
        One draw from p(x | x_obs).
    """
    # Convert inputs to float arrays 
    x_obs          = np.asarray(x_obs,         dtype=float)
    sigma_obs_diag = np.asarray(sigma_obs_diag, dtype=float)

    # Build prior covariance from its Cholesky factor
    L = np.zeros((n_param, n_param))
    L[np.tril_indices(n_param)] = chol_s
    Sigma_prior = L @ L.T                   


    # which dimensions are observed
    observed_mask = np.isfinite(x_obs) & np.isfinite(sigma_obs_diag)
    missing_mask  = ~observed_mask

    if observed_mask.all():                             # no NaNs anywhere
        Sigma_obs = Sigma_prior + np.diag(sigma_obs_diag**2)

        mu_post   = Sigma_prior @ solve(Sigma_obs, x_obs)
        Sigma_post = Sigma_prior - Sigma_prior @ solve(Sigma_obs, Sigma_prior)
        return np.random.multivariate_normal(mu_post, Sigma_post)

    # If nothing is observed, sample from the prior
    if not observed_mask.any():
        return np.random.multivariate_normal(np.zeros(n_param), Sigma_prior)

    # Partition the prior covariance
    Sigma_oo = Sigma_prior[np.ix_(observed_mask, observed_mask)]
    Sigma_mo = Sigma_prior[np.ix_(missing_mask,  observed_mask)]
    Sigma_mm = Sigma_prior[np.ix_(missing_mask,  missing_mask)]

    # Observation‑space covariance (add noise only where we can)
    Sigma_obs = Sigma_oo + np.diag(sigma_obs_diag[observed_mask]**2)

    # Posterior parameters for the missing part

    X                     = solve(Sigma_obs, Sigma_mo.T).T      # Σ_mo Σ_obs⁻¹
    mu_m_given_o          = X @ x_obs[observed_mask]
    Sigma_m_given_o       = Sigma_mm - X @ Sigma_mo.T

    # Draw sample 
    x_sample              = np.empty(n_param)
    x_sample[observed_mask] = x_obs[observed_mask]                # keep data
    x_sample[missing_mask]  = np.random.multivariate_normal(
                                   mu_m_given_o, Sigma_m_given_o)
    return x_sample


def _predict_one_chain(chain_idx,
                       n_draws,
                       X, X_err,
                       chol_draws,
                       w_in_1_draws, b_1_draws,
                       w_1_2_draws, b_2_draws,
                       w_out_draws, b_out_draws,
                       n_param):
    """
    Generate posterior predictive draws for one chain of the HBNN.

    For each draw in the chain, samples latent inputs given observed
    features and their errors, then applies the neural network forward pass
    to obtain predictions for all test points.

    Parameters
    ----------
    chain_idx : int
        Index of the chain to process.
    n_draws : int
        Number of posterior draws per chain.
    X : ndarray, shape (N_test, n_param)
        Test inputs (observed).
    X_err : ndarray, shape (N_test, n_param)
        Measurement uncertainties for test inputs.
    chol_draws : ndarray, shape (S, L)
        Draws of flattened Cholesky factors of the prior covariance.
    w_in_1_draws, b_1_draws, w_1_2_draws, b_2_draws, w_out_draws, b_out_draws : ndarrays
        Draws of network weights and biases matching model shapes.
    n_param : int
        Input dimensionality.

    Returns
    -------
    idx_slice : slice
        Slice into the global draws axis corresponding to this chain.
    Y_row : ndarray, shape (n_draws, N_test)
        Posterior predictive samples for the chain.
    """
    start = chain_idx * n_draws
    stop  = start + n_draws
    draws = slice(start, stop)

    Y = np.empty((n_draws, X.shape[0]), dtype=float)

    s_local = 0
    for s_idx in range(start, stop):

        # pull parameters for draw s_idx 
        chol_s   = chol_draws[s_idx]
        w1, b1   = w_in_1_draws[s_idx],  b_1_draws[s_idx]
        w2, b2   = w_1_2_draws[s_idx],   b_2_draws[s_idx]
        w_out    = w_out_draws[s_idx]
        b_out    = b_out_draws[s_idx]

        # latent + forward for every test star
        for i in range(X.shape[0]):
            x_lat = sample_latent_given_obs(
                X[i], X_err[i], chol_s, n_param
            )
            Y[s_local, i] = forward_pass(
                x_lat, w1, b1, w2, b2, w_out, b_out
            )

        s_local += 1

    return draws, Y


def sample_post_pred_HBNN_para(trace, X, X_er, n_hidden, n_param, target,
                          n_jobs=None):
    """
    Parallel posterior predictive for HBNN with latent-input sampling.

    Extracts weight/bias and covariance draws from an ArviZ trace, samples latent
    inputs given test observations and their errors, and computes posterior predictive
    draws in parallel across chains. Returns denormalized mean, std, and HDI bounds,
    along with pointwise LOO scores.

    Parameters
    ----------
    trace : arviz.InferenceData
        Posterior samples for the HBNN model.
    X : array-like, shape (N_test, n_param)
        Test inputs (observed).
    X_er : array-like, shape (N_test, n_param)
        Test input measurement uncertainties.
    n_hidden : int
        Hidden layer width (for reshaping network draws).
    n_param : int
        Input dimensionality (e.g., 3 or 4).
    target : Any
        Denormalization context used by `denormalise_val/err`.
    n_jobs : int, optional
        Number of worker processes; defaults to available CPUs.

    Returns
    -------
    tuple
        (y_draws, lpd_HBNN)
        - denormalised y_draws : ndarray, shape (nb of posterior predictive draws, N_test)
        - lpd_HBNN : ndarray
            Pointwise LOO log predictive densities for the model.
    """
    lpd_HBNN = find_pointwise_loo(trace)
    n_chol   = 10 if n_param == 4 else 6

    posterior  = trace.posterior
    n_chains   = posterior.sizes["chain"]
    n_draws    = posterior.sizes["draw"]
    S          = n_chains * n_draws

    X_test     = np.asarray(X,     float)
    X_test_err = np.asarray(X_er,  float)
    N_test     = X_test.shape[0]

    chol_draws   = np.asarray(posterior["Omega"]).reshape(S, n_chol)
    w_in_1_draws = np.asarray(posterior["w_in_1"]).reshape(S, n_hidden, n_param)
    b_1_draws    = np.asarray(posterior["bias_1"]).reshape(S, n_hidden)
    w_1_2_draws  = np.asarray(posterior["w_1_2"]).reshape(S, n_hidden, n_hidden)
    b_2_draws    = np.asarray(posterior["bias_2"]).reshape(S, n_hidden)
    w_out_draws  = np.asarray(posterior["w_2_out"]).reshape(S, n_hidden)
    b_out_draws  = np.asarray(posterior["bias_out"]).reshape(S)

    predictions = np.empty((S, N_test), dtype=float)

    if n_jobs is None:
        n_jobs = os.cpu_count()

    worker = partial(
        _predict_one_chain,
        n_draws=n_draws,
        X=X_test,
        X_err=X_test_err,
        chol_draws=chol_draws,
        w_in_1_draws=w_in_1_draws, b_1_draws=b_1_draws,
        w_1_2_draws=w_1_2_draws, b_2_draws=b_2_draws,
        w_out_draws=w_out_draws, b_out_draws=b_out_draws,
        n_param=n_param
    )

    print(f"starting parallel prediction on {n_jobs} worker(s)…")
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = {ex.submit(worker, c): c for c in range(n_chains)}

        for fut in as_completed(futures):
            chain_done = futures[fut]                  # chain index
            idx_slice, Y = fut.result()                # fill results
            predictions[idx_slice] = Y
            print(f"finished chain {chain_done+1}/{n_chains}")
            

    return denormalise_val(predictions, target), lpd_HBNN



def forward_pass(x_latent, w_in_1, b1, w_1_2, b2, w_2_out, b_out):
    """
    Two-layer ReLU network forward pass.

    Applies two affine layers with ReLU activations and a final linear readout.

    Parameters
    ----------
    x_latent : ndarray, shape (n_param,)
        Latent input features.
    w_in_1 : ndarray, shape (n_hidden, n_param)
        First-layer weights.
    b1 : ndarray, shape (n_hidden,)
        First-layer biases.
    w_1_2 : ndarray, shape (n_hidden, n_hidden)
        Second-layer weights.
    b2 : ndarray, shape (n_hidden,)
        Second-layer biases.
    w_2_out : ndarray, shape (n_hidden,)
        Output weights.
    b_out : float
        Output bias.

    Returns
    -------
    float
        Network output for the given input.
    """
    # Layer 1 with ReLU
    layer1 = x_latent @ w_in_1.T + b1
    layer1 = np.maximum(layer1, 0)  
    
    # Layer 2 with ReLU 
    layer2 = layer1 @ w_1_2 + b2
    layer2 = np.maximum(layer2, 0) 
    
    # Final output (linear)
    return layer2 @ w_2_out + b_out

# def sample_post_pred_HBNN(trace, X, X_er, n_hidden, n_param, target):
#     """
#     Sequential posterior predictive sampling for a hierarchical Bayesian neural network (HBNN).

#     Draws predictions from the posterior distribution of an HBNN model by iterating over
#     all posterior samples. For each draw, the function samples latent input variables
#     conditioned on observed features and their measurement errors, then performs a
#     forward pass through the network to obtain predictive samples.

#     Parameters
#     ----------
#     trace : arviz.InferenceData
#         Posterior samples from a fitted HBNN model.
#     X : array-like, shape (N_test, n_param)
#         Test input data.
#     X_er : array-like, shape (N_test, n_param)
#         Measurement uncertainties for each test input feature.
#     n_hidden : int
#         Number of hidden units per layer in the neural network.
#     n_param : int
#         Dimensionality of the input features (e.g., 3 or 4).
#     target : Any
#         Normalization context used by `denormalise_val` and `denormalise_err`.

#     Returns
#     -------
#     tuple
#         (pred, lpd_HBNN)
#         - pred : ndarray, shape (2, N_test)
#             Posterior predictive mean and standard deviation for each test input,
#             denormalized to the original target scale.
#         - lpd_HBNN : ndarray
#             Pointwise LOO log predictive densities for the HBNN model.
#     """
#     lpd_HBNN = find_pointwise_loo(trace)
    
#     if n_param == 4:
#         n_chol = 10
#     if n_param == 3:
#         n_chol = 6
    
#     posterior = trace.posterior
#     n_chains = posterior.sizes["chain"]
#     n_draws = posterior.sizes["draw"]
#     S = n_chains * n_draws
    
#     x_test_er = np.array(X_er)
#     x_test_ar = np.array(X)
    
#     posterior = trace.posterior
#     # Flatten chain/draw so we get S = n_chains * n_draws samples
#     n_chains = posterior.sizes["chain"]
#     n_draws  = posterior.sizes["draw"]
#     S        = n_chains * n_draws

#     # Extract relevant variables as numpy arrays 
    
#     corr_draws = np.array(posterior["Omega_corr"]).reshape(S, n_param, n_param) 

#     # Cholesky factor from the LKJ prior
#     chol_draws = np.array(posterior["Omega"]).reshape(S, n_chol)


#     # NN weights/biases
#     w_in_1_draws = np.array(posterior["w_in_1"]).reshape(S, n_hidden, n_param)   # shape => (S, n_hidden, 4)
#     b_1_draws    = np.array(posterior["bias_1"]).reshape(S, n_hidden)
#     w_1_2_draws  = np.array(posterior["w_1_2"]).reshape(S, n_hidden, n_hidden)   
#     b_2_draws    = np.array(posterior["bias_2"]).reshape(S, n_hidden)      
#     w_2_out_draws = np.array(posterior["w_2_out"]).reshape(S, n_hidden)   
#     b_out_draws   = np.array(posterior["bias_out"]).reshape(S,)

#     N_test = X.shape[0]
#     predictions = np.zeros((S, N_test))
    
#     total_steps = S           # every posterior draw × every test sta

#     pbar = tqdm(total=total_steps, desc="Posterior‑predictive draws")

#     # Loop over each posterior sample and each test row
#     s_idx = 0
#     for chain_i in range(n_chains):
#         print("Sampling chain number: ", chain_i, '/', n_chains)
#         for draw_i in range(n_draws):
#             # Extract this draw's parameters
#             chol_s = chol_draws[s_idx]
#             w_in_1_s = w_in_1_draws[s_idx]
#             b_1_s    = b_1_draws[s_idx]
#             w_1_2_s  = w_1_2_draws[s_idx]
#             b_2_s    = b_2_draws[s_idx]
#             w_2_out_s = w_2_out_draws[s_idx]
#             b_out_s   = b_out_draws[s_idx]

#             for i in range(N_test):
#                 #print(x_test_ar[i])
#                 x_obs_i = x_test_ar[i]         
#                 x_err_i = x_test_er[i]           

#                 # 1) sample X_latent_i ~ p(X_latent|X_obs=x_obs_i)
#                 x_latent_i = sample_latent_given_obs(
#                     x_obs_i,
#                     x_err_i,      # stdev for each feature
#                     chol_s,
#                     n_param
#                 )

#                 # 2) forward pass to get y_pred
#                 y_pred_i = forward_pass(
#                     x_latent_i,
#                     w_in_1_s, b_1_s,
#                     w_1_2_s,  b_2_s,
#                     w_2_out_s, b_out_s
#                 )

#                 predictions[s_idx, i] = y_pred_i 
#                 #pbar.update() 
#                 #print('tick')
            
#             s_idx += 1
#             pbar.update(1) 
#     pbar.close()
    
#     pred = np.array([denormalise_val(predictions.mean(axis=0), target),
#                       denormalise_err(predictions.std(axis=0), target)])
    
    
#     return pred, lpd_HBNN

def posterior_predictive_GP(
    gp_model, mu_gp, log_var_gp, trace,
    X_new_raw, X_er_new_raw, Xu, Xu_var,
    n_param, target,
    var_cols_x=(0,1),        # columns of X used in variance GP
    var_cols_xerr=(0,1),     # columns of X_err used in variance GP
    random_seed=42,
):
    """
    Posterior predictive for sparse fully heteroscedastic GP
    with log-variance GP:
        log_var = alpha_log_var + latent_GP(X_var)
        sigma   = exp(0.5 * log_var)

    Parameters
    ----------
    gp_model : pm.Model
        Fitted PyMC model.
    mu_gp : SparseLatent
        Sparse latent GP for the mean.
    log_var_gp : SparseLatent
        Sparse latent GP for log variance.
    trace : arviz.InferenceData
        Posterior samples.
    X_new_raw : array, shape (N_new, D)
        New inputs for mean GP, may contain NaNs.
    X_er_new_raw : array, shape (N_new, D_err)
        New input errors, may contain NaNs.
    Xu : ndarray
        Inducing points for mean GP.
    Xu_var : ndarray
        Inducing points for variance GP.
    n_param : int
        Number of mean-GP input dimensions.
    target : Any
        Used by denormalise_val / denormalise_err.
    var_cols_x : tuple
        Columns from X used in variance GP.
    var_cols_xerr : tuple
        Columns from X_err used in variance GP.
    random_seed : int
        RNG seed for posterior predictive y draws.

    Returns
    -------
    denormalised y_draws : ndarray, shape (nb of posterior predictive draws, N_test)
    lpd_GP : ndarray
        Pointwise LOO log predictive density.
    """
    lpd_GP = find_pointwise_loo(trace)

    X_new_raw = np.asarray(X_new_raw, float)
    X_er_new_raw = np.asarray(X_er_new_raw, float)

    N_new = X_new_raw.shape[0]

    # Missingness masks for mean GP inputs
    mask_mu = ~np.isfinite(X_new_raw)

    # Build variance-GP raw design matrix exactly as in training
    X_var_new_raw = np.hstack([
        X_new_raw[:, var_cols_x],
        X_er_new_raw[:, var_cols_xerr],
    ])

    mask_var = ~np.isfinite(X_var_new_raw)
    D_var = X_var_new_raw.shape[1]

    rng = np.random.default_rng(random_seed)

    with gp_model:
        # Observed placeholders
        X_mu_obs = X_new_raw
        X_var_obs = X_var_new_raw

        # Latent imputations for missing mean inputs
        X_mu_latent = pm.Normal(
            "X_new_latent",
            mu=0.0,
            sigma=1.0,
            shape=(N_new, n_param),
        )

        # Latent imputations for missing variance inputs
        X_var_latent = pm.Normal(
            "X_var_new_latent",
            mu=0.0,
            sigma=1.0,
            shape=(N_new, D_var),
        )

        # Fill missing values
        X_new = tt.where(mask_mu, X_mu_latent, X_mu_obs)
        X_var_new = tt.where(mask_var, X_var_latent, X_var_obs)

        # Mean GP conditional
        f_mu_pred = mu_gp.conditional_marginal("f_mu_pred", X_new, Xu)

        # Log-variance GP conditional
        log_var_pred_latent = log_var_gp.conditional_marginal(
            "log_var_pred_latent",
            X_var_new,
            Xu_var,
        )

        # Add intercept from fitted model
        log_var_pred = pm.Deterministic(
            "log_var_pred",
            gp_model["alpha_log_var"] + log_var_pred_latent
        )

        y = pm.Normal("y_pred", mu=f_mu_pred, sigma=pm.math.exp(0.5 * log_var_pred), shape=N_new)

        # Draw posterior predictive samples
        ppc = pm.sample_posterior_predictive(
            trace,
            var_names=["f_mu_pred", "log_var_pred", "y_pred"],
            predictions=True,
            random_seed=random_seed,
        )

    # Posterior predictive draws for y
    y_draws = ppc.predictions["y_pred"].stack(sample=("chain", "draw")).values

    return denormalise_val(y_draws, target).T, lpd_GP

# def posterior_predictive_GP(
#     gp_model, μ_gp, lg_σ_gp, trace,
#     X_new_raw, X_er_new_raw, Xu, Xu_er,
#     n_param, target):
#     """
#     Generate posterior predictive samples from a hierarchical sparse GP model.

#     Computes predictive means and uncertainties for new inputs using a pair of
#     Gaussian processes — one modeling the mean function (μ_GP) and one modeling
#     the log noise variance (σ_GP). Missing input values are imputed via latent
#     normal variables before prediction.

#     Parameters
#     ----------
#     gp_model : pm.Model
#         The fitted PyMC GP model containing both mean and noise processes.
#     μ_gp : SparseLatent
#         Sparse latent GP modeling the predictive mean.
#     lg_σ_gp : SparseLatent
#         Sparse latent GP modeling the log noise variance.
#     trace : arviz.InferenceData
#         Posterior samples from the fitted GP model.
#     X_new_raw : array-like, shape (N_new, n_param)
#         New input data for the mean process; may include NaN values.
#     X_er_new_raw : array-like, shape (N_new, n_param)
#         New input data for the noise process; may include NaN values.
#     Xu : ndarray
#         Inducing points used for the mean GP.
#     Xu_er : ndarray
#         Inducing points used for the noise GP.
#     n_param : int
#         Number of input dimensions.
#     target : Any
#         Normalization context for denormalizing predictions and errors.

#     Returns
#     -------
#     tuple
#         (stats, lpd_GP)
#         - stats : ndarray, shape (4, N_new)
#             Stacked posterior predictive summaries:
#             [mean, std, lower_hdi, upper_hdi], all denormalized.
#         - lpd_GP : ndarray
#             Pointwise LOO log predictive densities for the GP model.
#     """
#     lpd_GP = find_pointwise_loo(trace)
#     N_new = X_new_raw.shape[0]

#     # Build masks for missing sata
#     mask_mu = ~np.isfinite(X_new_raw)
#     mask_er = ~np.isfinite(X_er_new_raw)

#     with gp_model:
#         X_mu_obs = X_new_raw
#         X_er_obs = X_er_new_raw

#         # Latent replacements
#         X_mu_latent = pm.Normal(
#             "X_new_latent", mu=0, sigma=1,
#             shape=(N_new, n_param)
#         )
#         X_er_latent = pm.Normal(
#             "X_er_new_latent", mu=0, sigma=1,
#             shape=(N_new, n_param)
#         )

#         # Stitch together observed and latent
#         X_new = tt.where(mask_mu, X_mu_latent, X_mu_obs)

#         X_er_new = tt.where(mask_er, X_er_latent, X_er_obs)


#         # Conditional GPs
#         f_mu_pred   = μ_gp.conditional_marginal("f_mu_pred", X_new, Xu)
#         f_logσ_pred = lg_σ_gp.conditional_marginal("σ_pred", X_er_new, Xu_er)

#         # Joint posterior predictive over both
#         ppc = pm.sample_posterior_predictive(
#             trace,
#             var_names=["f_mu_pred","σ_pred"]
#         )
        
#         arr = ppc.posterior_predictive['f_mu_pred']
#         arr_combined = arr.stack(sample=("chain", "draw")).values

#         print(arr_combined.shape)
#         # np.savetxt("post_pred_GP_rad.txt", arr_combined)


        
#         summary_stats = az.summary(ppc.posterior_predictive, var_names="f_mu_pred")

#         sum_er = az.summary(ppc.posterior_predictive, var_names="σ_pred")
#         summary_er = denormalise_err(np.exp(sum_er['mean']), target)

#         Y_pred = denormalise_val(summary_stats['mean'], target)
#         er_stat = np.sqrt(np.array(summary_er)**2 + (denormalise_err(np.array(summary_stats['sd']), target))**2)

#         hdi_low = - np.array(summary_er) + denormalise_val(np.array(summary_stats['hdi_3%']), target)
#         hdi_hi = np.array(summary_er) + denormalise_val(np.array(summary_stats['hdi_97%']), target)
#         stats = np.array([Y_pred, er_stat, hdi_low, hdi_hi])

#     return stats, lpd_GP

def sample_pred_BART(model, X, X_er, target, draws=1000, chains=2):
    """
    Generate posterior predictive samples and LOO scores for a BART model.

    Samples from the posterior and posterior predictive distributions of a
    Bayesian Additive Regression Trees (BART) model, computes leave-one-out
    (LOO) log predictive densities, and returns denormalized prediction
    summaries for new data.

    Parameters
    ----------
    model : pm.Model
        PyMC BART model to sample from.
    X : array-like
        Input feature matrix for prediction.
    X_er : array-like
        Measurement uncertainties for input features.
    target : Any
        Normalization context for denormalizing predictions and errors.
    draws : int, optional
        Number of posterior draws per chain. Default is 1000.
    chains : int, optional
        Number of MCMC chains. Default is 2.

    Returns
    -------
    tuple
        (denormalised y_draws, lpd_BART)
        - denormalised y_draws : ndarray, shape (nb of posterior predictive draws, N_test)
        - lpd_BART : ndarray
            Pointwise LOO log predictive densities for the BART model.
    """
    with model:
        trace = pm.sample(draws=draws, tune=draws, chains=chains)
        trace.extend(pm.compute_log_likelihood(trace))
        # pp = pm.sample_posterior_predictive(trace)

    lpd_BART = find_pointwise_loo(trace)
    # print(az.rhat(trace))
        
    with model:
        pm.set_data({'X': X,
                     'X_er': X_er
            })
        pred = pm.sample_posterior_predictive(trace, predictions=True)

        y_draws = pred.predictions["y"].stack(sample=("chain", "draw")).values
    
    return denormalise_val(y_draws, target).T, lpd_BART




