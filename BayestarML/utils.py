#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 15:45:46 2025

@author: LamirelFamily
"""

import arviz as az
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error

def find_pointwise_loo(trace):
    """
    Compute pointwise leave-one-out (LOO) log predictive densities.

    Parameters
    ----------
    trace : arviz.InferenceData
        Posterior trace containing log-likelihood values.

    Returns
    -------
    numpy.ndarray
        Array of pointwise LOO log-scores for each data point.
    """
    return az.loo(trace, pointwise=True, scale="log").loo_i.values


def train(model, filename=False, draw=1000, chains=2,
          target_accept=0.95, max_treedepth=20, nuts_sampler="pymc"):
    """
    Sample from a PyMC model and save the posterior trace.

    Runs MCMC sampling, computes log-likelihoods, and stores the trace 
    in a NetCDF file.

    Parameters
    ----------
    model : pm.Model
        The PyMC model to sample from.
    filename : str
        Path to save the resulting trace file.
    draw : int, optional
        Number of posterior samples per chain. Default is 1000.
    chains : int, optional
        Number of MCMC chains. Default is 2.
    target_accept : float, optional
        Target acceptance rate for the sampler. Default is 0.95.

    Returns
    -------
    arviz.InferenceData
        Posterior samples with computed log-likelihoods.
    """

    print('target_accept=', target_accept)
    trace = pm.sample(draws=draw, tune=int(1.5*draw), chains=chains,
                      cores=chains, model=model, target_accept=target_accept,
                      max_treedepth=max_treedepth,
                      nuts_sampler=nuts_sampler,
                      )

    # # NUTPIE DEBUGGING Extract the learned mean training predictions directly from the trace
    # sampled_mu_f = trace.posterior["mu_μ"].mean(dim=["chain", "draw"]).values
    # print("Training predictions SD:", np.std(sampled_mu_f))
    # print("Training predictions range:", sampled_mu_f.min(), sampled_mu_f.max())

    # vs = ["ls", "ls_v", "eta", "eta_v"]
    # print("Posteriors of interest:\n", az.summary(trace, var_names=vs))

    df = az.summary(trace)
    df.sort_values(by="ess_bulk", inplace=True)
    df_bad_ess = df[df["ess_bulk"]<400]
    print("AZ Stats for ESS Bulk < 400:\n", df_bad_ess)

    r_hat_values = az.rhat(trace)
    all_rhats = []
    for var in r_hat_values.data_vars:
        max_rhat = r_hat_values[var].max().values.item()
        all_rhats.append((var, max_rhat))
    print("rhats: ", all_rhats)

    trace.extend(pm.compute_log_likelihood(trace, model=model, var_names='y'))
    loo = az.loo(trace)
    print("loo trace: ", loo)

    #print indices of bad pareto ks
    pareto_k_values = loo.pareto_k.values
    bad_indices = np.where(pareto_k_values > 0.7)[0]
    print(f"Found {len(bad_indices)} problematic data points.")
    print("Indices of bad points:", bad_indices)

    if filename:
        trace.to_netcdf(filename)

    return trace

def mard(y_true, y_pred):
    """
    Compute the mean absolute relative difference (MARD) in percent.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        Mean absolute relative difference (percentage).
    """
    relative_diff = np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))
    return np.mean(relative_diff) * 100

def mrd(y_true, y_pred):
    """
    Compute the mean relative difference (MRD) in percent.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        Mean relative difference (percentage).
    """
    relative_diff = (np.array(y_true) - np.array(y_pred)) / np.array(y_true)
    return np.mean(relative_diff) * 100  

def model_pred_plotter(y_true, y_pred, y_pred_err,
                       target:str, save_folder:str,
                       hyperp_str:str,
                       y_true_err=None):
    """Plot predictions of a trained model against true values
    Saves a preds figure and a residuals figure
    """
    model = hyperp_str[:2]
    plt.figure(figsize=(8, 6))
    plt.errorbar(y_true, y_pred, yerr=y_pred_err, fmt='o', label='Predictions with Uncertainty', alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('True '+target)
    plt.ylabel('Predicted '+target)
    plt.title(model+' Predictions with Uncertainty')
    plt.legend()
    plt.savefig(save_folder+"/preds_"+hyperp_str+".pdf")
    plt.close()

    if y_true_err is not None:
        #make another plot with 1sig err cloud from test target sigs
        y_true, y_true_err, y_pred, y_pred_err = [
            y.to_numpy() if isinstance(y, pd.Series) else y
            for y in [y_true, y_true_err, y_pred, y_pred_err]
        ]
        sorted_indices = np.argsort(y_true)
        if len(y_pred_err.shape)>1:
            y_true, y_true_err, y_pred, y_pred_err[0], y_pred_err[1] = [
                y[sorted_indices] for y in [y_true, y_true_err, y_pred, y_pred_err[0], y_pred_err[1]]
            ]
        else:
            y_true, y_true_err, y_pred, y_pred_err = [
                y[sorted_indices] for y in [y_true, y_true_err, y_pred, y_pred_err]
            ]
        plt.figure(figsize=(8, 6))
        plt.errorbar(y_true, y_pred, yerr=y_pred_err, fmt='o', label='Predictions with Uncertainty', alpha=0.7)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        #1 sig err cloud ----------------
        y_true_lower = y_true - y_true_err
        y_true_upper = y_true + y_true_err
        plt.fill_between(y_true, y_true_lower, y_true_upper, color="red", alpha=0.2, label="1-Sigma True "+target+" Errors")
        #--------------------------------
        plt.xlabel('True '+target)
        plt.ylabel('Predicted '+target)
        plt.title(model+' Predictions with Uncertainty')
        plt.legend()
        plt.savefig(save_folder+"/preds_sigCloud_"+hyperp_str+".pdf")
        plt.close()

    plt.figure(figsize=(8, 6))
    plt.errorbar(y_true, y_pred - y_true, yerr=y_pred_err, fmt='o', label='Predictions with Uncertainty', alpha=0.7)
    plt.hlines(0, y_true.min(), y_true.max(), 'r', linestyle='--')
    plt.xlabel('True '+target)
    plt.ylabel('Residual '+target)
    plt.title(model+' Prediction Residuals')
    plt.legend()
    plt.savefig(save_folder+"/res_"+hyperp_str+".pdf")
    plt.close()

def get_results(posterior_draws, data, outputs_folder_path, dataset_key, target, hyperp_str):
    """
    """
    stds = posterior_draws.std(0)
    means = posterior_draws.mean(0)
    print("\n"+target+" predictions")
    print("means: ", means)
    print("stdvs: ", stds)
    print("Unorm "+target+": ", data["unorm_y_test"])

    print("\n"+target+" accuracy stats")
    print('MAE: ', mean_absolute_error(data["unorm_y_test"], means))
    print('MARD: ', mard(data["unorm_y_test"], means))
    print('MRD: ', mrd(data["unorm_y_test"], means))

    model_pred_plotter(data["unorm_y_test"], means, stds, target, outputs_folder_path, hyperp_str, y_true_err=data["unorm_y_test_err"])

    if target.startswith("log"):
        target = target[3:]
        hyperp_str = "GP_"+target+hyperp_str[7:]
        y_draws = 10**(posterior_draws)
        y_pred = y_draws.mean(0)
        y_pred_err = y_draws.std(0)
        df_physical = pd.read_csv("data/"+dataset_key+".txt")
        df_physical.set_index("ID", inplace=True)
        y_true = df_physical.loc[data["test_stars"], target]
        y_true_err = df_physical.loc[data["test_stars"], "e"+target]

        print("\n"+target+" predictions")
        print("means: ", y_pred)
        print("stdvs: ", y_pred_err)
        print("Unorm "+target+": ", y_true)

        print("\n"+target+" accuracy stats")
        print('MAE: ', mean_absolute_error(y_true, y_pred))
        print('MARD: ', mard(y_true, y_pred))
        print('MRD: ', mrd(y_true, y_pred))

        model_pred_plotter(y_true, y_pred, y_pred_err, target, outputs_folder_path, hyperp_str, y_true_err=y_true_err)

        #Is median better?? 
        y_p16 = np.percentile(y_draws, 16, axis=0)
        y_p50 = np.percentile(y_draws, 50, axis=0)
        y_p84 = np.percentile(y_draws, 84, axis=0)

        print("\n"+target+" accuracy stats on median pred")
        print('MAE: ', mean_absolute_error(y_true, y_p50))
        print('MARD: ', mard(y_true, means))
        print('MRD: ', mrd(y_true, means))

        model_pred_plotter(y_true, y_pred, np.array([y_p50-y_p16, y_p84-y_p50]), target, outputs_folder_path, hyperp_str+"MEDIAN", y_true_err=y_true_err)