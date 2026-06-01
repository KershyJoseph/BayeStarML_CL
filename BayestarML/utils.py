#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 15:45:46 2025

@author: LamirelFamily
"""

import arviz as az
import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

def get_dataset(data_file, star_class='MS', logL=False):
    """
    ***Added logL option***
    Load and clean a stellar dataset for a given star class.

    Reads a tab-separated file of stellar parameters and their uncertainties,
    filters rows matching the specified class, removes entries with missing
    values, and returns the cleaned subset.

    Parameters
    ----------
    data_file : str
        Path to the tab-separated data file.
    star_class : str
        Stellar class to filter by (e.g., 'MS').

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame containing stars of the given class.
    """
    if logL == True:
        L = "logL"
        eL1 = "elogL1" 
        eL2 = "elogL2" 
    else:
        L = "L"
        eL1 = "eL1"
        eL2 = "eL2"

    data = pd.read_table(data_file, sep="\t", comment='#')
    # read data with errors
    data_MS = data[data['class'] == star_class]
    # select Main Sequence Stars

    df = data[
        ['R', 'eR1', 'eR2',
         'M', 'eM1', 'eM2',
         'Teff', 'eTeff1', 'eTeff2',
         'logg', 'elogg1', 'elogg2',    
         'FeH', 'eFeH1', 'eFeH2',
         L, eL1, eL2]
         ].copy()

    # clean NA values (simply remove the corresponding rows)
    df.dropna(inplace=True, axis=0)
    df_complete = data.loc[df.index].copy()

    return df_complete

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
          target_accept=0.95, max_treedepth=20):
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
                      nuts_sampler="nutpie",
                      )

    # Extract the learned mean training predictions directly from the trace
    sampled_mu_f = trace.posterior["mu_μ"].mean(dim=["chain", "draw"]).values
    print("Training predictions SD:", np.std(sampled_mu_f))
    print("Training predictions range:", sampled_mu_f.min(), sampled_mu_f.max())

    vs = ["ls", "ls_v", "eta", "eta_v"]
    print("Posteriors of interest:\n", az.summary(trace, var_names=vs))

    with pd.option_context("display.max_rows", None):
        df = az.summary(trace)
        df.sort_values(by="ess_bulk", inplace=True)
        print("AZ Stats for ESS Bulk < 400:\n", df[df["ess_bulk"]<400])

    r_hat_values = az.rhat(trace)
    all_rhats = []
    for var in r_hat_values.data_vars:
        max_rhat = r_hat_values[var].max().values.item()
        all_rhats.append((var, max_rhat))
    print("rhats: ", all_rhats)

    trace.extend(pm.compute_log_likelihood(trace, model=model, var_names='y')) 
    print("loo trace: ", az.loo(trace))

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
                       target:str, model:str, save_folder:str, hyperps=""):
    """Plot predictions of a trained model against true values
    Saves a preds figure and a residuals figure
    """
    plt.figure(figsize=(8, 6))
    plt.errorbar(y_true, y_pred, yerr=y_pred_err, fmt='o', label='Predictions with Uncertainty', alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('True '+target)
    plt.ylabel('Predicted '+target)
    plt.title(model+' Predictions with Uncertainty')
    plt.legend()
    plt.savefig(save_folder+"/"+model+target+"_preds_"+hyperps+".pdf")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.errorbar(y_true, y_pred - y_true, yerr=y_pred_err, fmt='o', label='Predictions with Uncertainty', alpha=0.7)
    plt.hlines(0, y_true.min(), y_true.max(), 'r', linestyle='--')
    plt.xlabel('True '+target)
    plt.ylabel('Residual '+target)
    plt.title(model+' Prediction Residuals')
    plt.legend()
    plt.savefig(save_folder+"/"+model+target+"_res_"+hyperps+".pdf")
    plt.close()
