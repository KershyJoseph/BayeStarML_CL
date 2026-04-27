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

def get_dataset(data_file, star_class):
    """
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
    data = pd.read_table(data_file, sep="\t")
    # read data with errors
    data_MS = data[data['class'] == star_class]
    # select Main Sequence Stars
    df = data_MS[
        ['R', 'eR1', 'eR2',
         'M', 'eM1', 'eM2',
         'Teff', 'eTeff1', 'eTeff2',
         'logg', 'elogg1', 'elogg2',    
         'Fe/H', 'eFe/H1', 'eFe/H2',
         'L', 'eL1', 'eL2']
         ].copy()

    # clean NA values (simply remove the corresponding rows)
    df.dropna(inplace=True, axis=0)
    df_complete = data_MS.loc[df.index].copy()

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


def train(model, filename, draw=1000, chains=2, target_accept=0.95):
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
    trace = pm.sample(draws=draw, tune=draw, chains=chains, model=model,
                      target_accept=target_accept)
    trace.extend(pm.compute_log_likelihood(trace, model=model, var_names='y'))
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
    
    
    