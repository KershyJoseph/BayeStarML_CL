#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 15:27:50 2025

@author: LamirelFamily
"""


import pymc as pm
import pymc_bart as pmb

def BART_M(X, X_er, Y, Y_er, m=250):
    """
    Build a BART model for predicting Y with input uncertainty in X.

    Constructs a PyMC model using Bayesian Additive Regression Trees (BART),
    where each input variable is treated as a normal random variable with
    mean `X` and standard deviation `X_er`. The model infers both the mean
    function and an uncertainty parameter for the target.

    Parameters
    ----------
    X : array-like
        Input features.
    X_er : array-like
        Measurement errors for each input feature.
    Y : array-like
        Observed target values.
    Y_er : array-like
        Measurement errors for Y (not used directly in the model).
    m : int, optional
        Number of trees in the BART ensemble. Default is 250.

    Returns
    -------
    pm.Model
        PyMC model defining the BART regression with learned noise parameter.
    """
    
    with pm.Model() as model_BART:
        
        X_in = pm.Data('X', X)
        X_in_er = pm.Data('X_er', X_er)
        
        X_normal = pm.Normal('X_dist', mu=X_in, sigma=X_in_er, shape=X_in.shape)
        
        mu = pmb.BART('mu', X_normal, Y.values, m=m)
             
        sig = pm.HalfCauchy('sig', beta=0.05)
        
        y = pm.Normal("y", mu=mu, sigma=sig, shape=X_in.shape[0], observed=Y)
        
    return model_BART


def BART_R(X, X_er, Y, Y_er, m=250):
    """
    Build a BART model with fixed output noise for predicting Y with input uncertainty.

    Similar to `BART_M`, but uses a fixed standard deviation (0.3) for the
    output noise instead of inferring it from the data.

    Parameters
    ----------
    X : array-like
        Input features.
    X_er : array-like
        Measurement errors for each input feature.
    Y : array-like
        Observed target values.
    Y_er : array-like
        Measurement errors for Y (not used directly in the model).
    m : int, optional
        Number of trees in the BART ensemble. Default is 250.

    Returns
    -------
    pm.Model
        PyMC model defining the BART regression with fixed noise parameter.
    """
    
    with pm.Model() as model_BART:
        
        X_in = pm.Data('X', X)
        X_in_er = pm.Data('X_er', X_er)
        
        X_normal = pm.Normal('X_dist', mu=X_in, sigma=X_in_er, shape=X_in.shape)
        
        mu = pmb.BART('mu', X_normal, Y.values, m=m)

        sig = 0.3
        
        y = pm.Normal("y", mu=mu, sigma=sig, shape=X_in.shape[0], observed=Y)
        
    return model_BART

