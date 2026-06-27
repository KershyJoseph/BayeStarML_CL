#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 15:27:50 2025

@author: LamirelFamily
"""


import pymc as pm
import pymc_bart as pmb

def BART(x, x_er, y, y_er, target, m=250):
    """
    Build a BART model for predicting y with input uncertainty in X.

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
    y : array-like
        Observed target values.
    y_er : array-like
        Measurement errors for y (not used directly in the model).
    target : str
        'M' or 'R' - for 'R' BART struggles to learn sigma from data so we set it to 0.2 for MS
    m : int, optional
        Number of trees in the BART ensemble. Default is 250.

    Returns
    -------
    pm.Model
        PyMC model defining the BART regression with learned noise parameter.
    """

    with pm.Model() as model_BART:

        x_in = pm.Data('x', x)
        x_in_er = pm.Data('x_er', x_er)

        x_normal = pm.Normal('x_dist', mu=x_in, sigma=x_in_er, shape=x_in.shape)

        mu = pmb.BART('mu', x_normal, y.values, m=m)

        if target == "M":
            sig = pm.HalfCauchy('sig', beta=0.05)
        elif target == "R":
            sig = 0.5

        y = pm.Normal("y", mu=mu, sigma=sig, shape=x_in.shape[0], observed=y)

    return model_BART
