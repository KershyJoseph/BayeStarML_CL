#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 14:49:36 2025

@author: LamirelFamily
"""

import numpy as np
import pymc as pm

def HBNN_M3(X_train, Y, X_error, Y_error, n_hidden):
    """
    Construct a 3-input hierarchical Bayesian neural network with correlated latent inputs.

    Builds a PyMC model where the observed inputs are modeled as noisy measurements 
    of latent variables drawn from a multivariate normal with an LKJ prior on the 
    correlation matrix. The latent variables feed into a two-layer neural network 
    with LeakyReLU activations. The output is modeled with a Student-t likelihood 
    to capture heavy-tailed residuals.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training inputs with 3 features.
    Y : array-like
        Observed target values.
    X_error : array-like
        Measurement uncertainties for each input feature.
    Y_error : array-like
        Measurement uncertainty for the target variable (not directly used).
    n_hidden : int
        Number of neurons in each hidden layer.

    Returns
    -------
    pm.Model
        PyMC model representing the hierarchical Bayesian neural network.
    """
    
    with pm.Model() as neural_network:
        
        # Data containers
        # X_data = pm.MutableData('X_data', X_train.values) 
        # sig_X = pm.MutableData('X_data_er', X_error.values)
        
        # Fixed LKJCholeskyCov specification
        chol, corr, sigmas = pm.LKJCholeskyCov(
            'Omega', 
            n=3, 
            eta=2,  
            sd_dist=pm.HalfNormal.dist(1), 
            compute_corr=True,
        )
        
        # Latent variables
        X_latent = pm.MvNormal(
            'X_latent', 
            mu=np.zeros(3),
            chol=chol,
            shape=(X_train.shape[0], 3)
        )
        
        # Observation model
        pm.Normal(
            "X_obs",
            mu=X_latent,
            sigma=X_error,
            observed=X_train
        )
        
        
        ann_input = pm.Deterministic('ann_input', X_latent)
        
        ann_output = pm.MutableData("ann_output" , Y)
        

        # Weights from input to hidden layer
        weights_in_1 = pm.Normal(
            "w_in_1", 0, sigma=0.1, shape=(n_hidden, X_latent.eval().shape[1])
        )

        
        weights_1_2 = pm.Normal(
            "w_1_2", 0, sigma=0.1, shape=(n_hidden,n_hidden)
        )


        # Weights from hidden layer to output
        weights_2_out = pm.Normal("w_2_out", 0, sigma=0.1, shape=(n_hidden))

        
        bias_1 = pm.Normal("bias_1", 0, sigma=1, shape=n_hidden)
        
        bias_2 = pm.Normal("bias_2", 0, sigma=1, shape=n_hidden)
        
        bias_out = pm.Normal("bias_out", 0, sigma=1)


        act_1 = pm.Deterministic('act_1', 
            pm.math.switch(pm.math.dot(ann_input, weights_in_1.T) + bias_1 > 0, 
                           pm.math.dot(ann_input, weights_in_1.T) + bias_1, 
                           0.01 * (pm.math.dot(ann_input, weights_in_1.T) + bias_1))
        )
        act_2 = pm.Deterministic('act_2', 
                                 pm.math.switch(pm.math.dot(act_1, weights_1_2) + bias_2 > 0,
                                                pm.math.dot(act_1, weights_1_2) + bias_2,
                                                0.01*(pm.math.dot(act_1, weights_1_2) + bias_2)))
        
        act_out = pm.Deterministic('act_out' , pm.math.dot(act_2, weights_2_out) + bias_out)
        
        er = pm.HalfCauchy('er', beta=1)
        
        nu = pm.HalfCauchy('nu', beta=3)


        out = pm.StudentT(
            "y",
            nu=nu,
            mu=act_out,
            sigma=er,
            observed=ann_output,
            shape=X_train.shape[0], 
        )

    return neural_network

def HBNN_M4(X_train, Y, X_error, Y_error, n_hidden):
    """
    Construct a 4-input hierarchical Bayesian neural network with correlated latent inputs.

    Similar to `HBNN_M3`, but extended to four input variables. Latent inputs are 
    drawn from a multivariate normal with an LKJCholeskyCov prior, capturing correlations 
    between features. The neural network uses two hidden layers with LeakyReLU activations 
    and a Student-t output likelihood for robustness to outliers.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training inputs with 4 features.
    Y : array-like
        Observed target values.
    X_error : array-like
        Measurement uncertainties for each input feature.
    Y_error : array-like
        Measurement uncertainty for the target variable (not directly used).
    n_hidden : int
        Number of neurons in each hidden layer.

    Returns
    -------
    pm.Model
        PyMC model defining the 4-input hierarchical Bayesian neural network.
    """
    
    with pm.Model() as neural_network:

        # X_data = pm.MutableData('X_data', X_train.values)  # Ensure numpy array
        # sig_X = pm.MutableData('X_data_er', X_error.values)
        
        # Fixed LKJCholeskyCov specification
        chol, corr, sigmas = pm.LKJCholeskyCov(
            'Omega', 
            n=4, 
            eta=2,  
            sd_dist=pm.HalfNormal.dist(1),  # Simpler prior
            compute_corr=True,
            #initval=Low_tri
        )
        
        # Latent variables
        X_latent = pm.MvNormal(
            'X_latent', 
            mu=np.zeros(4),
            chol=chol,
            shape=(X_train.shape[0], 4)
        )
        
        # Observation model
        pm.Normal(
            "X_obs",
            mu=X_latent,
            sigma=X_error,
            observed=X_train
        )
        
        
        ann_input = pm.Deterministic('ann_input', X_latent)
        
        ann_output = pm.MutableData("ann_output" , Y) #, dims="obs_id")
        

        # Weights from input to hidden layer
        weights_in_1 = pm.Normal(
            "w_in_1", 0, sigma=0.1, shape=(n_hidden, X_latent.eval().shape[1])
        )

        
        weights_1_2 = pm.Normal(
            "w_1_2", 0, sigma=0.1, shape=(n_hidden,n_hidden)
        )


        # Weights from hidden layer to output
        weights_2_out = pm.Normal("w_2_out", 0, sigma=0.1, shape=(n_hidden))

        
        bias_1 = pm.Normal("bias_1", 0, sigma=1, shape=n_hidden)
        
        bias_2 = pm.Normal("bias_2", 0, sigma=1, shape=n_hidden)
        
        bias_out = pm.Normal("bias_out", 0, sigma=1)


        act_1 = pm.Deterministic('act_1', 
            pm.math.switch(pm.math.dot(ann_input, weights_in_1.T) + bias_1 > 0, 
                           pm.math.dot(ann_input, weights_in_1.T) + bias_1, 
                           0.01 * (pm.math.dot(ann_input, weights_in_1.T) + bias_1))
        )
        act_2 = pm.Deterministic('act_2', pm.math.switch(pm.math.dot(act_1, weights_1_2) + bias_2 > 0,
                          pm.math.dot(act_1, weights_1_2) + bias_2,
                          0.01*(pm.math.dot(act_1, weights_1_2) + bias_2)))
        
        act_out = pm.Deterministic('act_out' , pm.math.dot(act_2, weights_2_out) + bias_out)
        
        er = pm.HalfCauchy('er', beta=1)

        out = pm.StudentT(
            "y",
            nu=5,
            mu=act_out,
            sigma=er,
            observed=ann_output,
            shape=X_train.shape[0], 
        )

    return neural_network

def HBNN_R4(X_train, Y, X_error, Y_error, n_hidden):
    """
    Construct a 4-input reduced Bayesian neural network with correlated latent inputs.

    Defines a simplified hierarchical model with four correlated latent inputs modeled 
    via an LKJCholeskyCov prior. The neural network structure mirrors `HBNN_M4`, but 
    assumes a fixed Gaussian output noise (σ = 0.15) instead of sampling it, reducing 
    model complexity and runtime.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training inputs with 4 features.
    Y : array-like
        Observed target values.
    X_error : array-like
        Measurement uncertainties for each input feature.
    Y_error : array-like
        Measurement uncertainty for the target variable (not directly used).
    n_hidden : int
        Number of neurons in each hidden layer.

    Returns
    -------
    pm.Model
        PyMC model defining the reduced 4-input Bayesian neural network.
    """
    
    with pm.Model() as neural_network:
        
        # Fixed LKJCholeskyCov specification
        chol, corr, sigmas = pm.LKJCholeskyCov(
            'Omega', 
            n=4, 
            eta=2,  
            sd_dist=pm.HalfNormal.dist(1),  # Simpler prior
            compute_corr=True,
            #initval=Low_tri
        )
        
        # Latent variables
        X_latent = pm.MvNormal(
            'X_latent', 
            mu=np.zeros(4),
            chol=chol,
            shape=(X_train.shape[0], 4)
        )
        
        # Observation model
        pm.Normal(
            "X_obs",
            mu=X_latent,
            sigma=X_error.values,
            observed=X_train.values
        )
        
        
        ann_input = pm.Deterministic('ann_input', X_latent)
        
        ann_output = pm.MutableData("ann_output" , Y) #, dims="obs_id")
        

        # Weights from input to hidden layer
        weights_in_1 = pm.Normal(
            "w_in_1", 0, sigma=0.1, shape=(n_hidden, X_latent.eval().shape[1])
        )

        
        weights_1_2 = pm.Normal(
            "w_1_2", 0, sigma=0.1, shape=(n_hidden,n_hidden)
        )


        # Weights from hidden layer to output
        weights_2_out = pm.Normal("w_2_out", 0, sigma=0.1, shape=(n_hidden))

        
        bias_1 = pm.Normal("bias_1", 0, sigma=1, shape=n_hidden)
        
        bias_2 = pm.Normal("bias_2", 0, sigma=1, shape=n_hidden)
        
        bias_out = pm.Normal("bias_out", 0, sigma=1)


        act_1 = pm.Deterministic('act_1', 
            pm.math.switch(pm.math.dot(ann_input, weights_in_1.T) + bias_1 > 0, 
                           pm.math.dot(ann_input, weights_in_1.T) + bias_1, 
                           0.01 * (pm.math.dot(ann_input, weights_in_1.T) + bias_1))
        )
        act_2 = pm.Deterministic('act_2', pm.math.switch(pm.math.dot(act_1, weights_1_2) + bias_2 > 0,
                          pm.math.dot(act_1, weights_1_2) + bias_2,
                          0.01*(pm.math.dot(act_1, weights_1_2) + bias_2)))
        
        act_out = pm.Deterministic('act_out' , pm.math.dot(act_2, weights_2_out) + bias_out)
        
        
        #er = pm.HalfCauchy('er', beta=1) + er_data
        er = 0.15

        
        out = pm.Normal('y', mu=act_out, sigma=er, observed=ann_output, 
                        shape=X_train.shape[0])
        
    return neural_network

def HBNN_R3(X_train, Y, X_error, Y_error, n_hidden):
    """
    Construct a 3-input reduced Bayesian neural network with correlated latent inputs.

    A computationally lighter version of `HBNN_M3` using fixed Gaussian noise (σ = 0.15)
    in the output layer. The latent input correlations are captured via an LKJ prior,
    and the network consists of two hidden layers with LeakyReLU activations.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training inputs with 3 features.
    Y : array-like
        Observed target values.
    X_error : array-like
        Measurement uncertainties for each input feature.
    Y_error : array-like
        Measurement uncertainty for the target variable (not directly used).
    n_hidden : int
        Number of neurons in each hidden layer.

    Returns
    -------
    pm.Model
        PyMC model defining the reduced 3-input Bayesian neural network.
    """
    with pm.Model() as neural_network:
        
        # Fixed LKJCholeskyCov specification
        chol, corr, sigmas = pm.LKJCholeskyCov(
            'Omega', 
            n=3, 
            eta=2,  
            sd_dist=pm.HalfNormal.dist(1), 
            compute_corr=True,
            #initval=Low_tri
        )
        
        # Latent variables
        X_latent = pm.MvNormal(
            'X_latent', 
            mu=np.zeros(3),
            chol=chol,
            shape=(X_train.shape[0], 3)
        )
        
        # Observation model
        pm.Normal(
            "X_obs",
            mu=X_latent,
            sigma=X_error.values,
            observed=X_train.values
        )
        
        
        ann_input = pm.Deterministic('ann_input', X_latent)
        
        ann_output = pm.MutableData("ann_output" , Y)
        

        # Weights from input to hidden layer
        weights_in_1 = pm.Normal(
            "w_in_1", 0, sigma=0.1, shape=(n_hidden, X_latent.eval().shape[1])
        )

        
        weights_1_2 = pm.Normal(
            "w_1_2", 0, sigma=0.1, shape=(n_hidden,n_hidden)
        )


        # Weights from hidden layer to output
        weights_2_out = pm.Normal("w_2_out", 0, sigma=0.1, shape=(n_hidden))

        
        bias_1 = pm.Normal("bias_1", 0, sigma=1, shape=n_hidden)
        
        bias_2 = pm.Normal("bias_2", 0, sigma=1, shape=n_hidden)
        
        bias_out = pm.Normal("bias_out", 0, sigma=1)


        act_1 = pm.Deterministic('act_1', 
            pm.math.switch(pm.math.dot(ann_input, weights_in_1.T) + bias_1 > 0, 
                           pm.math.dot(ann_input, weights_in_1.T) + bias_1, 
                           0.01 * (pm.math.dot(ann_input, weights_in_1.T) + bias_1))
        )
        act_2 = pm.Deterministic('act_2', 
                                 pm.math.switch(pm.math.dot(act_1, weights_1_2) + bias_2 > 0,
                                                pm.math.dot(act_1, weights_1_2) + bias_2,
                                                0.01*(pm.math.dot(act_1, weights_1_2) + bias_2)))
        
        act_out = pm.Deterministic('act_out' , pm.math.dot(act_2, weights_2_out) + bias_out)
        
        er = 0.15
        
        out = pm.Normal('y', mu=act_out, sigma=er, observed=ann_output, 
                        shape=X_train.shape[0])

    return neural_network


