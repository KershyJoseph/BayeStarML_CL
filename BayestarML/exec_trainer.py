#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 10:50:13 2025

@author: LamirelFamily
"""

from preprocess import return_train_test, prepare_pred4, denormalise_val, prepare_pred3
from utils import get_dataset, train, mard, mrd, model_pred_plotter
from models import hbnn, bart, gp
from pred_sampling import sample_post_pred_HBNN_para, posterior_predictive_GP, SIMPLE_sample_post_pred_HBNN_para
import arviz as az
import numpy as np
import pymc as pm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

import psutil
import time
from dataclasses import dataclass

@dataclass
class Dataset:
    x_train: pd.DataFrame
    x_train_er: pd.DataFrame
    x_test: pd.DataFrame
    x_test_err: pd.DataFrame

    mass_train: np.ndarray
    emass_train: np.ndarray
    mass_test: np.ndarray
    emass_test: np.ndarray

    rad_train: np.ndarray
    erad_train: np.ndarray
    rad_test: np.ndarray
    erad_test: np.ndarray

    unorm_mass: np.ndarray
    unorm_radius: np.ndarray

def mass_train_GP(data: Dataset, M_mean, M_var, outputs_folder , draw=1000, advi=False,
                  target_accept=.95, nutpie=False, sclass="MS"):
    """Function to train GP on mass prediction
    """
    hyperp_str = sclass+str(M_mean)+"_"+str(M_var)+"_"+str(draw)+"_"+str(target_accept)
    nuts_sampler = "pymc"
    if nutpie:
        hyperp_str += "NUTPIE"
        nuts_sampler = "nutpie"

    model, μ_gp, lg_σ_gp, μ_trace, var_trace, Xu, Xu_er = gp.sparse_fully_heteroscedastic_gp(data.x_train,
                                                                        data.x_train_er,
                                                                        data.mass_train,
                                                                        M_mean,
                                                                        M_var)

    if advi:
        approx = pm.fit(n=40000, method='advi', model=model, progressbar=True)
        trace = approx.sample(1000)
        print("ELBO:\n", approx.hist)
        trace.extend(pm.compute_log_likelihood(trace, model=model, var_names='y'))
        trace.to_netcdf(outputs_folder+"/Testing/GP_mass_testing/GPmass_ADVI_test"+hyperp_str+".nc")
    else:
        trace = train(model,
                  outputs_folder+"/GPmass/GPmass"+hyperp_str+".nc",
                  draw=draw, chains=4, target_accept=target_accept, nuts_sampler=nuts_sampler)

    pred, lpd = posterior_predictive_GP(model, μ_gp, lg_σ_gp, μ_trace, var_trace, trace,
                                        data.x_test, data.x_test_err, Xu, Xu_er, 4, 'Mass')

    stds = pred.std(0)
    means = pred.mean(0)
    print("means: ", means)
    print("stdvs: ", stds)
    print("Unorm mass: ", data.unorm_mass)

    print('MAE: ', mean_absolute_error(data.unorm_mass, means))
    print('MARD', mard(data.unorm_mass, means))
    print('MRD', mrd(data.unorm_mass, means))

    model_pred_plotter(data.unorm_mass, means, stds, 'Mass', 'GP', outputs_folder+'/GPmass', hyperp_str)

def radius_train_GP(data: Dataset, M_mean, M_var, outputs_folder , draw=1000, advi=False,
                    target_accept=.95, nutpie=False, sclass="MS"):
    """Function to train GP on radius prediction
    """
    hyperp_str = sclass+str(M_mean)+"_"+str(M_var)+"_"+str(draw)+"_"+str(target_accept)
    nuts_sampler = "pymc"
    if nutpie:
        hyperp_str += "NUTPIE"
        nuts_sampler = "nutpie"

    model, μ_gp, lg_σ_gp, μ_trace, var_trace, Xu, Xu_er = gp.sparse_fully_heteroscedastic_gp(
        data.x_train, data.x_train_er, data.rad_train, M_mean, M_var)

    if advi:
        approx = pm.fit(n=40000, method='advi', model=model, progressbar=True)
        trace = approx.sample(1000)
        print("ELBO:\n", approx.hist)

        trace.extend(pm.compute_log_likelihood(trace, model=model, var_names='y'))
        trace.to_netcdf(outputs_folder+"/Testing/GP_rad_testing/GP_ADVI_rad_"+hyperp_str+".nc")
    else:
        trace = train(model,
                  outputs_folder+"/GPrad/GPrad_"+hyperp_str+".nc",
                  draw=draw, chains=4,
                  target_accept=target_accept,
                  nuts_sampler=nuts_sampler)

    pred, lpd = posterior_predictive_GP(model, μ_gp, lg_σ_gp, μ_trace, var_trace, trace,
                                        data.x_test, data.x_test_err, Xu, Xu_er, 4, 'Radius')

    stds = pred.std(0)
    means = pred.mean(0)
    print("means: ", means)
    print("stdvs: ", stds)
    print(np.log10(np.log10(data.unorm_radius)))

    print('MAE: ', mean_absolute_error(np.log10(np.log10(data.unorm_radius)), means))
    print('MARD', mard(np.log10(np.log10(data.unorm_radius)), means))
    print('MRD', mrd(np.log10(np.log10(data.unorm_radius)), means))

    model_pred_plotter(np.log10(np.log10(data.unorm_radius)), means, stds, 'Radius', 'GP', outputs_folder+'/GPrad', hyperp_str)

def mass_train_SIMPLE_NN(data: Dataset, n_hidden, outputs_folder , draw=1000, chains=4,
                         target_accept=.95, nutpie=False, sclass="MS"):
    """
    ***Edit to only have one layer of 5 nodes***
    Function to train NN on mass prediction
    """
    #for output info
    hyperp_str = sclass+str(n_hidden)+"_"+str(draw)+"_"+str(chains)
    nuts_sampler = "pymc"
    if nutpie:
        hyperp_str += "NUTPIE"
        nuts_sampler = "nutpie"

    model = hbnn.HBNN_M4_simpler(data.x_train, mass_train, data.x_train_er, data.emass_train, n_hidden)
    model.debug(verbose=True)
    trace = train(model,
                  outputs_folder+"/NNmass/simpleNN_mass"+hyperp_str+".nc",
                  draw=draw, chains=chains,
                  target_accept=target_accept,
                  nuts_sampler=nuts_sampler)

    pred, lpd = SIMPLE_sample_post_pred_HBNN_para(trace, data.x_test, data.x_test_err, n_hidden, 4, "Mass")

    stds = pred.std(0)
    means = pred.mean(0)
    print("means: ", means)
    print("stdvs: ", stds)
    print("test set: ", data.unorm_mass)

    print('MAE: ', mean_absolute_error(data.unorm_mass, means))
    print('MARD', mard(data.unorm_mass, means))
    print('MRD', mrd(data.unorm_mass, means))

    model_pred_plotter(data.unorm_mass, means, stds, 'Mass', 'simpleNN', outputs_folder+'/NNmass', hyperp_str)

def mass_train_NN(data: Dataset, n_hidden, outputs_folder , draw=1000, chains=4,
                  target_accept=.95, nutpie=False, sclass="MS"):
    """Function to train NN on mass prediction
    """
    #for output info
    hyperp_str = sclass+str(n_hidden)+"_"+str(draw)+"_"+str(target_accept)
    nuts_sampler = "pymc"
    if nutpie:
        hyperp_str += "NUTPIE"
        nuts_sampler = "nutpie"

    model = hbnn.HBNN_M4(data.x_train, data.mass_train, data.x_train_er, data.emass_train, n_hidden)
    trace = train(model,
                  outputs_folder+"/NNmass/NNmass_"+hyperp_str+".nc",
                  draw=draw, chains=chains, target_accept=target_accept,
                  nuts_sampler=nuts_sampler)

    pred, lpd = sample_post_pred_HBNN_para(trace, data.x_test, data.x_test_err, n_hidden, 4, "Mass")

    stds = pred.std(0)
    means = pred.mean(0)
    print("means: ", means)
    print("stdvs: ", stds)
    with pd.option_context("display.max_rows", None):
        print("test set: ", data.unorm_mass)

    print('MAE: ', mean_absolute_error(data.unorm_mass, means))
    print('MARD', mard(data.unorm_mass, means))
    print('MRD', mrd(data.unorm_mass, means))

    model_pred_plotter(data.unorm_mass, means, stds, 'Mass', 'NN', outputs_folder+'/NNmass', hyperp_str)

def radius_train_NN(data: Dataset, n_hidden, outputs_folder , draw=1000, chains=4,
                    target_accept=.95, advi=False, nutpie=False, sclass="MS"):
    """Function to train NN on radius prediction
    """
    #for output info
    hyperp_str = sclass+str(n_hidden)+"_"+str(draw)
    nuts_sampler = "pymc"
    if nutpie:
        hyperp_str += "NUTPIE"
        nuts_sampler = "nutpie"

    model = hbnn.HBNN_M4(data.x_train, data.rad_train, data.x_train_er, data.erad_train, n_hidden)

    if advi:
        approx = pm.fit(n=100000, method='advi', model=model, progressbar=True)
        trace = approx.sample(1000)
        print("ELBO:\n", approx.hist)

        trace.extend(pm.compute_log_likelihood(trace, model=model, var_names='y'))
        trace.to_netcdf(outputs_folder+"/NN_rad_testing/NN_ADVI_rad_"+hyperp_str+".nc")
    else:
        trace = train(model,
                outputs_folder+"/NNrad/NNrad"+hyperp_str+".nc",
                draw=draw, chains=chains, target_accept=target_accept,
                nuts_sampler=nuts_sampler)

    pred, lpd = sample_post_pred_HBNN_para(trace, data.x_test, data.x_test_err, n_hidden, 4, "Radius")

    stds = pred.std(0)
    means = pred.mean(0)
    print("means: ", means)
    print("stdvs: ", stds)
    with pd.option_context("display.max_rows", None):
        print("test set: ", np.log10(data.unorm_radius))

    print('MAE: ', mean_absolute_error(np.log10(data.unorm_radius), means))
    print('MARD', mard(np.log10(data.unorm_radius), means))
    print('MRD', mrd(np.log10(data.unorm_radius), means))

    model_pred_plotter(np.log10(data.unorm_radius), means, stds, 'Radius', 'NN', outputs_folder+'/NNrad', hyperp_str)

if __name__ == '__main__':

    x_train = pd.read_csv("data/norm_train_data_700MS")
    x_test = pd.read_csv("data/norm_test_data_700MS")

    datasetMS = Dataset(
        x_train = x_train[['Teff', 'logg', 'FeH', 'logL']],
        x_train_er = x_train[['eTeff', 'elogg', 'eFeH', 'elogL']],
        x_test = x_test[['Teff', 'logg', 'FeH', 'logL']],
        x_test_err = x_test[['eTeff', 'elogg', 'eFeH', 'elogL']],

        rad_train=x_train["R"],
        erad_train=x_test["eR"],
        rad_test=x_test["R"],
        erad_test=x_test["eR"],

        mass_train=x_train["M"],
        emass_train=x_test["eM"],
        mass_test=x_test["M"],
        emass_test=x_test["eM"],

        unorm_mass = denormalise_val(x_test["M"], 'Mass'),
        unorm_radius = denormalise_val(x_test["R"], 'Radius')
        )

    df_train_RGB = get_dataset('DataExploring/good_RGB.txt', logL=True, logR=True)

    (x_trainRGB, x_train_erRGB, x_testRGB, x_test_errRGB, mass_trainRGB, emass_trainRGB,
    mass_testRGB, emass_testRGB, rad_trainRGB, erad_trainRGB, rad_testRGB, erad_testRGB
    ) = return_train_test(df_train_RGB, logL=True, logR=True)

    datasetRGB = Dataset(
        x_train = x_trainRGB[['Teff', 'logg', 'FeH', 'logL']],
        x_train_er = x_train_erRGB[['eTeff', 'elogg', 'eFeH', 'elogL']],

        x_test = x_testRGB[['Teff', 'logg', 'FeH', 'logL']],
        x_test_err = x_test_errRGB[['eTeff', 'elogg', 'eFeH', 'elogL']],

        rad_train=rad_trainRGB,
        erad_train=erad_trainRGB,
        rad_test=rad_testRGB,
        erad_test=erad_testRGB,

        mass_train=mass_trainRGB,
        emass_train=emass_trainRGB,
        mass_test=mass_testRGB,
        emass_test=emass_testRGB,

        unorm_mass = denormalise_val(mass_testRGB, 'Mass'),
        unorm_radius = denormalise_val(rad_testRGB, 'Radius')
        )

    #HAVE YOU UPDATED CONSTANTS.PY AND CHECKED OUTPUT FILE PATHS AND LOGL

    # print("''''''''''''''''''''''''\nNUTPIE GP TEST - MASS - PRIORS FROM NUTS + GP OBJS FIX\n......................")

    print("\n::::::::::::::::::::::::::::::::::::::")
    print("goodRGB5438")
    print("::::::::::::::::::::::::::::::::::::::\n")

    process = psutil.Process()
    # start_time_CPU = time.process_time()
    # start_time_wall = time.perf_counter()

    # print("NN mass NUTPIE - RGB stars. 3_2000. target_accept=0.95, TD 20.")
    # mass_train_NN(datasetRGB, 3, "Outputs5438RGB", 2000, nutpie=True, sclass="RGB")

    # end_time_CPU = time.process_time()
    mem1 = process.memory_info().rss / 1024**2
    # print(f"Peak Memory: {mem1:.2f} MB")
    # print(f"CPU time accumulated: {(end_time_CPU-start_time_CPU):.5f} s")
    # print(f"Total wall clock time: {time.perf_counter()-start_time_wall:.5f} s")

    # print("><><><><><><><><><><><><><><><><><><><><><><><><><><")

    start_time_CPU2 = time.process_time()
    start_time2 = time.time()

    print("NN mass - RGB stars. NUTPIE. With L in log space. 64, 1000, 4, target_accept=0.95. 20TD still.")
    mass_train_NN(datasetRGB, 64, "Outputs5438RGB", 1000, target_accept=0.95, nutpie=True, sclass="RGB")

    end_time_CPU2 = time.process_time()

    mem2 = process.memory_info().rss / 1024**2
    print(f"Peak Memory: {(mem2-mem1):.2f} MB")
    print(f"CPU time used: {(end_time_CPU2-start_time_CPU2):.5f} s")
    print(f"Total run time: {time.time()-start_time2:.5f} s")

    print("><><><><><><><><><><><><><><><><><><><><><><><><><><")

    # start_time_CPU3 = time.process_time()
    # start_time3 = time.time()

    # print("GP - radius - RGB stars. 100, 30, 1000. 20TD still.")
    # radius_train_GP(datasetRGB, 100, 30, "Outputs5438RGB", 1000, target_accept=0.95, sclass="RGB")

    # end_time_CPU3 = time.process_time()

    # mem3 = process.memory_info().rss / 1024**2
    # print(f"Peak Memory: {(mem3-mem2):.2f} MB")
    # print(f"CPU time used: {(end_time_CPU3-start_time_CPU3):.5f} s")
    # print(f"Total run time: {time.time()-start_time3:.5f} s")

    # print("><><><><><><><><><><><><><><><><><><><><><><><><><><")

    # start_time_CPU4 = time.process_time()
    # start_time4 = time.time()

    # print("GP - mass - RGB stars. 100, 30, 1000. 20TD still.")
    # mass_train_GP(datasetRGB, 100, 30, "Outputs5438RGB", 1000, target_accept=0.95, sclass="RGB")

    # end_time_CPU4 = time.process_time()

    # mem4 = process.memory_info().rss / 1024**2
    # print(f"Peak Memory: {(mem4-mem3):.2f} MB")
    # print(f"CPU time used: {(end_time_CPU4-start_time_CPU4):.5f} s")
    # print(f"Total run time: {time.time()-start_time4:.5f} s")

    # print("><><><><><><><><><><><><><><><><><><><><><><><><><")
    print("Salve Regina")














# The reason you are hitting PyMC warnings about "old GP objects" is because you've coded up a brilliant, custom implementation of a Sparse Heteroscedastic GP from scratch, rather than using PyMC's built-in pm.gp high-level classes.

# Because you aren't using their native pm.gp.Marginal or pm.gp.Latent wrapper classes to hold your models, PyMC's pm.sample_posterior_predictive doesn't inherently understand what your SparseLatent class is doing. When you reconstruct f_mu_pred and log_var_pred_latent inside the with gp_model: block during prediction, PyMC scans the context, notices that new random variables are being hooked up to older PyTensor computational graph fragments (μ_trace and var_trace), and panics with a warning thinking you've leaked an old model's state.

# Passing gp_trace around explicitly was a great instinct—and it works mathematically—but it triggers PyMC's internal guardrails because the graph is being modified dynamically after sampling.

# Here is the cleanest way to completely silence that warning and make your architecture robust, without completely rewriting your custom logic.

# The Fix: Decouple PPC from the Model Context
# When you use your own custom math for predictive conditionals, you don't actually need to inject those predictive variables back into the training gp_model context for sample_posterior_predictive.

# Instead, you can compile your conditional math directly into a clean, standalone PyTensor Function. This approach is faster, completely bypasses PyMC's graph tracking warnings, and treats your posterior predictive step as a pure mathematical mapping.

# Here is how you update your posterior_predictive_GP function:

# Python
# import pytensor

# def posterior_predictive_GP(
#     gp_model, mu_gp, log_var_gp, μ_trace, var_trace, trace,
#     X_new_raw, X_er_new_raw, Xu, Xu_var,
#     n_param, target,
#     var_cols_x=(0,1),         
#     var_cols_xerr=(0,1),     
#     random_seed=42,
# ):
#     lpd_GP = find_pointwise_loo(trace)
#     X_new_raw = np.asarray(X_new_raw, float)
#     X_er_new_raw = np.asarray(X_er_new_raw, float)
#     N_new = X_new_raw.shape[0]

#     # 1. We create a fresh, isolated model container just for compiling the prediction math
#     with pm.Model() as pred_model:
#         X_mu_obs = pm.Data("X_mu_obs", X_new_raw)
#         X_var_obs = pm.Data("X_var_obs", X_var_new_raw)

#         # Re-handle your missingness masks locally
#         mask_mu = ~np.isfinite(X_new_raw)
#         X_var_new_raw = np.hstack([X_new_raw[:, var_cols_x], X_er_new_raw[:, var_cols_xerr]])
#         mask_var = ~np.isfinite(X_var_new_raw)
#         D_var = X_var_new_raw.shape[1]

#         X_mu_latent = pm.Normal("X_new_latent", mu=0.0, sigma=1.0, shape=(N_new, n_param))
#         X_var_latent = pm.Normal("X_var_new_latent", mu=0.0, sigma=1.0, shape=(N_new, D_var))

#         X_new = tt.where(mask_mu, X_mu_latent, X_mu_obs)
#         X_var_new = tt.where(mask_var, X_var_latent, X_var_obs)

#         # Calculate math conditionals using your classes
#         f_mu_pred = mu_gp.conditional_marginal("f_mu_pred", X_new, Xu, gp_trace=μ_trace)
#         log_var_pred_latent = log_var_gp.conditional_marginal(
#             "log_var_pred_latent", X_var_new, Xu_var, gp_trace=var_trace
#         )
        
#         # Pull parameters safely via the original model graph names
#         log_var_pred = gp_model["alpha_log_var"] + log_var_pred_latent
#         sigma_pred = pm.math.exp(0.5 * log_var_pred)

#     # 2. Compile an explicit PyTensor function. 
#     # This takes your posterior sample inputs and maps them directly to your outputs.
#     # We find all free parameters required by the graph (ls, eta, alpha_log_var, etc.)
#     input_rvs = [v for v in pred_model.free_RVs if v not in [X_mu_latent, X_var_latent]]
#     # Add the parent variables from your training model that are required
#     input_rvs.extend([gp_model["alpha_log_var"]]) 

#     # Compile the mathematical graph directly
#     print("Compiling predictive math graph...")
#     predict_fn = pytensor.function(inputs=input_rvs, outputs=[f_mu_pred, sigma_pred])

#     # 3. Vectorized Evaluation over the Trace 
#     # Instead of calling sample_posterior_predictive, loop or map over your posteriors safely
#     posterior_samples = trace.posterior.stack(sample=("chain", "draw"))
    
#     y_draws_list = []
#     rng = np.random.default_rng(random_seed)

#     for i in range(len(posterior_samples.sample)):
#         sample = posterior_samples.isel(sample=i)
        
#         # Extract values from trace to pass to the compiled graph
#         feed_dict = {}
#         for r_var in input_rvs:
#             feed_dict[r_var] = sample[r_var.name].values

#         # Compute mu and sigma for this specific posterior draw
#         mu_val, sigma_val = predict_fn(**feed_dict)
        
#         # Sample your final observed y analytically
#         y_sample = rng.normal(loc=mu_val, scale=sigma_val)
#         y_draws_list.append(y_sample)

#     y_draws = np.array(y_draws_list) # Shape: (samples, N_test)

#     return denormalise_val(y_draws, target), lpd_GP
# Why this solves your issue completely
# No More Warning Contexts: By isolating the prediction step inside with pm.Model() as pred_model:, you prevent PyMC from scanning your original training context and seeing "mutated" or "redefined" GP operations.

# Deterministic Inputs: Compiling via pytensor.function converts your custom conditional logic directly into a fast C/JAX compiled execution loop.

# Bypasses the Bug entirely: PyMC's internal tracking triggers warning flags whenever custom classes reference external tensors. Converting it to an explicit function execution makes it immune to these namespace checks.