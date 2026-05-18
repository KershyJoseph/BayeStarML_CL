#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 10:50:13 2025

@author: LamirelFamily
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from preprocess import return_train_test, prepare_pred4, denormalise_val, prepare_pred3
from utils import get_dataset, train, mard, mrd
from models import hbnn, bart, gp
from pred_sampling import sample_post_pred_HBNN_para, posterior_predictive_GP, SIMPLE_sample_post_pred_HBNN_para
import arviz as az
import numpy as np
import pymc as pm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

import multiprocessing as mp
import tracemalloc
import psutil
import time

def mass_train_GP(M_mean, M_var, draws=1000, advi=False, target_accept=.95):
    """Function to train GP on mass prediction
    """
    hyperp_str = str(M_mean)+"_"+str(M_var)+"_"+str(draws)

    model, μ_gp, lg_σ_gp, Xu, Xu_er = gp.sparse_fully_heteroscedastic_gp(x_train,
                                                                        x_train_er,
                                                                        mass_train,
                                                                        M_mean,
                                                                        M_var)

    if advi:
        approx = pm.fit(n=40000, method='advi', model=model, progressbar=True)
        trace = approx.sample(1000)
        print("ELBO:\n", approx.hist)

        trace.extend(pm.compute_log_likelihood(trace, model=model, var_names='y'))
        trace.to_netcdf("Outputs/Testing/GP_mass_testing/GPmass_ADVI_test"+hyperp_str+".nc")

    else:
        trace = train(model,
                  "Outputs/GP_mass_full_w_int_lognorm_"+hyperp_str+".nc",
                  draw=draws, chains=4, target_accept=target_accept)

        r_hat_values = az.rhat(trace)
        all_rhats = []
        for var in r_hat_values.data_vars:
            max_rhat = r_hat_values[var].max().values.item()
            all_rhats.append((var, max_rhat))
        print(all_rhats)

    print(az.loo(trace))

    pred, lpd = posterior_predictive_GP(model, μ_gp, lg_σ_gp, trace,
                                        x_test, x_test_err, Xu, Xu_er, 4, 'mass')
    print(pred.std(0))
    print(pred.mean(0))
    print("Unorm mass: ", unorm_mass)

    for i, std in enumerate(pred.std(0)):
        if std > 1:
            print(f"Mass err bigger than 1Msol for star with mass {unorm_mass.iloc[i]}")
            print(f"(Predicted {pred.mean(0)[i]} +/- {std} Msol)")
            print("------------------------")

    print('MAE: ', mean_absolute_error(unorm_mass, pred.mean(0)))

    print('MARD', mard(unorm_mass, pred.mean(0)))

    print('MRD', mrd(unorm_mass, pred.mean(0)))

    plt.figure(figsize=(8, 6))
    plt.errorbar(unorm_mass, pred.mean(0), yerr=pred.std(0), fmt='o', label='Predictions with Uncertainty', alpha=0.7)
    plt.plot([unorm_mass.min(), unorm_mass.max()], [unorm_mass.min(), unorm_mass.max()], 'r--')
    plt.xlabel('True Mass')
    plt.ylabel('Predicted Mass')
    plt.title('GP Predictions with Uncertainty')
    plt.legend()
    plt.savefig("Outputs/Testing/GP_mass_testing/GPadvi_"+hyperp_str+"_mass_preds.pdf")

    plt.figure(figsize=(8, 6))
    plt.errorbar(unorm_mass, pred.mean(0) - unorm_mass, yerr=pred.std(0), fmt='o', label='Predictions with Uncertainty', alpha=0.7)
    plt.hlines(0, unorm_mass.min(), unorm_mass.max(), 'r', linestyle='--')
    plt.xlabel('True Mass')
    plt.ylabel('Residual Mass')
    plt.legend()
    plt.savefig("Outputs/Testing/GP_mass_testing/GPadvi_"+hyperp_str+"_mass_res.pdf")

def radius_train_GP(M_mean, M_var, draws=1000, advi=False, target_accept=.95):
    """Function to train GP on radius prediction
    """
    hyperp_str = str(M_mean)+"_"+str(M_var)+"_"+str(draws)

    model, μ_gp, lg_σ_gp, Xu, Xu_er = gp.sparse_fully_heteroscedastic_gp(x_train,
                                                                        x_train_er,
                                                                        rad_train, M_mean, M_var)

    if advi:
        approx = pm.fit(n=40000, method='advi', model=model, progressbar=True)
        trace = approx.sample(1000)
        print("ELBO:\n", approx.hist)

        trace.extend(pm.compute_log_likelihood(trace, model=model, var_names='y'))
        trace.to_netcdf("Outputs/Testing/GP_rad_testing/GP_ADVI_rad_"+hyperp_str+".nc")

    else:
        trace = train(model,
                  "Outputs/GP_radius_full_w_int_lognorm_"+hyperp_str+".nc",
                  draw=draws, chains=4, target_accept=target_accept,
                  max_treedepth=20)

    r_hat_values = az.rhat(trace)
    all_rhats = []
    for var in r_hat_values.data_vars:
        max_rhat = r_hat_values[var].max().values.item()
        all_rhats.append((var, max_rhat))

    print(all_rhats)

    print(az.loo(trace))

    pred, lpd = posterior_predictive_GP(model, μ_gp, lg_σ_gp, trace,
                                        x_test, x_test_err, Xu, Xu_er, 4, 'radius')
    print(pred.std(0))
    print(pred.mean(0))
    print(unorm_radius)

    print('MAE: ', mean_absolute_error(unorm_radius, pred.mean(0)))

    print('MARD', mard(unorm_radius, pred.mean(0)))

    print('MRD', mrd(unorm_radius, pred.mean(0)))

    plt.figure(figsize=(8, 6))
    plt.errorbar(unorm_radius, pred.mean(0), yerr=pred.std(0), fmt='o', label='Predictions with Uncertainty', alpha=0.7)
    plt.plot([unorm_radius.min(), unorm_radius.max()], [unorm_radius.min(), unorm_radius.max()], 'r--')
    plt.xlabel('True Radius')
    plt.ylabel('Predicted Radius')
    plt.title('GP Predictions with Uncertainty')
    plt.legend()
    plt.savefig("Outputs/GPrad_preds"+hyperp_str+".pdf")

    plt.figure(figsize=(8, 6))
    plt.errorbar(unorm_radius, pred.mean(0) - unorm_radius, yerr=pred.std(0), fmt='o', label='Predictions with Uncertainty', alpha=0.7)
    plt.hlines(0, unorm_radius.min(), unorm_radius.max(), 'r', linestyle='--')
    plt.xlabel('True Mass')
    plt.ylabel('Residual Mass')
    plt.legend()
    plt.savefig("Outputs/GPrad_residuals"+hyperp_str+".pdf")

def mass_train_SIMPLE_NN(n_hidden=15, draw=1000, chains=4, target_accept=.95):
    """
    ***Edit to only have one layer of 5 nodes***
    Function to train NN on mass prediction
    """
    #for output info
    string_specs = "_goodMS_"+str(n_hidden)+"_"+str(draw)+"_"+str(chains)

    model = hbnn.HBNN_M4_simpler(x_train, mass_train, x_train_er, emass_train, n_hidden)
    model.debug(verbose=True)
    trace = train(model,
                  "Outputs/MS/NN_mass_M4simpler"+string_specs+"_nrns.nc",
                  draw=draw, chains=chains, cores=chains, target_accept=target_accept)

    r_hat_values = az.rhat(trace)
    all_rhats = []
    for var in r_hat_values.data_vars:
        max_rhat = r_hat_values[var].max().values.item()
        all_rhats.append((var, max_rhat))

    print("rhats: ", all_rhats)

    print("loo trace: ", az.loo(trace))

    pred, lpd = SIMPLE_sample_post_pred_HBNN_para(trace, x_test, x_test_err, n_hidden, 4, "mass")

    print("stdvs: ", pred.std(0))
    print("means: ", pred.mean(0))
    print("test set: ", unorm_mass)

    print('MAE: ', mean_absolute_error(unorm_mass, pred.mean(0)))

    print('MARD', mard(unorm_mass, pred.mean(0)))

    print('MRD', mrd(unorm_mass, pred.mean(0)))

    plt.figure(figsize=(8, 6))
    plt.errorbar(unorm_mass, pred.mean(0), yerr=pred.std(0), fmt='o', label='Predictions with Uncertainty', alpha=0.7)
    plt.plot([unorm_mass.min(), unorm_mass.max()], [unorm_mass.min(), unorm_mass.max()], 'r--')
    plt.xlabel('True Mass')
    plt.ylabel('Predicted Mass')
    plt.title('NN Predictions with Uncertainty')
    plt.legend()
    plt.savefig("Outputs/MS/M4NNsimpler_mass_predictions"+string_specs+".pdf")

    plt.figure(figsize=(8, 6))
    plt.errorbar(unorm_mass, pred.mean(0) - unorm_mass, yerr=pred.std(0), fmt='o', label='Predictions with Uncertainty', alpha=0.7)
    plt.hlines(0, unorm_mass.min(), unorm_mass.max(), 'r', linestyle='--')
    plt.xlabel('True Mass')
    plt.ylabel('Residual Mass')
    plt.legend()
    plt.savefig("Outputs/MS/M4NNsimpler_mass_residuals"+string_specs+".pdf")

def mass_train_NN(n_hidden=15, draw=1000, chains=4, target_accept=.95):
    """Function to train NN on mass prediction
    """
    #for output info
    string_specs = "_goodMS_"+str(n_hidden)+"_"+str(draw)+"_"+str(chains)

    model = hbnn.HBNN_M4(x_train, mass_train, x_train_er, emass_train, n_hidden)
    model.debug(verbose=True)
    trace = train(model,
                  "Outputs/NN_final_mass_M4"+string_specs+"_nrns.nc",
                  draw=draw, chains=chains, target_accept=target_accept,
                  max_treedepth=20)

    r_hat_values = az.rhat(trace)
    all_rhats = []
    for var in r_hat_values.data_vars:
        max_rhat = r_hat_values[var].max().values.item()
        all_rhats.append((var, max_rhat))

    print("rhats: ", all_rhats)

    print("loo trace: ", az.loo(trace))

    pred, lpd = sample_post_pred_HBNN_para(trace, x_test, x_test_err, n_hidden, 4, "mass")

    print("stdvs: ", pred.std(0))
    print("means: ", pred.mean(0))
    print("test set: ", unorm_mass)

    print('MAE: ', mean_absolute_error(unorm_mass, pred.mean(0)))

    print('MARD', mard(unorm_mass, pred.mean(0)))

    print('MRD', mrd(unorm_mass, pred.mean(0)))

    plt.figure(figsize=(8, 6))
    plt.errorbar(unorm_mass, pred.mean(0), yerr=pred.std(0), fmt='o', label='Predictions with Uncertainty', alpha=0.7)
    plt.plot([unorm_mass.min(), unorm_mass.max()], [unorm_mass.min(), unorm_mass.max()], 'r--')
    plt.xlabel('True Mass')
    plt.ylabel('Predicted Mass')
    plt.title('NN Predictions with Uncertainty')
    plt.legend()
    plt.savefig("Outputs/NN_mass_final_preds"+string_specs+".pdf")

    plt.figure(figsize=(8, 6))
    plt.errorbar(unorm_mass, pred.mean(0) - unorm_mass, yerr=pred.std(0), fmt='o', label='Predictions with Uncertainty', alpha=0.7)
    plt.hlines(0, unorm_mass.min(), unorm_mass.max(), 'r', linestyle='--')
    plt.xlabel('True Mass')
    plt.ylabel('Residual Mass')
    plt.legend()
    plt.savefig("Outputs/NN_mass_final_ress"+string_specs+".pdf")

def radius_train_NN(n_hidden, draw=1000, chains=4, advi=False): 
    """Function to train NN on radius prediction
    """
    #for output info
    hyperp_str = "_goodMS_"+str(n_hidden)+"_"+str(draw)#+"_"+str(chains)

    model = hbnn.HBNN_M4(x_train, rad_train, x_train_er, erad_train, n_hidden)

    if advi:
        approx = pm.fit(n=100000, method='advi', model=model, progressbar=True)
        trace = approx.sample(1000)
        print("ELBO:\n", approx.hist)

        trace.extend(pm.compute_log_likelihood(trace, model=model, var_names='y'))
        trace.to_netcdf("Outputs/NN_rad_testing/NN_ADVI_rad_"+hyperp_str+".nc")
    else:
        trace = train(model,
                "Outputs/bigNNruns/NNrad"+hyperp_str+"nrns.nc",
                draw=draw, chains=chains, max_treedepth=20)

    r_hat_values = az.rhat(trace)
    all_rhats = []
    for var in r_hat_values.data_vars:
        max_rhat = r_hat_values[var].max().values.item()
        all_rhats.append((var, max_rhat))

    print("rhats: ", all_rhats)

    print("loo trace: ", az.loo(trace))

    pred, lpd = sample_post_pred_HBNN_para(trace, x_test, x_test_err, n_hidden, 4, "radius")

    print("stdvs: ", pred.std(0))
    print("means: ", pred.mean(0))
    print("test set: ", unorm_radius)

    print('MAE: ', mean_absolute_error(unorm_radius, pred.mean(0)))

    print('MARD', mard(unorm_radius, pred.mean(0)))

    print('MRD', mrd(unorm_radius, pred.mean(0)))

    plt.figure(figsize=(8, 6))
    plt.errorbar(unorm_radius, pred.mean(0), yerr=pred.std(0), fmt='o', label='Predictions with Uncertainty', alpha=0.7)
    plt.plot([unorm_radius.min(), unorm_radius.max()], [unorm_radius.min(), unorm_radius.max()], 'r--')
    plt.xlabel('True Radius')
    plt.ylabel('Predicted Radius')
    plt.title('NN Predictions with Uncertainty')
    plt.legend()
    plt.savefig("Outputs/bigNNruns/NN_rad_preds"+hyperp_str+".pdf")

    plt.figure(figsize=(8, 6))
    plt.errorbar(unorm_radius, pred.mean(0) - unorm_radius, yerr=pred.std(0), fmt='o', label='Predictions with Uncertainty', alpha=0.7)
    plt.hlines(0, unorm_radius.min(), unorm_radius.max(), 'r', linestyle='--')
    plt.xlabel('True Radius')
    plt.ylabel('Residual Radius')
    plt.legend()
    plt.savefig("Outputs/bigNNruns/NN_rad_res"+hyperp_str+".pdf")

if __name__ == '__main__':
    #pick which function(s) to run when file is run
    mp.set_start_method('spawn', force=True)

    # print("NN advi radius testing - all 2 layer with n=100,000")
    # trials = [4, 8, 16, 32, 64]
    # for t in trials:
    #     print("-------------")
    #     print(f"Trial with {t} nodes.")
    #     print("-------------")
    #     radius_train_NN(t, advi=True)

    #HAVE YOU UPDATED CONSTANTS.PY AND CHECKED OUTPUT FILE PATHS

    process = psutil.Process()
    tracemalloc.start() #for memory usage estimate
    snapshot1 = tracemalloc.take_snapshot()
    start_time = time.process_time()

    print("NN radius 4_1000. 1000 draws. 20TD. 1.5Tuning.")
    print("(On good MS)")
    # mass_train_GP(50,20,1000,target_accept=0.99)
    # radius_train_GP(80,40,1000,target_accept=.99)
    # mass_train_NN(64,2000,4,target_accept=.99)
    radius_train_NN(4, 1000, 4)

    end_time = time.process_time()
    #from Gemini
    snapshot2 = tracemalloc.take_snapshot()
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    print("[ Top 5 memory changes ]")
    for stat in top_stats[:5]:
        print(stat)

    print(f"Peak Memory: {process.memory_info().rss / 1024**2:.2f} MB")
    print(f"Training time: {(end_time-start_time):.5f} s")

