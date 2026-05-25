#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 10:50:13 2025

@author: LamirelFamily
"""

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

def mass_train_GP(data: Dataset, M_mean, M_var, draws=1000, advi=False, target_accept=.95):
    """Function to train GP on mass prediction
    """
    hyperp_str = "MS"+str(M_mean)+"_"+str(M_var)+"_"+str(draws)+"_"+str(target_accept)

    model, μ_gp, lg_σ_gp, Xu, Xu_er = gp.sparse_fully_heteroscedastic_gp(data.x_train,
                                                                        data.x_train_er,
                                                                        data.mass_train,
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
                  "Outputs/GPmass/GPmass"+hyperp_str+".nc",
                  draw=draws, chains=4, target_accept=target_accept, max_treedepth=20)

    r_hat_values = az.rhat(trace)
    all_rhats = []
    for var in r_hat_values.data_vars:
        max_rhat = r_hat_values[var].max().values.item()
        all_rhats.append((var, max_rhat))
    print(all_rhats)

    print("LOO")
    print(az.loo(trace))

    pred, lpd = posterior_predictive_GP(model, μ_gp, lg_σ_gp, trace,
                                        data.x_test, data.x_test_err, Xu, Xu_er, 4, 'mass')

    stds = pred.std(0)
    means = pred.mean(0)
    print(means)
    print(stds)
    print("Unorm mass: ", data.unorm_mass)

    # for i, std in enumerate(stds):
    #     if std > 1:
    #         print(f"Mass err bigger than 1Msol for star with mass {data.unorm_mass.iloc[i]}")
    #         print(f"(Predicted {means[i]} +/- {std} Msol)")
    #         print("------------------------")

    print('MAE: ', mean_absolute_error(data.unorm_mass, means))
    print('MARD', mard(data.unorm_mass, means))
    print('MRD', mrd(data.unorm_mass, means))

    plt.figure(figsize=(8, 6))
    plt.errorbar(data.unorm_mass, means, yerr=stds, fmt='o', label='Predictions with Uncertainty', alpha=0.7)
    plt.plot([data.unorm_mass.min(), data.unorm_mass.max()], [data.unorm_mass.min(), data.unorm_mass.max()], 'r--')
    plt.xlabel('True Mass')
    plt.ylabel('Predicted Mass')
    plt.title('GP Predictions with Uncertainty')
    plt.legend()
    plt.savefig("Outputs/GPmass/GPmass_preds"+hyperp_str+".pdf")

    plt.figure(figsize=(8, 6))
    plt.errorbar(data.unorm_mass, means - data.unorm_mass, yerr=stds, fmt='o', label='Predictions with Uncertainty', alpha=0.7)
    plt.hlines(0, data.unorm_mass.min(), data.unorm_mass.max(), 'r', linestyle='--')
    plt.xlabel('True Mass')
    plt.ylabel('Residual Mass')
    plt.legend()
    plt.savefig("Outputs/GPmass/GPmass_res"+hyperp_str+".pdf")

def radius_train_GP(data: Dataset, M_mean, M_var, draws=1000, advi=False, target_accept=.95):
    """Function to train GP on radius prediction
    """
    hyperp_str = "RGB"+str(M_mean)+"_"+str(M_var)+"_"+str(draws)+"_"+str(target_accept)

    model, μ_gp, lg_σ_gp, Xu, Xu_er = gp.sparse_fully_heteroscedastic_gp(data.x_train,
                                                                        data.x_train_er,
                                                                        data.rad_train, M_mean, M_var)

    if advi:
        approx = pm.fit(n=40000, method='advi', model=model, progressbar=True)
        trace = approx.sample(1000)
        print("ELBO:\n", approx.hist)

        trace.extend(pm.compute_log_likelihood(trace, model=model, var_names='y'))
        trace.to_netcdf("Outputs/Testing/GP_rad_testing/GP_ADVI_rad_"+hyperp_str+".nc")

    else:
        trace = train(model,
                  "Outputs/RGB/GPrad_"+hyperp_str+".nc",
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
                                        data.x_test, data.x_test_err, Xu, Xu_er, 4, 'radius')

    stds = pred.std(0)
    means = pred.mean(0)
    print(stds)
    print(means)
    print(data.unorm_radius)

    print('MAE: ', mean_absolute_error(data.unorm_radius, means))

    print('MARD', mard(data.unorm_radius, means))

    print('MRD', mrd(data.unorm_radius, means))

    plt.figure(figsize=(8, 6))
    plt.errorbar(data.unorm_radius, means, yerr=stds, fmt='o', label='Predictions with Uncertainty', alpha=0.7)
    plt.plot([data.unorm_radius.min(), data.unorm_radius.max()], [data.unorm_radius.min(), data.unorm_radius.max()], 'r--')
    plt.xlabel('True Radius')
    plt.ylabel('Predicted Radius')
    plt.title('GP Predictions with Uncertainty')
    plt.legend()
    plt.savefig("Outputs/RGB/GPrad_preds"+hyperp_str+".pdf")

    plt.figure(figsize=(8, 6))
    plt.errorbar(data.unorm_radius, means - data.unorm_radius, yerr=stds, fmt='o', label='Predictions with Uncertainty', alpha=0.7)
    plt.hlines(0, data.unorm_radius.min(), data.unorm_radius.max(), 'r', linestyle='--')
    plt.xlabel('True Mass')
    plt.ylabel('Residual Mass')
    plt.legend()
    plt.savefig("Outputs/RGB/GPrad_res"+hyperp_str+".pdf")

def mass_train_SIMPLE_NN(data: Dataset, n_hidden=5, draw=1000, chains=4, target_accept=.95):
    """
    ***Edit to only have one layer of 5 nodes***
    Function to train NN on mass prediction
    """
    #for output info
    string_specs = "goodRGB_"+str(n_hidden)+"_"+str(draw)+"_"+str(chains)

    model = hbnn.HBNN_M4_simpler(data.x_train, mass_train, data.x_train_er, data.emass_train, n_hidden)
    model.debug(verbose=True)
    trace = train(model,
                  "Outputs/RGB/NNsimple_mass_"+string_specs+".nc",
                  draw=draw, chains=chains,
                  target_accept=target_accept, max_treedepth=20)

    r_hat_values = az.rhat(trace)
    all_rhats = []
    for var in r_hat_values.data_vars:
        max_rhat = r_hat_values[var].max().values.item()
        all_rhats.append((var, max_rhat))

    print("rhats: ", all_rhats)

    print("loo trace: ", az.loo(trace))

    pred, lpd = SIMPLE_sample_post_pred_HBNN_para(trace, data.x_test, data.x_test_err, n_hidden, 4, "mass")

    stds = pred.std(0)
    means = pred.mean(0)
    print("stdvs: ", stds)
    print("means: ", means)
    print("test set: ", data.unorm_mass)

    print('MAE: ', mean_absolute_error(data.unorm_mass, means))

    print('MARD', mard(data.unorm_mass, means))

    print('MRD', mrd(data.unorm_mass, means))

    plt.figure(figsize=(8, 6))
    plt.errorbar(data.unorm_mass, means, yerr=stds, fmt='o', label='Predictions with Uncertainty', alpha=0.7)
    plt.plot([data.unorm_mass.min(), data.unorm_mass.max()], [data.unorm_mass.min(), data.unorm_mass.max()], 'r--')
    plt.xlabel('True Mass')
    plt.ylabel('Predicted Mass')
    plt.title('NN Predictions with Uncertainty')
    plt.legend()
    plt.savefig("Outputs/RGB/NNsimple_mass_preds_"+string_specs+".pdf")

    plt.figure(figsize=(8, 6))
    plt.errorbar(data.unorm_mass, means - data.unorm_mass, yerr=stds, fmt='o', label='Predictions with Uncertainty', alpha=0.7)
    plt.hlines(0, data.unorm_mass.min(), data.unorm_mass.max(), 'r', linestyle='--')
    plt.xlabel('True Mass')
    plt.ylabel('Residual Mass')
    plt.legend()
    plt.savefig("Outputs/RGB/NNsimple_mass_preds_"+string_specs+".pdf")

def mass_train_NN(data: Dataset, n_hidden=15, draw=1000, chains=4, target_accept=.95):
    """Function to train NN on mass prediction
    """
    #for output info
    string_specs = "RGB"+str(n_hidden)+"_"+str(draw)+"_"+str(target_accept)+"_20TD"

    model = hbnn.HBNN_M4(data.x_train, mass_train, data.x_train_er, data.emass_train, n_hidden)
    model.debug(verbose=True)
    trace = train(model,
                  "Outputs/RGB/NNmass_"+string_specs+"nrns.nc",
                  draw=draw, chains=chains, target_accept=target_accept,
                  max_treedepth=20)

    r_hat_values = az.rhat(trace)
    all_rhats = []
    for var in r_hat_values.data_vars:
        max_rhat = r_hat_values[var].max().values.item()
        all_rhats.append((var, max_rhat))

    print("rhats: ", all_rhats)

    print("loo trace: ", az.loo(trace))

    pred, lpd = sample_post_pred_HBNN_para(trace, data.x_test, data.x_test_err, n_hidden, 4, "mass")

    stds = pred.std(0)
    means = pred.mean(0)
    print("stdvs: ", stds)
    print("means: ", means)
    print("test set: ", data.unorm_mass)

    print('MAE: ', mean_absolute_error(data.unorm_mass, means))

    print('MARD', mard(data.unorm_mass, means))

    print('MRD', mrd(data.unorm_mass, means))

    plt.figure(figsize=(8, 6))
    plt.errorbar(data.unorm_mass, means, yerr=stds, fmt='o', label='Predictions with Uncertainty', alpha=0.7)
    plt.plot([data.unorm_mass.min(), data.unorm_mass.max()], [data.unorm_mass.min(), data.unorm_mass.max()], 'r--')
    plt.xlabel('True Mass')
    plt.ylabel('Predicted Mass')
    plt.title('NN Predictions with Uncertainty')
    plt.legend()
    plt.savefig("Outputs/RGB/NNmass_preds"+string_specs+".pdf")

    plt.figure(figsize=(8, 6))
    plt.errorbar(data.unorm_mass, means - data.unorm_mass, yerr=stds, fmt='o', label='Predictions with Uncertainty', alpha=0.7)
    plt.hlines(0, data.unorm_mass.min(), data.unorm_mass.max(), 'r', linestyle='--')
    plt.xlabel('True Mass')
    plt.ylabel('Residual Mass')
    plt.legend()
    plt.savefig("Outputs/RGB/NNmass_ress"+string_specs+".pdf")

def radius_train_NN(data: Dataset, n_hidden, draw=1000, chains=4, target_accept=.95, advi=False): 
    """Function to train NN on radius prediction
    """
    #for output info
    hyperp_str = "RGB"+str(n_hidden)+"_"+str(draw)#+"_"+str(chains)

    model = hbnn.HBNN_M4(data.x_train, data.rad_train, data.x_train_er, data.erad_train, n_hidden)

    if advi:
        approx = pm.fit(n=100000, method='advi', model=model, progressbar=True)
        trace = approx.sample(1000)
        print("ELBO:\n", approx.hist)

        trace.extend(pm.compute_log_likelihood(trace, model=model, var_names='y'))
        trace.to_netcdf("Outputs/NN_rad_testing/NN_ADVI_rad_"+hyperp_str+".nc")
    else:
        trace = train(model,
                "Outputs/RGB/NNrad"+hyperp_str+"nrns.nc",
                draw=draw, chains=chains, max_treedepth=20, target_accept=target_accept)

    r_hat_values = az.rhat(trace)
    all_rhats = []
    for var in r_hat_values.data_vars:
        max_rhat = r_hat_values[var].max().values.item()
        all_rhats.append((var, max_rhat))

    print("rhats: ", all_rhats)

    print("loo trace: ", az.loo(trace))

    pred, lpd = sample_post_pred_HBNN_para(trace, data.x_test, data.x_test_err, n_hidden, 4, "radius")

    stds = pred.std(0)
    means = pred.mean(0)
    print("stdvs: ", stds)
    print("means: ", means)
    print("test set: ", data.unorm_radius)

    print('MAE: ', mean_absolute_error(data.unorm_radius, means))

    print('MARD', mard(data.unorm_radius, means))

    print('MRD', mrd(data.unorm_radius, means))

    plt.figure(figsize=(8, 6))
    plt.errorbar(data.unorm_radius, means, yerr=stds, fmt='o', label='Predictions with Uncertainty', alpha=0.7)
    plt.plot([data.unorm_radius.min(), data.unorm_radius.max()], [data.unorm_radius.min(), data.unorm_radius.max()], 'r--')
    plt.xlabel('True Radius')
    plt.ylabel('Predicted Radius')
    plt.title('NN Predictions with Uncertainty')
    plt.legend()
    plt.savefig("Outputs/RGB/NNrad_preds"+hyperp_str+".pdf")

    plt.figure(figsize=(8, 6))
    plt.errorbar(data.unorm_radius, means - data.unorm_radius, yerr=stds, fmt='o', label='Predictions with Uncertainty', alpha=0.7)
    plt.hlines(0, data.unorm_radius.min(), data.unorm_radius.max(), 'r', linestyle='--')
    plt.xlabel('True Radius')
    plt.ylabel('Residual Radius')
    plt.legend()
    plt.savefig("Outputs/RGB/NNrad_res"+hyperp_str+".pdf")

if __name__ == '__main__':
    #pick which function(s) to run when file is run
    mp.set_start_method('spawn', force=True)

    #load data
    df_train = get_dataset('DataExploring/good_RGB.txt', logL=True)

    (x_train, x_train_er, x_test, x_test_err, mass_train, emass_train,
    mass_test, emass_test, rad_train, erad_train, rad_test, erad_test
    ) = return_train_test(df_train, logL=True)

    dataset = Dataset(
        x_train = x_train[['Teff', 'logg', 'Fe/H', 'logL']],
        x_train_er = x_train_er[['eTeff', 'elogg', 'eFe/H', 'elogL']],

        x_test = x_test[['Teff', 'logg', 'Fe/H', 'logL']],
        x_test_err = x_test_err[['eTeff', 'elogg', 'eFe/H', 'elogL']],

        rad_train=rad_train,
        erad_train=erad_train,
        rad_test=rad_test,
        erad_test=erad_test,

        mass_train=mass_train,
        emass_train=emass_train,
        mass_test=mass_test,
        emass_test=emass_test,

        unorm_mass = denormalise_val(mass_test, 'mass'),
        unorm_radius = denormalise_val(rad_test, 'radius')
        )

    #HAVE YOU UPDATED CONSTANTS.PY AND CHECKED OUTPUT FILE PATHS AND LOGL

    print("Latest goodMS with high mass filter.")
    print("::::::::::::::::::::::::::::::::::::::")

    process = psutil.Process()
    #tracemalloc.start() #for memory usage estimate
    #snapshot1 = tracemalloc.take_snapshot()
    start_time_CPU = time.process_time()
    start_time = time.time()

    print("bigGPrun - mass - MS stars. 50, 20, target_accept=0.99. TD 20 still.")
    mass_train_GP(dataset, 50, 20, 4, target_accept=0.99)

    end_time_CPU = time.process_time()
    #from Gemini
    # snapshot2 = tracemalloc.take_snapshot()
    # top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    # print("[ Top 5 memory changes ]")
    # for stat in top_stats[:5]:
    #     print(stat)

    mem1 = process.memory_info().rss / 1024**2
    print(f"Peak Memory: {mem1:.2f} MB")
    print(f"CPU time used: {(end_time_CPU-start_time_CPU):.5f} s")
    print(f"Total run time: {time.time()-start_time:.5f} s")

    print("><><><><><><><><><><><><><><><><><><><><><><><><><><")

    # start_time_CPU2 = time.process_time()
    # start_time2 = time.time()

    # print("bigNNrun - mass - RGB stars. With L in log space. On SimpleNN to start with. 8, 1000, 4, target_accept=0.95. 20TD still.")
    # # print("bigGPrun - radius - 80_40_1000_4 with 20TD, 0.99TA and hopefully improved priors.")
    # # print("(On good MS)")
    # mass_train_SIMPLE_NN(dataset, 8, 1000, target_accept=0.95)

    # end_time_CPU2 = time.process_time()

    # mem2 = process.memory_info().rss / 1024**2
    # print(f"Peak Memory: {(mem2-mem1):.2f} MB")
    # print(f"CPU time used: {(end_time_CPU2-start_time_CPU2):.5f} s")
    # print(f"Total run time: {time.time()-start_time2:.5f} s")

    # print("><><><><><><><><><><><><><><><><><><><><><><><><><><")

