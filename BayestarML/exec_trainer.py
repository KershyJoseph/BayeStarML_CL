#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 10:50:13 2025

@author: LamirelFamily
"""

from preprocess import return_train_test, prepare_pred4, denormalise_val, prepare_pred3
from utils import get_dataset, train, mard, mrd
from models import hbnn, bart, gp
from pred_sampling import sample_post_pred_HBNN_para, posterior_predictive_GP
import arviz as az
import numpy as np
import pymc as pm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

df_train = get_dataset('DataExploring/MS_sample_31.txt', 'MS')

(x_train, x_train_er, x_test, x_test_err, mass_train, emass_train,
 mass_test, emass_test, rad_train, erad_train, rad_test, erad_test
) = return_train_test(df_train)

unorm_mass = denormalise_val(mass_test, 'mass')
unorm_mass = denormalise_val(rad_test, 'radius')

x_train = x_train[['Teff', 'logg', 'Fe/H', 'L']]
x_train_er = x_train_er[['eTeff', 'elogg', 'eFe/H', 'eL']]

x_test = x_test[['Teff', 'logg', 'Fe/H', 'L']]
x_test_er = x_test_err[['eTeff', 'elogg', 'eFe/H', 'eL']]

def mass_train_GP(M_mean, M_var):
    """Function to train GP on mass prediction
    """
    # model = hbnn.HBNN_R3(x_train3, rad_train, x_train3_er, erad_train, 15)
    model, μ_gp, lg_σ_gp, Xu, Xu_er = gp.sparse_fully_heteroscedastic_gp(x_train,
                                                                        x_train_er,
                                                                        mass_train, M_mean, M_var)

    trace = train(model,
                  "Output/GP_mass_full_w_int_lognorm_"+str(M_mean)+"_"+str(M_var)+".nc",
                  draw=100, chains=2)
    # trace = az.from_netcdf("Radius_output/GP_hetero_new_2026_mass_4param_gamma_etav_80_40.nc")

    # trace.extend(pm.compute_log_likelihood(trace, model=model, var_names='y'))

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
    print(unorm_mass)

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
    plt.savefig("Outputs/GP_mass_predictions.pdf")

    plt.figure(figsize=(8, 6))
    plt.errorbar(unorm_mass, pred.mean(0) - unorm_mass, yerr=pred.std(0), fmt='o', label='Predictions with Uncertainty', alpha=0.7)
    plt.hlines(0, unorm_mass.min(), unorm_mass.max(), 'r', linestyle='--')
    plt.xlabel('True Mass')
    plt.ylabel('Residual Mass')
    plt.legend()
    plt.savefig("Outputs/GP_mass_residuals.pdf")

def radius_train_GP(M_mean, M_var):
    """Function to train GP on radius prediction
    """
    # model = hbnn.HBNN_R3(x_train3, rad_train, x_train3_er, erad_train, 15)
    model, μ_gp, lg_σ_gp, Xu, Xu_er = gp.sparse_fully_heteroscedastic_gp(x_train,
                                                                        x_train_er,
                                                                        mass_train, M_mean, M_var)

    trace = train(model,
                  "Output/GP_mass_full_w_int_lognorm_"+str(M_mean)+"_"+str(M_var)+".nc",
                  draw=100, chains=2)
    # trace = az.from_netcdf("Radius_output/GP_hetero_new_2026_mass_4param_gamma_etav_80_40.nc")

    # trace.extend(pm.compute_log_likelihood(trace, model=model, var_names='y'))

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
    print(unorm_mass)

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
    plt.savefig("Outputs/GP_mass_predictions.pdf")

    plt.figure(figsize=(8, 6))
    plt.errorbar(unorm_mass, pred.mean(0) - unorm_mass, yerr=pred.std(0), fmt='o', label='Predictions with Uncertainty', alpha=0.7)
    plt.hlines(0, unorm_mass.min(), unorm_mass.max(), 'r', linestyle='--')
    plt.xlabel('True Mass')
    plt.ylabel('Residual Mass')
    plt.legend()
    plt.savefig("Outputs/GP_mass_residuals.pdf")


if __name__ == '__main__':
    