#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 10:50:13 2025

@author: LamirelFamily
"""

from preprocess import split_pro, return_train_test, prepare_pred4, denormalise_val, prepare_pred3
from utils import get_dataset, train, mard, mrd
from models import hbnn
from pred_sampling import sample_post_pred_HBNN_para, sample_post_pred_HBNN
import arviz as az
import numpy as np
import pymc as pm
import pandas as pd
from sklearn.metrics import mean_absolute_error

df_train = get_dataset('Datasets/data_sample_mass_radius.txt', 'MS')
(x_train, x_train_er, x_test, x_test_err, mass_train, emass_train,
  mass_test, emass_test, rad_train, erad_train, rad_test, erad_test
) = return_train_test(df_train)

unorm_rad = denormalise_val(rad_test, 'radius')

x_train3 = x_train[['Teff', 'logg', 'Meta']]
x_train3_er = x_train_er[['eTeff', 'elogg', 'eMeta']]

x_test3 = x_test[['Teff', 'logg', 'Meta']]
x_test3_er = x_test_err[['eTeff', 'elogg', 'eMeta']]

def main():
 
    model = hbnn.HBNN_R3(x_train3, rad_train, x_train3_er, erad_train, 15)
    
    trace = train(model, "Radius_output/HBNN_sig_015_15_nodes_3_param.nc", draw=1000, chains=4)
    # trace = az.from_netcdf("Radius_output/HBNN_sig_015_15_nodes_3_param.nc")
    
        
    trace.extend(pm.compute_log_likelihood(trace, model=model, var_names='y'))
    
    r_hat_values = az.rhat(trace)
    all_rhats = []
    for var in r_hat_values.data_vars:
        max_rhat = r_hat_values[var].max().values.item()
        all_rhats.append((var, max_rhat))

    print(all_rhats)
    
    print(az.loo(trace))
    
    pred, lpd = sample_post_pred_HBNN_para(trace, x_test3, x_test3_er, 15, 3, 'radius')
    
    print('MAE: ', mean_absolute_error(unorm_rad, pred[0]))
    
    print('MARD', mard(unorm_rad, pred[0]))
    
    print('MRD', mrd(unorm_rad, pred[0]))
    
if __name__ == '__main__':
    main()