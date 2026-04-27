#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 15:52:30 2025

@author: LamirelFamily
"""

from preprocess import return_train_test, prepare_pred4, denormalise_val, prepare_pred3
from utils import get_dataset, mard, mrd
from models import bart, gp
from pred_sampling import sample_pred_BART, posterior_predictive_GP, sample_post_pred_HBNN_para
from bhs import run_stack
import arviz as az
import numpy as np
import pandas as pd
# import pymc as pm
# from sklearn.metrics import mean_absolute_error
# from utils import find_pointwise_loo


def predict4(X, X_er, target, test=False):

    df_train = get_dataset('Datasets/data_sample_mass_radius.txt', 'MS')
    (x_train, x_train_er, x_test, x_test_err, mass_train, emass_train,
      mass_test, emass_test, rad_train, erad_train, rad_test, erad_test
    ) = return_train_test(df_train)

    if test == True:
        X = x_test
        X_er = x_test_err

    if target == 'mass':
        
        unorm_mass = denormalise_val(mass_test, 'mass')

        
        bart4_model = bart.BART_M(x_train, x_train_er, mass_train, emass_train)
        bart4_pred, lpd_BART4 = sample_pred_BART(bart4_model,
                                      X,
                                      X_er, 'mass',
                                      1000, 2)


        gp4_model, μ_gp4, lg_σ_gp4, Xu4, Xu_er4 = gp.sparse_fully_heteroscedastic_gp(x_train, x_train_er, mass_train, 80, 40)

        gp4_trace = az.from_netcdf('models/model_artifacts/gp_mass_80_40.nc')
        gp4_pred, lpd_GP4 = posterior_predictive_GP(gp4_model, μ_gp4, lg_σ_gp4, 
                                            gp4_trace, X,
                                            X_er,
                                            Xu4, Xu_er4, 4, 'mass')


        hbnn4_trace = az.from_netcdf('models/model_artifacts/HBNN_mass.nc')
        hbnn4_pred, lpd_HBNN4 = sample_post_pred_HBNN_para(hbnn4_trace,  
                                                      X,
                                                      X_er,
                                                      15, 4, 'mass')


        (bhs_trace, bhs_pred, bhs_w) = run_stack(bart4_pred, hbnn4_pred, gp4_pred,
                                            x_train, X, lpd_BART4, lpd_HBNN4,
                                            lpd_GP4)

        if test == True:
            mard_BART = mard(unorm_mass, bart4_pred.mean(0))
            mrd_BART = mrd(unorm_mass, bart4_pred.mean(0))

            print('MARD BART:', mard_BART)
            print('MRD BART:', mrd_BART)

            mard_GP = mard(unorm_mass, gp4_pred.mean(0))
            mrd_GP = mrd(unorm_mass, gp4_pred.mean(0))

            print('MARD GP:', mard_GP)
            print('MRD GP:', mrd_GP)

            mard_HBNN = mard(unorm_mass, hbnn4_pred.mean(0))
            mrd_HBNN = mrd(unorm_mass, hbnn4_pred.mean(0))

            print('MARD HBNN:', mard_HBNN)
            print('MRD HBNN:', mrd_HBNN)

            mard_BHS = mard(unorm_mass, bhs_pred.mean(0))
            mrd_BHS = mrd(unorm_mass, bhs_pred.mean(0))

            print('MARD BHS:', mard_BHS)
            print('MRD BHS:', mrd_BHS)

        return [bart4_pred, gp4_pred, hbnn4_pred], bhs_pred, bhs_w

    if target == 'radius':

        unorm_rad = denormalise_val(rad_train, 'radius')

        bart4_model = bart.BART_R(x_train, x_train_er, rad_train, erad_train)
        bart4_pred, lpd_BART4 = sample_pred_BART(bart4_model,
                                      X,
                                      X_er, 'radius',
                                      1000,4)

        gp4_model, μ_gp4, lg_σ_gp4, Xu4, Xu_er4 = gp.sparse_fully_heteroscedastic_gp(x_train, x_train_er, rad_train, 80, 40)
        gp4_trace = az.from_netcdf('models/model_artifacts/gp_radius.nc') 
        gp4_pred, lpd_GP4 = posterior_predictive_GP(gp4_model, μ_gp4, lg_σ_gp4, 
                                            gp4_trace, X,
                                            X_er,
                                            Xu4, Xu_er4, 4, 'radius')

        hbnn4_trace = az.from_netcdf('models/model_artifacts/HBNN_sig_015_15_nodes_radius_4_param.nc')
        hbnn4_pred, lpd_HBNN4 = sample_post_pred_HBNN_para(hbnn4_trace,  
                                                      X,
                                                      X_er,
                                                      15, 4, 'radius')

        (bhs_trace, bhs_pred, bhs_w) = run_stack(bart4_pred, hbnn4_pred, gp4_pred,
                                            x_train, X, lpd_BART4, lpd_HBNN4,
                                            lpd_GP4)

        if test == True:
            mard_BART = mard(unorm_rad, bart4_pred.mean(0))
            mrd_BART = mrd(unorm_rad, bart4_pred.mean(0))

            print('MARD BART:', mard_BART)
            print('MRD BART:', mrd_BART)

            mard_GP = mard(unorm_rad, gp4_pred.mean(0))
            mrd_GP = mrd(unorm_rad, gp4_pred.mean(0))

            print('MARD GP:', mard_GP)
            print('MRD GP:', mrd_GP)

            mard_HBNN = mard(unorm_rad, hbnn4_pred.mean(0))
            mrd_HBNN = mrd(unorm_rad, hbnn4_pred.mean(0))

            print('MARD HBNN:', mard_HBNN)
            print('MRD HBNN:', mrd_HBNN)

            mard_BHS = mard(unorm_rad, bhs_pred.mean(0))
            mrd_BHS = mrd(unorm_rad, bhs_pred.mean(0))

            print('MARD BHS:', mard_BHS)
            print('MRD BHS:', mrd_BHS)

        return [bart4_pred, gp4_pred, hbnn4_pred], bhs_pred, bhs_w

def predictNAN(X, X_er, target, test=False):

    df_train = get_dataset('Datasets/data_sample_mass_radius.txt', 'MS')
    (x_train, x_train_er, x_test, x_test_err, mass_train, emass_train,
      mass_test, emass_test, rad_train, erad_train, rad_test, erad_test
    ) = return_train_test(df_train)
    
    if test == True:
        X = x_test
        X_er = x_test_err
        
    if target == 'mass':
        
        unorm_mass = denormalise_val(mass_test, 'mass')
        
        hbnn4_trace = az.from_netcdf('models/model_artifacts/HBNN_mass.nc')
        hbnn4_pred, lpd_HBNN4 = sample_post_pred_HBNN_para(hbnn4_trace,  
                                                      X,
                                                      X_er,
                                                      15, 4, 'mass')
        
        if test == True:
            mard_HBNN = mard(unorm_mass, hbnn4_pred.mean(0))
            mrd_HBNN = mrd(unorm_mass, hbnn4_pred.mean(0))
        
        return hbnn4_pred
    
    if target == 'radius':
        
        unorm_rad = denormalise_val(rad_test, 'radius')
        
        hbnn4_trace = az.from_netcdf('models/model_artifacts/HBNN_sig_015_15_nodes_radius_4_param.nc')
        hbnn4_pred, lpd_HBNN4 = sample_post_pred_HBNN_para(hbnn4_trace,  
                                                      X,
                                                      X_er,
                                                      15, 4, 'radius')  
        if test == True:
            
            mard_HBNN = mard(unorm_rad, hbnn4_pred.mean(0))
            mrd_HBNN = mrd(unorm_rad, hbnn4_pred.mean(0))
            
            print('MARD HBNN:', mard_HBNN)
            print('MRD HBNN:', mrd_HBNN)
            
        return hbnn4_pred

def predict3(X, X_er, target, test=False):
    
    df_train = get_dataset('Datasets/data_sample_mass_radius.txt', 'MS')
    (x_train, x_train_er, x_test, x_test_err, mass_train, emass_train,
      mass_test, emass_test, rad_train, erad_train, rad_test, erad_test
    ) = return_train_test(df_train)
    
    
    x_train3 = x_train[['Teff', 'logg', 'Fe/H']]
    x_train3_er = x_train_er[['eTeff', 'elogg', 'eFe/H']]
    
    if test == True:
        X = x_test[['Teff', 'logg', 'Fe/H']]
        X_er = x_test_err[['eTeff', 'elogg', 'eFe/H']]
    
    if target == 'mass':
        
        unorm_mass = denormalise_val(mass_test, 'mass')
        
        bart3_model = bart.BART_M(x_train3,
                                  x_train3_er,
                                  mass_train, emass_train)
        bart3_pred, lpd_BART3 = sample_pred_BART(bart3_model,
                                      X,
                                      X_er, 'mass',
                                      1000, 4)

        gp3_model, μ_gp3, lg_σ_gp3, Xu3, Xu_er3 = gp.sparse_fully_heteroscedastic_gp(x_train,
                                                                                      x_train_er, 
                                                                                      mass_train, 
                                                                                      80, 40)

        gp3_trace = az.from_netcdf('models/model_artifacts/gp_mass_3_param.nc')
        gp3_pred, lpd_GP3 = posterior_predictive_GP(gp3_model, μ_gp3, lg_σ_gp3, 
                                            gp3_trace, X,
                                            X_er,
                                            Xu3, Xu_er3, 3, 'mass')

        hbnn3_trace = az.from_netcdf('models/model_artifacts/HBNN_mass_3_param.nc')
        hbnn3_pred, lpd_HBNN3 = sample_post_pred_HBNN_para(hbnn3_trace,  
                                                      X,
                                                      X_er,
                                                      10, 3, 'mass')

        
        (bhs_trace, bhs_pred, bhs_w) = run_stack(bart3_pred, hbnn3_pred, gp3_pred,
                                            x_train3, X, lpd_BART3, lpd_HBNN3,
                                            lpd_GP3)
        
        if test == True:
            mard_BART = mard(unorm_mass, bart3_pred.mean(0))
            mrd_BART = mrd(unorm_mass, bart3_pred.mean(0))
            
            print('MARD BART:', mard_BART)
            print('MRD BART:', mrd_BART)
            
            mard_GP = mard(unorm_mass, gp3_pred.mean(0))
            mrd_GP = mrd(unorm_mass, gp3_pred.mean(0))
            
            print('MARD GP:', mard_GP)
            print('MRD GP:', mrd_GP)
            
            mard_HBNN = mard(unorm_mass, hbnn3_pred.mean(0))
            mrd_HBNN = mrd(unorm_mass, hbnn3_pred.mean(0))
            
            print('MARD HBNN:', mard_HBNN)
            print('MRD HBNN:', mrd_HBNN)
            
            mard_BHS = mard(unorm_mass, bhs_pred.mean(0))
            mrd_BHS = mrd(unorm_mass, bhs_pred.mean(0))
            
            print('MARD BHS:', mard_BHS)
            print('MRD BHS:', mrd_BHS)
        
        return [bart3_pred, gp3_pred, hbnn3_pred], bhs_pred, bhs_w
    
    if target == 'radius':
        
        unorm_rad = denormalise_val(rad_test, 'radius')
        
        bart3_model = bart.BART_R(x_train3,
                                  x_train3_er,
                                  rad_train, erad_train)
        
        bart3_pred, lpd_BART3 = sample_pred_BART(bart3_model,
                                      X,
                                      X_er, 'radius',
                                      1000, 4)

        gp3_model, μ_gp3, lg_σ_gp3, Xu3, Xu_er3 = gp.sparse_fully_heteroscedastic_gp(x_train3, 
                                                                                     x_train3_er,
                                                                                     rad_train, 80, 40)
        gp3_trace = az.from_netcdf('models/model_artifacts/gp_radius_3_param.nc')
        gp3_pred, lpd_GP3 = posterior_predictive_GP(gp3_model, μ_gp3, lg_σ_gp3, 
                                            gp3_trace, X,
                                            X_er,
                                            Xu3, Xu_er3, 3, 'radius')

        hbnn3_trace = az.from_netcdf('models/model_artifacts/HBNN_sig_015_15_nodes_radius_3_param.nc')
        hbnn3_pred, lpd_HBNN3 = sample_post_pred_HBNN_para(hbnn3_trace,  
                                                      X,
                                                      X_er,
                                                      15, 3, 'radius')


        (bhs_trace, bhs_pred, bhs_w) = run_stack(bart3_pred, hbnn3_pred, gp3_pred,
                                            x_train3, X, lpd_BART3, lpd_HBNN3,
                                            lpd_GP3)

        if test == True:
            mard_BART = mard(unorm_rad, bart3_pred.mean(0))
            mrd_BART = mrd(unorm_rad, bart3_pred.mean(0))

            print('MARD BART:', mard_BART)
            print('MRD BART:', mrd_BART)

            mard_GP = mard(unorm_rad, gp3_pred.mean(0))
            mrd_GP = mrd(unorm_rad, gp3_pred.mean(0))

            print('MARD GP:', mard_GP)
            print('MRD GP:', mrd_GP)

            mard_HBNN = mard(unorm_rad, hbnn3_pred.mean(0))
            mrd_HBNN = mrd(unorm_rad, hbnn3_pred.mean(0))

            print('MARD HBNN:', mard_HBNN)
            print('MRD HBNN:', mrd_HBNN)

            mard_BHS = mard(unorm_rad, bhs_pred.mean(0))
            mrd_BHS = mrd(unorm_rad, bhs_pred.mean(0))

            print('MARD BHS:', mard_BHS)
            print('MRD BHS:', mrd_BHS)

        return [bart3_pred, gp3_pred, hbnn3_pred], bhs_pred, bhs_w


def main():
    # X, X_er = prepare_pred4("Datasets/dataset_A_trimmed_8cols_NASAFLAG.csv")
    # # evaluate_missingness_grid(target='mass', save_path='mass_metrics_missing_params.csv')
    # # evaluate_missingness_grid(target='radius',
    # #                           gp_trace_path='models/model_artifacts/gp_radius.nc',
    # #                           hbnn_trace_path='models/model_artifacts/HBNN_sig_015_15_nodes_radius_4_param.nc',
    # #                           save_path='radius_metrics_missing_params.csv')
    # #print(X)
    # pred, w4 = predict4(X, X_er, 'radius')
    # pred.to_csv("Results/NASAFLAG_8col_radius_res.csv")
    # w4.to_csv("Results/NASAFLAG_8col_A_radius_w.csv")

    # X3, X3_er = prepare_pred3("Datasets/dataset_B_trimmed_6cols_NASAFLAG.csv")
    # pred3, w3 = predict3(X3, X3_er, 'radius')
    # pred3.to_csv("Results/NASAFLAG_6col_radius_res.csv")
    # w3.to_csv("Results/NASAFLAG_6col_radius_w.csv")

    X1, X1_er = prepare_pred4("Datasets/dataset_C_trimmed_8cols_NASAFLAG.csv")

    # pred1,_ = predict4(X1, X1_er, 'radius', test=True)
    # pred1.to_csv("Results/post_pred_bhs_rad.csv")
    # #w1.to_csv("Results/NASAFLAG_6col_dataC_radius_w.csv")

    pred2,_ = predict4(X1, X1_er, 'mass', test=True)
    pred2.to_csv("Results/post_pred_bhs_mass.csv")
    # w2.to_csv("Results/NASAFLAG_6col_dataC_mass_w.csv")
    # X2, X2_er = prepare_pred3("Datasets/ARIEL_level_0.csv")
    # print(X2)

    # pred, w3 = predict3(X2, X2_er, 'radius', test=True)
    # pred.to_csv("Results/Ariel_radius_pred_lvl0.csv")

    # pred, w3 = predict3(X2, X2_er, 'mass', test=True)
    # pred.to_csv("Results/Ariel_mass_pred_lvl0_check.csv")
    #w3.to_csv("Results/Ariel_radius_w_lvl0.csv")

    #pred, w4 = predictNAN(X, X_er, 'mass', test=False)
    #pred.to_csv("test_results_4_param_mass_w_hdi.csv")
    #print(pred)
    #w4.to_csv('weights_results_4_param_mass_w_hdi.csv')

if __name__ == '__main__':
    main()
