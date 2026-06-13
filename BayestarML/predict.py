#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 15:52:30 2025

@author: LamirelFamily
"""
from preprocess import load_data
from utils import mard, mrd
from models import bart, gp
from pred_sampling import sample_pred_BART, posterior_predictive_GP, sample_post_pred_HBNN_para
from BayestarML.models.bhs import run_stack
import arviz as az

def predict(X, X_er, target,
             training_dataset_key, GP_trace_path, NN_trace_path,
             BART_m, Mmean, Mvar, NNnodes,
             test=False):

    data, training_dim = load_data(training_dataset_key, target)

    if test == True:
        X = data["x_test"]
        X_er = data["x_test_err"]

    print("-------Start BART buisness----------")
    bart4_model = bart.BART_M(data["x_train"], data["x_train_err"], data["y_train"], data["y_train_err"], m=BART_m)
    bart4_pred, lpd_BART4 = sample_pred_BART(bart4_model,
                                    X,
                                    X_er, target,
                                    1000, 2)

    print("-------Start GP buisness----------")
    gp4_model, μ_gp4, lg_σ_gp4, μ_trace4, var_trace4, Xu4, Xu_er4 = gp.sparse_fully_heteroscedastic_gp(data["x_train"], data["x_train_err"], data["y_train"], Mmean, Mvar)
    gp4_trace = az.from_netcdf(GP_trace_path)
    gp4_pred, lpd_GP4 = posterior_predictive_GP(
        gp4_model, μ_gp4, lg_σ_gp4, μ_trace4, var_trace4, gp4_trace,
        X, X_er, Xu4, Xu_er4, training_dim, target)

    print("-------Start HBNN buisness----------")
    hbnn4_trace = az.from_netcdf(NN_trace_path)
    hbnn4_pred, lpd_HBNN4 = sample_post_pred_HBNN_para(hbnn4_trace,  
                                                    X,
                                                    X_er,
                                                    NNnodes, training_dim, target)

    print("-------Start BHS buisness----------")
    (bhs_trace, bhs_pred, bhs_w) = run_stack(bart4_pred, hbnn4_pred, gp4_pred,
                                        data["x_train"], X, lpd_BART4, lpd_HBNN4,
                                        lpd_GP4)
    bhs_trace.to_netcdf("/BHStrace_Mass_"+str(BART_m)+"_"+str(Mmean)+"_"+str(Mvar)+"_"+str(NNnodes)+".nc")

    if test == True:
        mard_BART = mard(data["unorm_y_test"], bart4_pred.mean(0))
        mrd_BART = mrd(data["unorm_y_test"], bart4_pred.mean(0))

        print('MARD BART:', mard_BART)
        print('MRD BART:', mrd_BART)

        mard_GP = mard(data["unorm_y_test"], gp4_pred.mean(0))
        mrd_GP = mrd(data["unorm_y_test"], gp4_pred.mean(0))

        print('MARD GP:', mard_GP)
        print('MRD GP:', mrd_GP)

        mard_HBNN = mard(data["unorm_y_test"], hbnn4_pred.mean(0))
        mrd_HBNN = mrd(data["unorm_y_test"], hbnn4_pred.mean(0))

        print('MARD HBNN:', mard_HBNN)
        print('MRD HBNN:', mrd_HBNN)

        mard_BHS = mard(data["unorm_y_test"], bhs_pred.mean(0))
        mrd_BHS = mrd(data["unorm_y_test"], bhs_pred.mean(0))

        print('MARD BHS:', mard_BHS)
        print('MRD BHS:', mrd_BHS)

        return [bart4_pred, gp4_pred, hbnn4_pred], bhs_pred, bhs_w, X, X_er, data["unorm_y_test"]

    else:
        return [bart4_pred, gp4_pred, hbnn4_pred], bhs_pred, bhs_w
