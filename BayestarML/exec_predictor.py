#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 18:53:33 2025

@author: LamirelFamily
"""

from preprocess import prepare_pred4, prepare_pred3, denormalise_val, denormalise_err
from predict import predict3, predict4
from utils import mard, mrd, model_pred_plotter
from sklearn.metrics import mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt

def bart_bhs_pred(target:str):
    """Train BART and BHS and then make some predictions
    """
    X, X_er = prepare_pred4("Datasets/plato_data.txt", logL=True)
    _, pred, w4 = predict4(X=X, X_er=X_er, target=target,
                           training_dataset_path="DataExploring/good_MS.txt",
                           GP_trace_path="Outputs700MS/GPmass/GPmassMS50_20_1000_0.95.nc",
                           NN_trace_path="Outputs700MS/NNmass/NNmass_MS16_2000_0.95.nc",
                           BART_m = 200, Mmean=50, Mvar=20, NNnodes=16)

    df_p = pd.read_csv("Datasets/plato_data.txt", sep='\t')
    if target=='Radius':
        t = "R"
    elif target=='Mass':
        t = "M"
    else:
        raise ValueError("target should be string 'Mass' or 'Radius'")
    unorm_target = df_p[t]

    means = pred.mean(0)
    stds = pred.std(0)

    print('MAE on plato ', target,': ', mean_absolute_error(unorm_target, means))
    print('MARD on plato ', target,': ', mard(unorm_target, means))
    print('MRD on plato ', target,': ', mrd(unorm_target, means))

    model_pred_plotter(unorm_target, means, stds, target, 'BHS', 'Outputs700MS/BHSmass', 'PLATO_trynoMH')


def bart_bhs_train(target):
    """Just train up BART and BHS and make preds on test set.
    Get figures of preds as funcs of params.
    """
    _, bhs_pred, bhs_w, X, Xer, y = predict4(X=None, X_er=None, target=target,
                                        training_dataset_path="DataExploring/good_MS.txt",
                                        GP_trace_path="Outputs707MS/GPmass/GPmassMS50_20_1000_0.95.nc",
                                        NN_trace_path="Outputs707MS/NNmass/NNmass_MS16_2000_0.95_20TDnrns.nc",
                                        BART_m = 200, Mmean=50, Mvar=20, NNnodes=16,
                                        test=True) # disregards X, X_er for test=True / uses test values

    target_ms = bhs_pred.mean(0)
    target_stds = bhs_pred.std(0)

    model_pred_plotter(y, target_ms, target_stds, target, 'BHS', 'Outputs707MS/BHS_'+target, 'train')

    plt.figure()
    plt.errorbar(X["logL"], target_ms, target_stds, fmt='o', alpha=0.5,
                 label="BHS "+target+" Predictions")
    plt.plot(X["logL"], y, 'x',
             label="True "+target)
    plt.xlabel("Luminosity (Lsol)")
    plt.ylabel(target+" ("+target[0]+"sol)")
    plt.title("Test set predictions")
    plt.legend()
    plt.savefig("Outputs707MS/BHS_"+target+"/bhs_L_M.pdf")
    plt.close()

    plt.figure()
    plt.errorbar(X["logg"], target_ms, target_stds, fmt='o', alpha=0.5,
                 label="BHS "+target+" Predictions")
    plt.plot(X["logg"], y, 'x',
             label="True "+target)
    plt.xlabel("Surface Gravity log(g) (dex)")
    plt.ylabel(target+" ("+target[0]+"sol)")
    plt.title("Test set predictions")
    plt.legend()
    plt.savefig("Outputs707MS/BHS_"+target+"/bhs_logg_M.pdf")
    plt.close()

if __name__ == '__main__':
    bart_bhs_pred('Mass')
