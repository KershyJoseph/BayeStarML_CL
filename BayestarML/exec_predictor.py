#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 18:53:33 2025

@author: LamirelFamily
"""

from preprocess import prepare_pred4, prepare_pred3, denormalise_val, denormalise_err
from predict import predict3, predict4
from utils import mard, mrd
from sklearn.metrics import mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt

def bart_bhs_pred(target):
    """Train BART and BHS and then make some predictions
    """
    _, bhs_pred, bhs_w = predict4(X=None, X_er=None, target=target,
                                  training_dataset_path="DataExploring/good_MS.txt",
                                  GP_trace_path="Outputs/bigGPruns/GPrad_50_20_1000_0.99.nc",
                                  NN_trace_path="Outputs/bigNNruns/NNrad_goodMS_16_1000nrns.nc",
                                  BART_m = 300, Mmean=50, Mvar=20, NNnodes=16,
                                  test=True) # disregards X, X_er for test=True / uses test values

    X, X_er = prepare_pred4("Datasets/plato_data.txt")
    _, pred, w4 = predict4(X=X, X_er=X_er, target=target,
                           training_dataset_path="DataExploring/good_MS.txt",
                           GP_trace_path="Outputs/bigGPruns/GPrad_50_20_1000_0.99.nc",
                           NN_trace_path="Outputs/bigNNruns/NNrad_goodMS_16_1000nrns.nc",
                           BART_m = 300, Mmean=50, Mvar=20, NNnodes=16)

    df_p = pd.read_csv("Datasets/plato_data.txt", sep='\t')
    if target=='radius':
        t = "R"
    elif target=='mass':
        t = "M"
    else:
        raise ValueError("target should be string 'mass' or 'radius'")
    unorm_target = df_p[t]

    means = pred.mean(0)
    stds = pred.std(0)

    print('MAE on plato ', target,': ', mean_absolute_error(unorm_target, means))
    print('MARD on plato ', target,': ', mard(unorm_target, means))
    print('MRD on plato ', target,': ', mrd(unorm_target, means))

    plt.figure(figsize=(8, 6))
    plt.errorbar(unorm_target, means, yerr=stds, fmt='o', label='Predictions with Uncertainty', alpha=0.7)
    plt.plot([unorm_target.min(), unorm_target.max()], [unorm_target.min(), unorm_target.max()], 'r--')
    plt.xlabel('True ', target)
    plt.ylabel('Predicted ', target)
    plt.title('BHS Predictions with Uncertainty')
    plt.legend()
    plt.savefig("Outputs/predictions/BHS_plato_rad_preds.pdf")

    plt.figure(figsize=(8, 6))
    plt.errorbar(unorm_target, means - unorm_target, yerr=stds, fmt='o', label='Predictions with Uncertainty', alpha=0.7)
    plt.hlines(0, unorm_target.min(), unorm_target.max(), 'r', linestyle='--')
    plt.xlabel('True ', target)
    plt.ylabel('Residual ', target)
    plt.legend()
    plt.savefig("Outputs/predictions/BHS_plato_rad_res.pdf")

if __name__ == '__main__':
    bart_bhs_pred('radius')
