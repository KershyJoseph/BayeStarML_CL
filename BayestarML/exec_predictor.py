#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 18:53:33 2025

@author: LamirelFamily
"""

from preprocess import prepare_pred4, prepare_pred3
from predict import predict3, predict4

def main():
    X, X_er = prepare_pred4("Datasets/dataset_A_trimmed_8cols_NASAFLAG.csv")
    # will need to change the traces in predict4 if you re-train
    # trace files aren't included in the github repo due to size
    _, bhs_pred, w4 = predict4(X, X_er, 'mass', test=True) # disregards X, X_er for test=True / uses test values
    _, pred, w4 = predict4(X, X_er, 'mass') # make predictions on new data (X, X_er)

if __name__ == '__main__':
    main()
