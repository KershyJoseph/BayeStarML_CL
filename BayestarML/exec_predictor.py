#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 18:53:33 2025

@author: LamirelFamily
"""

from preprocess import prepare_pred4, prepare_pred3
from predict import predict3, predict4

def train_bart_bhs():
    #X, X_er = prepare_pred4("Datasets/dataset_A_trimmed_8cols_NASAFLAG.csv")

    _, bhs_pred, bhs_w = predict4(target='mass',
                                  dataset_path="DataExploring/good_MS.txt",
                                  test=True) # disregards X, X_er for test=True / uses test values
    #_, pred, w4 = predict4(X, X_er, 'mass') # make predictions on new data (X, X_er)

if __name__ == '__main__':
    train_bart_bhs
