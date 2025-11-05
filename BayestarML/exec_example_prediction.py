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
    pred, w4 = predict4(X, X_er, 'radius', test=True)
    pred, w4 = predict4(X, X_er, 'mass')
    pred.to_csv("Results/NASAFLAG_8col_radius_res.csv")
    w4.to_csv("Results/NASAFLAG_8col_A_radius_w.csv")
    
    X3, X3_er = prepare_pred3("Datasets/dataset_B_trimmed_6cols_NASAFLAG.csv")
    pred3, w3 = predict3(X3, X3_er, 'radius')
    pred3.to_csv("Results/NASAFLAG_6col_radius_res.csv")
    w3.to_csv("Results/NASAFLAG_6col_radius_w.csv")
    

    
if __name__ == '__main__':
    main()