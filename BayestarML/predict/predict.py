#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 15:52:30 2025

@author: LamirelFamily
"""
from BayestarML.src.models import bart, gp
from BayestarML.src.train_utils import load_data, mard, mrd, get_results
from BayestarML.src.data_processing_utils import prepare_pred_data
from BayestarML.src.pred_sampling import sample_pred_bart, posterior_predictive_GP, sample_post_pred_HBNN_para
from BayestarML.src.models.bhs import run_stack
import arviz as az

def predict(x, x_er, target,
            training_dataset_key, gp_trace_path, nn_trace_path,
            bart_m, m_mean, m_var, nn_nodes,
            test=False):
    """Train BART and BHS and then make some predictions
    """

    data, training_dim = load_data(training_dataset_key, target)

    if test:
        x = data["x_test"]
        x_er = data["x_test_err"]

    print("-------Start BART buisness----------")
    bart4_model = bart.BART_M(data["x_train"], data["x_train_err"], data["y_train"], data["y_train_err"], m=bart_m)
    bart4_pred, lpd_BART4 = sample_pred_bart(bart4_model,
                                    x,
                                    x_er, target,
                                    1000, 2)

    print("-------Start GP buisness----------")
    gp4_model, μ_gp4, lg_σ_gp4, μ_trace4, var_trace4, xu4, xu_er4 = gp.sparse_fully_heteroscedastic_gp(data["x_train"], data["x_train_err"], data["y_train"], m_mean, m_var)
    gp4_trace = az.from_netcdf(gp_trace_path)
    gp4_pred, lpd_GP4 = posterior_predictive_GP(
        gp4_model, μ_gp4, lg_σ_gp4, μ_trace4, var_trace4, gp4_trace,
        x, x_er, xu4, xu_er4, training_dim, target)

    print("-------Start HBNN buisness----------")
    hbnn4_trace = az.from_netcdf(nn_trace_path)
    hbnn4_pred, lpd_HBNN4 = sample_post_pred_HBNN_para(hbnn4_trace,  
                                                    x,
                                                    x_er,
                                                    nn_nodes, training_dim, target)

    print("-------Start BHS buisness----------")
    (bhs_trace, bhs_pred, bhs_w) = run_stack(bart4_pred, hbnn4_pred, gp4_pred,
                                        data["x_train"], x, lpd_BART4, lpd_HBNN4,
                                        lpd_GP4)
    bhs_trace.to_netcdf("/BHStrace_Mass_"+str(bart_m)+"_"+str(m_mean)+"_"+str(m_var)+"_"+str(nn_nodes)+".nc")

    if test:
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

        hyperp_str = training_dataset_key+"_"+str(bart_m)+"_"+str(m_mean)+"_"+str(m_var)+"_"+str(nn_nodes)
        get_results(pred, data, "BayestarML/predict/outputs_bhs", training_dataset_key, target, hyperp_str)

    return [bart4_pred, gp4_pred, hbnn4_pred], bhs_pred, bhs_w

if __name__ == "__main__":

    _, pred, _, data = predict(x=None, x_er=None,
                        target="M", training_dataset_key="700ms",
                        gp_trace_path= "BayestarML/Outputs700MS/GP_M/GP_M_MS50_20_1000_0.95.nc",
                        nn_trace_path= "BayestarML/Outputs700MS/NN_M/NN_M_MS16_1000_0.95NUTPIE.nc",
                        bart_m=200, m_mean=50, m_var=20, nn_nodes=16,
                        test=True)

