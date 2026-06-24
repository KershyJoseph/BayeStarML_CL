#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 15:52:30 2025

@author: LamirelFamily
"""
from BayestarML.src.models import bart, gp
from BayestarML.src.train_utils import load_data, mard, mrd, get_results
from BayestarML.src.predict_utils import prepare_pred_data, get_bhs_weights, plot_bhs_weights
from BayestarML.src.pred_sampling import sample_pred_bart, posterior_predictive_GP, sample_post_pred_HBNN_para, SIMPLE_sample_post_pred_HBNN_para
from BayestarML.src.models.bhs import run_stack
import arviz as az

def predict(x, x_er, interp_mask,
            target, training_dataset_key, 
            gp_trace_path, nn_trace_path,
            bart_m, m_mean, m_var, nn_nodes,
            y_compare=None, savename="", NN_1layer=False,
            bhs_trace=None, plot_density=False):
    """Train BART and BHS and then make some predictions
    """
    outputs_folder = "BayestarML/predict/outputs_bhs"
    data, training_dim = load_data(training_dataset_key, target)
    hyperp_str = savename+"_"+training_dataset_key+"_"+target+"_"+str(bart_m)+"_"+str(m_mean)+"_"+str(m_var)+"_"+str(nn_nodes)

    print("-------Start BART----------")
    bart4_model = bart.BART_M(data["x_train"], data["x_train_err"], data["y_train"], data["y_train_err"], m=bart_m)
    bart4_pred, lpd_BART4 = sample_pred_bart(bart4_model,
                                    x,
                                    x_er, target, training_dataset_key,
                                    1000, 2)

    print("-------Start GP----------")
    gp4_model, μ_gp4, lg_σ_gp4, μ_trace4, var_trace4, xu4, xu_er4 = gp.sparse_fully_heteroscedastic_gp(data["x_train"], data["x_train_err"], data["y_train"], m_mean, m_var)
    gp4_trace = az.from_netcdf(gp_trace_path)
    gp4_pred, lpd_GP4 = posterior_predictive_GP(
        gp4_model, μ_gp4, lg_σ_gp4, μ_trace4, var_trace4, gp4_trace,
        x, x_er, xu4, xu_er4, training_dim, target, training_dataset_key)

    print("-------Start HBNN----------")
    hbnn4_trace = az.from_netcdf(nn_trace_path)
    if NN_1layer:
        hbnn4_pred, lpd_HBNN4 = SIMPLE_sample_post_pred_HBNN_para(hbnn4_trace,  
                                                        x,
                                                        x_er,
                                                        nn_nodes, training_dim, target,
                                                        training_dataset_key)
    else:
        hbnn4_pred, lpd_HBNN4 = sample_post_pred_HBNN_para(hbnn4_trace,  
                                                        x,
                                                        x_er,
                                                        nn_nodes, training_dim, target,
                                                        training_dataset_key)

    print("-------Start BHS----------")
    (bhs_trace, bhs_pred, bhs_w) = run_stack(bart4_pred, hbnn4_pred, gp4_pred,
                                        data["x_train"], x, lpd_BART4, lpd_HBNN4,
                                        lpd_GP4)
    bhs_trace.to_netcdf(outputs_folder+"/"+hyperp_str+".nc")

    if y_compare: #get MARD and MRD for model vs y_compare values
        if y_compare == "test_set": 
            y_test = data["unorm_y_test"]
        else:
            y_test = y_compare

        mard_BART = mard(y_test, bart4_pred.mean(0))
        mrd_BART = mrd(y_test, bart4_pred.mean(0))

        print('MARD BART:', mard_BART)
        print('MRD BART:', mrd_BART)

        mard_GP = mard(y_test, gp4_pred.mean(0))
        mrd_GP = mrd(y_test, gp4_pred.mean(0))

        print('MARD GP:', mard_GP)
        print('MRD GP:', mrd_GP)

        mard_HBNN = mard(y_test, hbnn4_pred.mean(0))
        mrd_HBNN = mrd(y_test, hbnn4_pred.mean(0))

        print('MARD HBNN:', mard_HBNN)
        print('MRD HBNN:', mrd_HBNN)

        mard_BHS = mard(y_test, bhs_pred.mean(0))
        mrd_BHS = mrd(y_test, bhs_pred.mean(0))

        print('MARD BHS:', mard_BHS)
        print('MRD BHS:', mrd_BHS)

        get_results(bhs_pred, data, interp_mask, outputs_folder, training_dataset_key, target, hyperp_str, plot_density)

    return [bart4_pred, gp4_pred, hbnn4_pred], bhs_pred, bhs_w


if __name__ == "__main__":

    features = ["Teff", "FeH", "logL", "logg"]
    target = "M"
    dataset_key = "5336rgb"
    star_id, x, x_er, interp_mask = prepare_pred_data("test_set", dataset_key, features, target)

    _, pred, ws = predict(x, x_er, interp_mask, target, dataset_key,
        gp_trace_path = "BayestarML/train/outputs5438rgb/GP_M/GP_M_5336rgb_100_30_2000_0.95.nc",
        nn_trace_path = "BayestarML/train/outputs5336rgb/NN_M/NN_M_5336rgb_8_2000_0.95.nc",
        bart_m=700, m_mean=100, m_var=30, nn_nodes=8,
        y_compare="test_set", savename="BHS", plot_density=True)

    df_weights = get_bhs_weights(ws, star_id, dataset_key+"_bhs_"+target+"_ws.txt")

    plot_bhs_weights(df_weights, target, "BayestarML/data/"+dataset_key+".txt", dataset_key+"_bhs_"+target+"_ws_plot.pdf")

    # print("M BHS predictions:\n", pred.mean(0), pred.std(0))



















###########################################
#Trace paths

#700MS Mass
# gp_trace_path= "BayestarML/Outputs700MS/GP_M/GP_M_MS50_20_1000_0.95.nc",
# nn_trace_path= "BayestarML/train/outputs700ms/NN_M/NN_M_700ms_8_2000_0.95NUTPIE_init_mean.nc",

#700MS Radius
# gp_trace_path= "BayestarML/Outputs700MS/GP_R/GP_R_MS50_20_1000_0.95.nc",
# nn_trace_path= "BayestarML/Outputs700MS/NNrad/NNradMS16_2000NUTPIE.nc",

#488MS MASS
# gp_trace_path= "BayestarML/train/outputs488ms/GP_M/GP_M_488ms_20_10_1000_0.95.nc",
# nn_trace_path= "BayestarML/train/outputs488ms/NN_M/NN_M_488ms_16_1000_0.95.nc",

#5438RGB MASS
# gp_trace_path= "BayestarML/Outputs5438RGB/GPmass/GPmassRGB100_30_1000_0.95.nc",
# nn_trace_path= "BayestarML/train/outputs5438rgb/NN_M/NN_M_5438rgb_8_2000_0.95.nc",

#5336 RGB MASS 
# gp_trace_path = "BayestarML/train/outputs5438rgb/GP_M/GP_M_5336rgb_100_30_2000_0.95.nc",
# nn_trace_path = "BayestarML/train/outputs5336rgb/NN_M/NN_M_5336rgb_8_2000_0.95.nc",