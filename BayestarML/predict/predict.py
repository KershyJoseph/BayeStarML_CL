#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 15:52:30 2025

@author: LamirelFamily
"""
from BayestarML.src.models import bart, gp
from BayestarML.src.train_utils import load_data, mard, mrd, model_pred_plotter, get_ranges_point_metrics
from BayestarML.src.predict_utils import prepare_pred_data, get_bhs_weights, plot_bhs_weights
from BayestarML.src.pred_sampling import sample_pred_bart, posterior_predictive_GP, sample_post_pred_HBNN_para, SIMPLE_sample_post_pred_HBNN_para
from BayestarML.src.models.bhs import run_stack
from sklearn.metrics import mean_absolute_error
import arviz as az
import pandas as pd

def predict(x, x_er, interp_mask,
            target, training_dataset_key, 
            gp_trace_path, nn_trace_path,
            bart_m, m_mean, m_var, nn_nodes,
            y_compare=None, y_comp_er=None, savename="", NN_1layer=False,
            plot_density=False, color="test"):
    """Train BART and BHS and then make some predictions
    """
    outputs_folder = "BayestarML/predict/outputs_bhs"
    data, training_dim = load_data(training_dataset_key, target)
    hyperp_str = savename+"_"+training_dataset_key+"_"+target+"_"+str(bart_m)+"_"+str(m_mean)+"_"+str(m_var)+"_"+str(nn_nodes)

    print("-------Start BART----------")
    bart4_model = bart.BART(data["x_train"], data["x_train_err"], data["y_train"], data["y_train_err"], target=target, m=bart_m)
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

    if y_compare is not None: #get MARD and MRD for model vs y_compare values
        #interpolated vals
        y_compare = y_compare[interp_mask]

        mard_BART = mard(y_compare, bart4_pred.mean(0)[interp_mask])
        mrd_BART = mrd(y_compare, bart4_pred.mean(0)[interp_mask])
        mae_BART = mean_absolute_error(y_compare, bart4_pred.mean(0)[interp_mask])

        print("\nBART interp point metrics:")
        print('MAE BART:', mae_BART)
        print('MARD BART:', mard_BART)
        print('MRD BART:', mrd_BART)

        print("\nBART ranges:")
        get_ranges_point_metrics(bart4_pred.mean(0)[interp_mask], y_compare, target, [0.8, 1.4])
        get_ranges_point_metrics(bart4_pred.mean(0)[interp_mask], y_compare, target, [0.0, 0.625])

        mard_GP = mard(y_compare, gp4_pred.mean(0)[interp_mask])
        mrd_GP = mrd(y_compare, gp4_pred.mean(0)[interp_mask])
        mae_GP = mean_absolute_error(y_compare, gp4_pred.mean(0)[interp_mask])

        print("\nGP interp point metrics:")
        print('MAE GP:', mae_GP)
        print('MARD GP:', mard_GP)
        print('MRD GP:', mrd_GP)

        print("\nGP ranges:")
        get_ranges_point_metrics(gp4_pred.mean(0)[interp_mask], y_compare, target, [0.8, 1.4])
        get_ranges_point_metrics(gp4_pred.mean(0)[interp_mask], y_compare, target, [0.0, 0.625])

        mard_HBNN = mard(y_compare, hbnn4_pred.mean(0)[interp_mask])
        mrd_HBNN = mrd(y_compare, hbnn4_pred.mean(0)[interp_mask])
        mae_HBNN = mean_absolute_error(y_compare, hbnn4_pred.mean(0)[interp_mask])

        print("HBNN interp point metrics:")
        print('\nMAE HBNN:', mae_HBNN)
        print('MARD HBNN:', mard_HBNN)
        print('MRD HBNN:', mrd_HBNN)

        print("\nHBNN ranges:")
        get_ranges_point_metrics(hbnn4_pred.mean(0)[interp_mask], y_compare, target, [0.8, 1.4])
        get_ranges_point_metrics(hbnn4_pred.mean(0)[interp_mask], y_compare, target, [0.0, 0.625])

        mard_BHS = mard(y_compare, bhs_pred.mean(0)[interp_mask])
        mrd_BHS = mrd(y_compare, bhs_pred.mean(0)[interp_mask])
        mae_BHS = mean_absolute_error(y_compare, bhs_pred.mean(0)[interp_mask])

        print("BHS interp point metrics:")
        print('\nMAE BHS:', mae_BHS)
        print('MARD BHS:', mard_BHS)
        print('MRD BHS:', mrd_BHS)

        print("\nBHS ranges:")
        get_ranges_point_metrics(bhs_pred.mean(0)[interp_mask], y_compare, target, [0.8, 1.4])
        get_ranges_point_metrics(bhs_pred.mean(0)[interp_mask], y_compare, target, [0.0, 0.625])

        model_pred_plotter(y_compare, y_comp_er, bhs_pred.mean(0), bhs_pred.std(0), interp_mask, target, outputs_folder, hyperp_str, colour=color, plot_density=plot_density)

    return [bart4_pred, gp4_pred, hbnn4_pred], bhs_pred, bhs_w


def rgb_M_predder():

    features = ["Teff", "logg", "FeH", "L"]
    target = "M"
    dataset_key = "5336rgb"
    star_id, x, x_er, y_compare, y_comp_er, interp_mask = prepare_pred_data("RGB_pred_stars_Yu18.txt", dataset_key, features, target, add_log_vars=["L"], check_consistency=True, symmetric_errs=True)

    #only use interpolated values for Yu18 pred
    star_id, x, x_er, y_compare, y_comp_er = star_id[interp_mask], x[interp_mask], x_er[interp_mask], y_compare[interp_mask], y_comp_er[interp_mask]
    print(f"Only making predictions on the {len(x)} stars marked as feature interpolation.")

    interp_mask = interp_mask[interp_mask] #reset as well

    _, pred, ws = predict(x, x_er, interp_mask, target, dataset_key,
        gp_trace_path = "BayestarML/train/outputs5336rgb/GP_M/GP_M_5336rgb_100_30_2000_0.95.nc",
        nn_trace_path = "BayestarML/train/outputs5336rgb/NN_M/NN_M_5336rgb_8_2000_0.95.nc",
        bart_m=800, m_mean=100, m_var=30, nn_nodes=8,
        y_compare=y_compare, y_comp_er=y_comp_er, savename="BHS_YuMpred_", NN_1layer=False, plot_density=True, color="pred")

    df_weights = get_bhs_weights(ws, star_id, "Yu18_rgb_m_preds_ws.txt")

    plot_bhs_weights(df_weights, target, "BayestarML/predict/prediction_datasets/RGB_pred_stars_Yu18.txt", "Yu18_rgb_m_preds_ws_plot.pdf")

    # print("M BHS predictions:\n", pred.mean(0), pred.std(0))


def ms_R():
    features = ["Teff", "logg", "FeH", "logL"]
    target = "R"
    dataset_key = "693ms"
    star_id, x, x_er, y_compare, y_comp_er, interp_mask = prepare_pred_data("test_set", dataset_key, features, target)

    _, pred, ws = predict(x, x_er, interp_mask, target, dataset_key,
        gp_trace_path = "BayestarML/train/outputs693ms/GP_R/GP_R_693ms_50_15_2000_0.95.nc",
        nn_trace_path = "BayestarML/train/outputs693ms/NN_R/NN_1layer_R_693ms_8_2000_0.95.nc",
        bart_m=500, m_mean=50, m_var=15, nn_nodes=8,
        y_compare=y_compare, y_comp_er=y_comp_er, savename="BHS_", NN_1layer=True, color="test")

    df_weights = get_bhs_weights(ws, star_id, dataset_key+"_bhs_"+target+"_ws.txt")


def ms_M():
    features = ["Teff", "logg", "FeH", "logL"]
    target = "M"
    dataset_key = "693ms"
    star_id, x, x_er, y_compare, y_comp_er, interp_mask = prepare_pred_data("test_set", dataset_key, features, target)

    _, pred, ws = predict(x, x_er, interp_mask, target, dataset_key,
        gp_trace_path = "BayestarML/train/outputs693ms/GP_M/GP_M_693ms_50_15_1000_0.95.nc",
        nn_trace_path = "BayestarML/train/outputs693ms/NN_M/NN_M_693ms_8_2000_0.95.nc",
        bart_m=500, m_mean=50, m_var=15, nn_nodes=8,
        y_compare=y_compare, y_comp_er=y_comp_er, savename="BHS", NN_1layer=False, color="test")

    df_weights = get_bhs_weights(ws, star_id, dataset_key+"_bhs_"+target+"_ws.txt")



def rgb_M():
    features = ["Teff", "logg", "FeH", "L"]
    target = "M"
    dataset_key = "5336rgb"
    star_id, x, x_er, y_compare, y_comp_er, interp_mask = prepare_pred_data("test_set", dataset_key, features, target, add_log_vars=["L"])

    predict(x, x_er, interp_mask, target, dataset_key,
            "BayestarML/train/outputs5336rgb/GP_M/GP_M_5336rgb_100_30_2000_0.95.nc",
            "BayestarML/train/outputs5336rgb/NN_M/NN_M_5336rgb_8_2000_0.95.nc",
            800, 100, 30, 8, y_compare, y_comp_er, "bhs", plot_density=True)


def ms_M_predder():
    features = ["Teff", "logg", "FeH", "logL"]
    target = "M"
    dataset_key = "693ms"
    star_id, x, x_er, y_compare, y_comp_er, interp_mask = prepare_pred_data("NASAexop_archive_stars_all6.txt", dataset_key, features, target, check_consistency=True)

    _, pred, ws = predict(x, x_er, interp_mask, target, dataset_key,
        gp_trace_path = "BayestarML/train/outputs693ms/GP_M/GP_M_693ms_50_15_1000_0.95.nc",
        nn_trace_path = "BayestarML/train/outputs693ms/NN_M/NN_M_693ms_8_2000_0.95.nc",
        bart_m=500, m_mean=50, m_var=15, nn_nodes=8,
        y_compare=y_compare, y_comp_er=y_comp_er, savename="BHS_NExA_M", NN_1layer=False, color="pred", plot_density=True)

    bhs_preds = pd.DataFrame({
        "ID": star_id,
        "bhs_M_pred": pred.mean(0),
        "bhs_M_std": pred.std(0)
    })

    bhs_preds.to_csv("BayestarML/predict/outputs_bhs/bhs_NExA_M_preds.txt", index=False)


def ms_R_predder():
    features = ["Teff", "logg", "FeH", "logL"]
    target = "R"
    dataset_key = "693ms"
    star_id, x, x_er, y_compare, y_comp_er, interp_mask = prepare_pred_data("NASAexop_archive_stars_all6.txt", dataset_key, features, target, check_consistency=True)

    _, pred, _ = predict(x, x_er, interp_mask, target, dataset_key,
        gp_trace_path = "BayestarML/train/outputs693ms/GP_R/GP_R_693ms_50_15_2000_0.95.nc",
        nn_trace_path = "BayestarML/train/outputs693ms/NN_R/NN_1layer_R_693ms_8_2000_0.95.nc",
        bart_m=500, m_mean=50, m_var=15, nn_nodes=8,
        y_compare=y_compare, y_comp_er=y_comp_er, savename="BHS_NExA_R", NN_1layer=True, color="pred", plot_density=True)

    bhs_preds = pd.DataFrame({
        "ID": star_id,
        "bhs_R_pred": pred.mean(0),
        "bhs_R_std": pred.std(0)
    })

    bhs_preds.to_csv("BayestarML/predict/outputs_bhs/bhs_NExA_R_preds.txt", index=False)


if __name__ == '__main__':
    ms_M()
    print("^ That was ms M, bart 500")

    ms_R()
    print("^ That was ms R, bart 500")

    rgb_M()
    print("^ That was rgb M, bart 800")















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
# gp_trace_path = "BayestarML/train/outputs5336rgb/GP_M/GP_M_5336rgb_100_30_2000_0.95.nc",
# nn_trace_path = "BayestarML/train/outputs5336rgb/NN_M/NN_M_5336rgb_8_2000_0.95.nc",

#693MS MASS 
# gp_trace_path = "BayestarML/train/outputs693ms/GP_M/GP_M_693ms_50_15_1000_0.95.nc",
# nn_trace_path = "BayestarML/train/outputs693ms/NN_M/NN_M_693ms_8_2000_0.95.nc",

#693MS RADIUS
# gp_trace_path = "BayestarML/train/outputs693ms/GP_R/GP_R_693ms_50_15_2000_0.95.nc",
# nn_trace_path = "BayestarML/train/outputs693ms/NN_R/NN_1layer_R_693ms_8_2000_0.95.nc",