"""Executable to prepare a given database file for training on the BHS model
"""

import pandas as pd
import numpy as np
from preprocess import select_clean_data, error_filter, spreadomatic, return_train_test_edit, normalise, HRplot
from models.baselineXGB import studyrunner

def prep_data(filename:__path__,
              training_fs:list, targets:list,
              s_class:str,
              add_logvars:list,
              abs_err_lims:dict, percent_err_lims:dict,
              L_check:bool = False,
              plot_errs:bool = False,
              HRplot:bool = False,
              XGBoost:bool = False):
    """
    """
    df = pd.read_csv("data/"+filename, sep='\t', comment="#")
    df.set_index("ID", inplace=True)

    #if RGB, fix Teff errs: add 50K systematic error in quadrature to all Sch-St Teff errs, as taken from APOGEE https://iopscience.iop.org/article/10.3847/1538-4357/ac4891
    if s_class == "RGB":
        old_Teff = df.loc[df["source"]=="Schonhut-Stasik24", ["eTeff1", "eTeff2"]]
        df.loc[df["source"]=="Schonhut-Stasik24", ["eTeff1", "eTeff2"]] = np.sqrt(old_Teff**2 + 50**2)

    #select desired params and create avg symmetric err cols
    df = select_clean_data(df, training_fs, targets, s_class, add_logvars, L_check=L_check)
    #re-write training and targets in case of log switch
    for var in add_logvars:
        for list in [training_fs, targets]:
            if var in list:
                list.remove(var)
                list.append("log"+var)

    #filter by error limits and plot error distributions
    plot_params = None
    if plot_errs:
        plot_params = {
            "M": [7, "%"],
            "R": [7, "%"],
            "logL": [0.05, "dex"],
            "Teff": [100, "K"],
            "logg": [0.05, "dex"],
            "FeH": [0.1, "dex"] #0.1 for RGB
        }
    df = error_filter(df,
                    savename=s_class+"_err_dist.pdf",
                    abs_err_lims=abs_err_lims,
                    percent_err_lims=percent_err_lims,
                    plot_params=plot_params)

    #look at spread in target variables and decide whether or not to cut training range
    for var in targets:
        repeat=True
        spreadomatic(df, var)
        while repeat:
            lim0 = float(input(f"What lower limit do you want to put on {var}?\n"))
            lim1 = float(input(f"What upper limit do you want to put on {var}?\n"))
            df_copy = df[(df[var]>=lim0) & (df[var]<=lim1)]
            print(f"{len(df_copy)} stars left after trimming {var} range.")
            spreadomatic(df_copy, var)
            repeat = input(f"Continue checking {var} spread? (yes/no)\n") == "yes"
        df = df_copy
    print(f"{len(df)} stars left after cutting target ranges.")

    #get MUs and SIGs for normalisation of each param, and write to constants.json, as well as MIN and MAX
    X_train, X_test, Y_train, Y_test, = return_train_test_edit(df, training_fs, targets, s_class)
    final_set_name = str(len(df))+s_class

    #normalise data and create final txt files for normalised training and testing datasets
    X_train_norm, Y_train_norm = normalise(X_train, Y_train, "constants"+final_set_name+".json")
    X_test_norm, Y_test_norm = normalise(X_test, Y_test, "constants"+final_set_name+".json")

    #save files
    normalised_train_data = pd.concat([X_train_norm, Y_train_norm], axis=1)
    normalised_test_data = pd.concat([X_test_norm, Y_test_norm], axis=1)
    normalised_train_data.to_csv(f"data/norm_train_data_"+final_set_name+".txt")
    normalised_test_data.to_csv(f"data/norm_test_data_"+final_set_name+".txt")

    print("Normalised training and testing sets for "+final_set_name+" now saved in BayestarML/data.")

    #plot HR diagram of final data set (test and training) if desired
    if HRplot:
        print(len(df))
        HRplot(df, final_set_name+"_HRplot.pdf")

    #get XGBoost estimate on final data set if desired
    if XGBoost:
        for t in targets:
            print("\nStarting XGBoost benchmark on "+final_set_name+", target: "+t+"\n")
            X, y = df[training_fs], df[t]
            studyrunner(X, y, final_set_name+"_"+t+".json")
            print("For target "+t+"\n")

if __name__ == "__main__":
    #choose database to select data from for training
    filename = "datos_todos_v20261905.txt"
    #col headings should be 'col' for value and 'ecol1', 'ecol2' for corresponding errors

    training_fs = ["Teff", "logg", "FeH", "L"]
    targets = ["M", "R"]
    s_class = "RGB"
    add_logvars = ["L", "R"] #add a log column with errs for these variables

    abs_err_lims = {
        "elogL": 0.05,
        "eTeff": 100,
        "elogg": 0.05,
        "eFeH": 0.1
    }
    percent_err_lims = {
        "eM": 7,
        "eR": 7
    }

    #prepare data
    prep_data(filename,
              training_fs, targets,
              s_class,
              add_logvars,
              abs_err_lims, percent_err_lims,
              L_check=True, plot_errs=True, XGBoost=True)


# 700MS
# abs_err_lims = {
#     "elogL": 0.5
# }
# percent_err_lims = {
#     "eM": 100,
#     "eR": 100
# }