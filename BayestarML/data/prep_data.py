"""Executable to prepare a given database file for training on the BHS model"""

import numpy as np
import pandas as pd
from preprocess import (
    HRplot,
    error_filter,
    normalise,
    plot_feature_target,
    return_train_test,
    select_clean_data,
    spreadomatic,
)

from BayestarML.src.models.baselinexgb import studyrunner


def prep_data(
    filename: str,
    training_fs: list,
    targets: list,
    s_class: str,
    add_logvars: list,
    abs_err_lims: dict,
    percent_err_lims: dict,
    target_lims: dict = None,
    L_check: bool = False,
    plot_errs: bool = False,
    plotHR: bool = False,
    plot_t_f=False,
    xgboost: bool = False,
):
    """ """
    df = pd.read_csv("BayestarML/data/" + filename, sep="\t", comment="#")
    df.set_index("ID", inplace=True)

    # if RGB, fix Teff errs: add 50K systematic error in quadrature to all Sch-St Teff errs, as were taken straight from APOGEE https://iopscience.iop.org/article/10.3847/1538-4357/ac4891
    if s_class == "RGB":
        old_Teff = df.loc[df["source"] == "Schonhut-Stasik24", ["eTeff1", "eTeff2"]]
        df.loc[df["source"] == "Schonhut-Stasik24", ["eTeff1", "eTeff2"]] = np.sqrt(
            old_Teff**2 + 50**2
        )

    # select desired params and create avg symmetric err cols
    df = select_clean_data(
        df, training_fs, targets, s_class, add_logvars, L_check=L_check
    )
    # update training and targets in case of log switch
    for var in add_logvars:
        for list in [training_fs, targets]:
            if var in list:
                list.remove(var)
                list.append("log" + var)

    # filter by error limits and plot error distributions
    plot_params = None
    if plot_errs:
        plot_params = {
            "M": [7, "%"],
            "R": [7, "%"],
            "logL": [0.05, "dex"],
            "Teff": [100, "K"],
            "logg": [0.05, "dex"],
            "FeH": [0.1, "dex"],  # 0.1 for RGB
        }
    df = error_filter(
        df,
        savename=s_class + "_err_dist.pdf",
        abs_err_lims=abs_err_lims,
        percent_err_lims=percent_err_lims,
        plot_params=plot_params,
    )

    if target_lims:
        for target, lims in target_lims.items():
            df = df[(df[target] >= lims[0]) & (df[target] <= lims[1])]
    else:  # look at spread in target variables and decide whether or not to cut training range
        for var in targets:
            repeat = True
            spreadomatic(df, var)
            while repeat:
                lim0 = float(input(f"What lower limit do you want to put on {var}?\n"))
                lim1 = float(input(f"What upper limit do you want to put on {var}?\n"))
                df_copy = df[(df[var] >= lim0) & (df[var] <= lim1)]
                print(f"{len(df_copy)} stars left after trimming {var} range.")
                spreadomatic(df_copy, var)
                repeat = input(f"Continue checking {var} spread? (yes/no)\n") == "yes"
            df = df_copy
    print(f"{len(df)} stars left after cutting target ranges.")
    dataset_key = str(len(df)) + s_class
    df.to_csv("BayestarML/data/" + dataset_key + ".txt")

    # get MUs and SIGs for normalisation of each param, and write to constants.json, as well as MIN and MAX
    (
        X_train,
        X_test,
        Y_train,
        Y_test,
    ) = return_train_test(df, training_fs, targets, dataset_key)

    # normalise data and create final txt files for normalised training and testing datasets
    X_train_norm, Y_train_norm = normalise(X_train, Y_train, dataset_key)
    X_test_norm, Y_test_norm = normalise(X_test, Y_test, dataset_key)

    # save files
    normalised_train_data = pd.concat([X_train_norm, Y_train_norm], axis=1)
    normalised_test_data = pd.concat([X_test_norm, Y_test_norm], axis=1)
    normalised_train_data.to_csv("BayestarML/data/" + dataset_key + "_norm_train.txt")
    normalised_test_data.to_csv("BayestarML/data/" + dataset_key + "_norm_test.txt")

    print(
        "Normalised training and testing sets for "
        + dataset_key
        + " now saved in data folder."
    )

    # plot HR diagram of final data set (test and training) if desired
    if plotHR:
        HRplot(df, dataset_key + "_HRplot.pdf")
        print("HRplot saved in figures")

    # plot target-feature relations if desired
    if plot_t_f:
        for t in targets:
            for f in training_fs:
                savename = s_class + "/" + dataset_key + "_" + f + "_" + t + ".pdf"
                plot_feature_target(df, savename, f, t)

    # get xgboost estimate on final data set if desired
    if xgboost:
        for t in targets:
            print(
                "\nStarting xgboost benchmark on "
                + dataset_key
                + ", target: "
                + t
                + "\n"
            )
            X, y = df[training_fs], df[t]
            studyrunner(X, y, "xgb_" + dataset_key + "_" + t + ".json")
            print("For target " + t + "\n")


if __name__ == "__main__":
    # choose database to select data from for training
    filename = "datos_todos_v20261905.txt"
    # col headings should be 'col' for value and 'ecol1', 'ecol2' for corresponding errors

    training_fs = ["Teff", "logg", "FeH", "L"]
    targets = ["M", "R"]
    s_class = "rgb"
    add_logvars = ["L", "R"]  # add a log column with errs for these variables
    target_lims = {  # to skip the target range cutting step if you already know the limits you want
        "logR": [0, 1.6],
        "M": [0.75, 2.25],
    }

    abs_err_lims = {"elogL": 0.05, "eTeff": 100, "elogg": 0.05, "eFeH": 0.1}
    percent_err_lims = {"eM": 7, "eR": 7}

    # prepare data
    prep_data(
        filename,
        training_fs,
        targets,
        s_class,
        add_logvars,
        abs_err_lims,
        percent_err_lims,
        target_lims,
        L_check=True,
        xgboost=True,
    )


# 700ms
# abs_err_lims = {
#     "elogL": 0.5
# }
# percent_err_lims = {
#     "eM": 100,
#     "eR": 100
# }
