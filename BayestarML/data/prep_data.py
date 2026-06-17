"""Executable to prepare a given database file for training on the BHS model
"""

import numpy as np
import pandas as pd
from BayestarML.src.data_utils import (
    hr_plot,
    error_filter,
    normalise,
    plot_feature_target,
    return_train_test,
    select_clean_data,
    spreadomatic
)
from BayestarML.src.models.baseline_xgb import studyrunner


def prep_data(
    filename: str,
    training_fs: list,
    targets: list,
    s_class: str,
    add_logvars: list = None,
    abs_err_lims: dict = None,
    percent_err_lims: dict = None,
    target_lims: dict = None,
    check_detached:bool = True,
    lum_check: bool = False,
    plot_errs: bool = False,
    plot_hr: bool = False,
    plot_t_f=False,
    xgboost: bool = False,
):
    """ """
    df = pd.read_csv("BayestarML/data/training_databases/" + filename, sep="\t", comment="#")
    df.set_index("ID", inplace=True)

    # if RGB, fix Teff errs: add 50K systematic error in quadrature to all Sch-St Teff errs, as were taken straight from APOGEE https://iopscience.iop.org/article/10.3847/1538-4357/ac4891
    if s_class == "rgb":
        old_Teff = df.loc[df["source"] == "Schonhut-Stasik24", ["eTeff1", "eTeff2"]]
        df.loc[df["source"] == "Schonhut-Stasik24", ["eTeff1", "eTeff2"]] = np.sqrt(
            old_Teff**2 + 50**2
        )

    # select desired params and create avg symmetric err cols
    df = select_clean_data(
        df, training_fs, targets, s_class, add_logvars, check_detached=check_detached, lum_check=lum_check
    )
    # update training and targets in case of log switch
    if add_logvars:
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
            "FeH": [0.15, "dex"],  # 0.1 for RGB
        }
    df = error_filter(
        df,
        savename=s_class + "2018_err_dist.pdf",
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
        Y_test
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
    if plot_hr:
        hr_plot(df, dataset_key + "_hr_plot.pdf")
        print("HR plot saved in figures")

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
    filename = "all6_2018_data.txt"
    # col headings should be 'col' for value and 'ecol1', 'ecol2' for corresponding errors

    training_fs = ["Teff", "logg", "FeH", "logL"]
    targets = ["M", "R"]
    s_class = "ms"
    add_logvars = None  # add a log column with errs for these variables

    #skip the target range cutting step if you already know the limits you want
    target_lims = {
        "R": [0,100], #rgb [0, 1.6],
        "M": [0,100]#rgb [0.75, 2.25],
    }

    # 700ms
    abs_err_lims = {
        "elogL": 0.5
    }
    percent_err_lims = {
        "eM": 100,
        "eR": 100
    }

    # 5438rgb
    # abs_err_lims = {"elogL": 0.05, "eTeff": 100, "elogg": 0.05, "eFeH": 0.1}
    # percent_err_lims = {"eM": 7, "eR": 7}

    # prepare data
    prep_data(
        filename,
        training_fs,
        targets,
        s_class,
        # add_logvars,
        # abs_err_lims,
        # percent_err_lims,
        target_lims=target_lims,
        check_detached=False,
        # lum_check=False,
        xgboost=True,
        # plot_hr=True,
        # plot_t_f=True
    )
