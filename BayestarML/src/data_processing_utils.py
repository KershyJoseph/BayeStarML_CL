"""Utility functions for preparing data for training or prediction."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

RANDOM_SEED = 5732


def denormalise_val(y, dataset_key, var, train_tar: str = "targets"):
    """ """
    with open("BayestarML/data/" + dataset_key + "_constants.json", "r") as f:
        constants = json.load(f)
    constants = constants[train_tar]
    return y * constants["SIG"][var] + constants["MU"][var]


def denormalise_err(y, dataset_key, var, train_tar: str = "targets"):
    """ """
    with open("BayestarML/data/" + dataset_key + "_constants.json", "r") as f:
        constants = json.load(f)
    sig = constants[train_tar]["SIG"]
    return y * sig[var]


def logomatic(df: pd.DataFrame, add_logvars: list):
    """Add a log(var) column to df with bounds method
    var should be string key of existing column in df
    """
    for var in add_logvars:
        invalids = df["e" + var + "2"] >= df[var]
        print(
            f"< Removing {len(df[invalids])} star(s) with {var}(s) that couldn't be logged >"
        )
        df = df[~invalids]
        df["log" + var] = np.log10(df[var])
        df["elog" + var + "1"] = (
            np.log10(df[var] + df["e" + var + "1"]) - df["log" + var]
        )
        df["elog" + var + "2"] = df["log" + var] - np.log10(
            df[var] - df["e" + var + "2"]
        )
    return df


def add_symmetric_errs(df, errs1, errs2):
    """ """
    df1 = df[errs1].copy()
    df2 = df[errs2].copy()
    df1.columns = errs2
    df_err = (df1 + df2) / 2
    df_err.columns = [err[:-1] for err in errs2]
    return pd.concat([df, df_err], axis=1)


def L_consistency_check(df, savename: str, logscale=True):
    """ """
    df = df.copy()
    df_L_check = df[df["L_from_SB"] == 0]
    df_L_check["L_SB"] = df_L_check["R"] ** 2 * (df_L_check["Teff"] / 5772) ** 4
    R = df_L_check["R"]
    Teff = df_L_check["Teff"]
    df_L_check["L_SB_+err"] = np.sqrt(
        (R**2 * ((Teff + df_L_check["eTeff1"]) / 5772) ** 4 - df_L_check["L_SB"]) ** 2
        + ((R + df_L_check["eR1"]) ** 2 * (Teff / 5772) ** 4 - df_L_check["L_SB"]) ** 2
    )
    df_L_check["L_SB_-err"] = np.sqrt(
        (R**2 * ((Teff - df_L_check["eTeff2"]) / 5772) ** 4 - df_L_check["L_SB"]) ** 2
        + ((R - df_L_check["eR2"]) ** 2 * (Teff / 5772) ** 4 - df_L_check["L_SB"]) ** 2
    )

    # compute distance from recorded Ls
    df_L_check["L_SB_avg_err"] = (df_L_check["L_SB_+err"] + df_L_check["L_SB_-err"]) / 2
    df_L_check["total_L_err"] = np.sqrt(
        df_L_check["L_SB_avg_err"] ** 2 + df_L_check["eL1"] ** 2
    )
    df_L_check["L_dist"] = df_L_check["L_SB"] - df_L_check["L"]
    df_L_check["L_sig_distance"] = (
        np.abs(df_L_check["L_dist"]) / df_L_check["total_L_err"]
    )
    df_bad_Ls = df_L_check[df_L_check["L_sig_distance"] > 3]

    plt.close()
    plt.figure()
    yerr = np.array([df_L_check["L_SB_-err"], df_L_check["L_SB_+err"]])
    xerr = np.array([df_L_check["eL2"], df_L_check["eL1"]])
    plt.errorbar(
        df_L_check["L"],
        df_L_check["L_SB"],  # x,y,yerr,xerr
        yerr=yerr,
        xerr=xerr,
        fmt="go",
        ecolor="gray",
        alpha=0.4,
        zorder=1,
    )
    yerr2 = np.array([df_bad_Ls["L_SB_-err"], df_bad_Ls["L_SB_+err"]])
    xerr2 = np.array([df_bad_Ls["eL2"], df_bad_Ls["eL1"]])
    plt.errorbar(
        df_bad_Ls["L"],
        df_bad_Ls["L_SB"],
        yerr=yerr2,
        xerr=xerr2,
        fmt="none",
        ecolor="red",
        alpha=0.5,
        zorder=2,
    )
    sns.scatterplot(data=df_bad_Ls, x="L", y="L_SB", hue="source", alpha=0.5, zorder=3)
    plt.xlabel("L")
    plt.ylabel("L from SB")
    plt.plot(
        [0, df_L_check["L"].max()],
        [0, df_L_check["L"].max()],
        linestyle="--",
        color="r",
    )
    if logscale:
        plt.xscale("log")
        plt.yscale("log")
    plt.show()
    plt.savefig("figures/L_check/" + savename)

    df.drop(df_bad_Ls.index, inplace=True)
    print(
        f"{len(df)} stars after checking L consistency with R and Teff to 3 sigma via SB law."
    )
    return df


def select_clean_data(
    df: pd.DataFrame,
    training_fs: list,
    targets: list,
    s_class: str = None,
    add_logvars: list = None,
    check_detached=True,
    L_check=True,
):
    """ """
    df = df.copy()
    if s_class:
        df = df[(df["class"] == s_class)]
        print(f"Working with {len(df)} " + s_class + " stars.")
    if check_detached:
        df = df[(df["well_detached"] != False)]
        print(
            f"{len(df)} stars left after filtering those not from well-detached binaries."
        )

    all_params = training_fs + targets
    # check params are present based on whether both errors are
    errs1 = [f"e{param}1" for param in all_params]
    errs2 = [f"e{param}2" for param in all_params]
    all_errs = errs1 + errs2
    df_allps = df[(df[all_errs].notna().all(axis=1)) & (df[all_errs].gt(0).all(axis=1))]
    print(
        f"{len(df_allps)} stars left after checking all training features and targets present with err>0 for each."
    )

    if L_check:
        df_allps = L_consistency_check(df_allps, s_class + "_Lcheck.pdf")

    # make any vars log10 scale
    if add_logvars:
        df_allps_log = logomatic(df_allps, add_logvars)
        errs1 += [f"elog{var}1" for var in add_logvars]
        errs2 += [f"elog{var}2" for var in add_logvars]

    # make an avg symmetric err col for all vars, log or not
    df_final = add_symmetric_errs(df_allps_log, errs1, errs2)

    return df_final


def error_filter(
    df, savename, abs_err_lims=None, percent_err_lims=None, plot_params=None
):
    """Filter df based on specified error tolerances."""
    df = df.copy()
    mask = pd.Series(True, index=df.index)
    if abs_err_lims:
        for evar, lim in abs_err_lims.items():
            mask &= df[evar] <= lim
    if percent_err_lims:
        for evar, p_lim in percent_err_lims.items():
            df["percent_" + evar] = 100 * df[evar] / df[evar[1:]]
            mask &= df["percent_" + evar] <= p_lim
    df_filtered = df[mask]
    print(f"{len(df_filtered)} stars left after error tolerance filtering.")

    if plot_params:
        plot_cols = int((len(plot_params) + 1) / 2)
        plt.close()
        fig, ax = plt.subplots(2, plot_cols)
        i, j = 0, 0
        for param, spec in plot_params.items():
            # spec should be a list [nominal err limit to plot, units of value]
            k = int(j)
            err = "e" + param
            if spec[1] == "%":  # make percentage errs have the right df key
                err = "percent_e" + param
            counts, _, _ = ax[i, k].hist(df_filtered[err], bins="auto")
            ax[i, k].vlines(
                spec[0],
                0,
                counts.max(),
                linestyle="--",
                color="r",
                label=str(spec[0]) + spec[1],
            )
            ax[i, k].set_title(param)
            ax[i, k].set_ylabel("Number")
            ax[i, k].set_xlabel(f"Error ({spec[1]})")
            ax[i, k].legend()
            # alternate i between 0 and 1
            i -= 1
            i = abs(i)
            # step j up to length plot(cols) waiting once each time
            j += 1 / 2
        plt.tight_layout()
        plt.show()
        fig.savefig("figures/err_dists/" + savename)

    return df_filtered


def spreadomatic(df, var, hue=None):
    """Make a histogram for a given var (which should be one of df's keys)"""
    plt.close()
    plt.figure()
    sns.histplot(data=df, x=var, hue=hue)
    plt.ylabel("Number of stars")
    plt.show()


def return_train_test(df, training_fs, targets, dataset_key):
    """ """
    training_fs_errs = [f"e{f}" for f in training_fs]
    x = pd.concat([df[training_fs], df[training_fs_errs]], axis=1)
    target_errs = [f"e{t}" for t in targets]
    y = pd.concat([df[targets], df[target_errs]], axis=1)

    # do split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_SEED
    )

    # get MU, SIG, MIN and MAX from training set
    x_means = x_train[training_fs].mean()
    y_means = y_train[targets].mean()
    x_stds = x_train[training_fs].std(ddof=0)
    y_stds = y_train[targets].std(ddof=0)

    x_min = x_train[training_fs].min()
    y_min = y_train[targets].min()
    x_max = x_train[training_fs].max()
    y_max = y_train[targets].max()

    x_constants = {
        "MU": x_means.to_dict(),
        "SIG": x_stds.to_dict(),
        "MIN": x_min.to_dict(),
        "MAX": x_max.to_dict(),
    }

    y_constants = {
        "MU": y_means.to_dict(),
        "SIG": y_stds.to_dict(),
        "MIN": y_min.to_dict(),
        "MAX": y_max.to_dict(),
    }

    master_constants = {"training_fs": x_constants, "targets": y_constants}

    with open("BayestarML/data/" + dataset_key + "_constants.json", "w") as f:
        json.dump(master_constants, f, indent=4)
    print(
        "Mu, std, min and max values for each parameter now stored in BayestarML/data/"
        + dataset_key
        + "_constants.json"
    )

    return x_train, x_test, y_train, y_test


def normalise(x: pd.DataFrame, y: pd.DataFrame, dataset_key: str, x_only: bool = False):
    """
    Normalise all de data
    """
    with open("BayestarML/data/" + dataset_key + "_constants.json", "r") as f:
        constants = json.load(f)

    # training fs
    x_constants = constants["training_fs"]
    x = x.copy()
    for param, sig in x_constants["SIG"].items():
        # standardise values
        x[param] = (x[param] - x_constants["MU"][param]) / sig
        # standardise errs
        x["e" + param] /= sig
    if x_only:
        return x

    # targets
    y_constants = constants["targets"]
    y = y.copy()
    for param, sig in y_constants["SIG"].items():
        # standardise values
        y[param] = (y[param] - y_constants["MU"][param]) / sig
        # standardise errs
        y["e" + param] /= sig

    return x, y


def HRplot(df, savename: str, hue: str = None):
    """Plot logTeff (higher Teff to the left) against logL for stars in df"""
    df = logomatic(df, ["Teff"])

    x = df["logTeff"]
    x_err = [df["elogTeff2"], df["elogTeff1"]]
    y = df["logL"]
    y_err = [df["elogL2"], df["elogL1"]]

    plt.close()
    fig, ax = plt.subplots()

    fmt = "o"
    if hue:
        sns.scatterplot(
            data=df, x="logTeff", y="logL", hue=hue, ax=ax, zorder=3, alpha=0.8
        )
        fmt = "none"
    ax.errorbar(x, y, y_err, x_err, fmt=fmt, ecolor="grey", alpha=0.5, zorder=2)
    ax.xaxis.set_inverted(True)
    ax.set_xlabel("log[ Teff (K) ]")
    ax.set_ylabel("log[ L (Lsol) ]")

    fig.savefig("figures/HRdiagrams/" + savename)
    plt.close()


def plot_feature_target(df: pd.DataFrame, savename: str, feature: str, target: str):
    """Plot target as a function of feature - should be keys in df"""
    plt.figure()
    x = df[feature]
    x_err = df["e" + feature]  # +"1"]
    y = df[target]
    y_err = df["e" + target]
    plt.errorbar(x, y, y_err, x_err, fmt="o", alpha=0.3)
    # plt.plot(np.log10(1.193), 0.499, 'rx')
    # plt.plot(np.log10(0.08), 0.566, 'rx')
    plt.xlabel(feature)
    plt.ylabel(target + " (" + target[0] + "sol)")
    plt.savefig("figures/feature_target_figs/" + savename)
    plt.close()


def prepare_pred_data(
    filename: str, training_dataset_key: str, features: list, add_log_vars: list = None
):
    """
    Normalize input data and return DataFrames for normalized values and errors.
    Check all input data within training ranges.

    Parameters:
    - teff, logg, FeH, l: Input values (can be scalars or arrays)
    - eteff, elogg, eFeH, el: Associated errors (can be scalars or arrays)

    Returns:
    - x_test: DataFrame with normalized values (columns: 'Teff', 'logg', 'FeH', 'L')
    - x_test_error: DataFrame with normalized errors (columns: 'eTeff', 'elogg', 'eFeH', 'eL')
    """
    x = pd.read_csv("predict_BayestarML/data/" + filename, sep="\t")
    # filter to stars with all training features present with err. Add symmetric err column and log vars if needed.
    x = select_clean_data(
        x,
        features,
        targets="",
        add_logvars=add_log_vars,
        check_detached=False,
        L_check=False,
    )

    with open("BayestarML/data/" + training_dataset_key + "_constants.json", "r") as f:
        constants = json.load(f)
        x_constants = constants["training_fs"]

    # check predicting within feature training ranges
    for f in features:
        MIN = x_constants["MIN"][f]
        MAX = x_constants["MAX"][f]
        RANGE = MAX - MIN
        len_before = len(x)
        x[f] = x[
            (x[f] >= MIN + 0.025 * RANGE) & (x[f] <= MAX - 0.025 * RANGE)
        ]  # keep middle 95%
        len_after = len(x)
        print(
            f"Removed {len_before - len_after} stars checking {f} inputs within middle 95% {f} training range."
        )

    # normalise input data
    x_norm = normalise(x, None, training_dataset_key, x_only=True)

    return x_norm  # might need modifying for scalar inputs?
