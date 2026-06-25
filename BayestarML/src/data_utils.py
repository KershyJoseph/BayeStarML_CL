"""Utility functions for preparing data for training or prediction."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde

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


def consistency_check(df, param, flag_col, savename: str, logscale=True):
    """
    """
    df_check = df.copy()
    if flag_col:
        df_check = df_check[df_check[flag_col] == 0]
    
    if param == "logg":
        x_col, y_col = "logg", "logg_from_M,R"
        x_label = "Recorded log(g) (dex)"
        y_label = "Calculated log(g) from M and R (dex)"
        x = df_check[x_col]
        xerr_up = df_check["elogg1"]
        xerr_down = df_check["elogg2"]
        print(f"Checking {len(df_check)} stars for logg consistency")

        R = df_check["R"]
        eR1 = df_check["eR1"]
        eR2 = df_check["eR2"]
        M = df_check["M"]
        eM1 = df_check["eM1"]
        eM2 = df_check["eM2"]
        GMsol = 1.3271244e26 # Andrej Prsa 2016 - adjusted to be cm**3/s**2!
        Rsol = 6.957e10 #cm

        y = np.log10((GMsol * M)/(R * Rsol)**2)
        df_check[y_col] = y
        yerr_up = np.sqrt(
            (np.log10((GMsol * (M + eM1))/(R * Rsol)**2) - y)**2
            +
            (np.log10((GMsol * M)/((R-eR1) * Rsol)**2) - y)**2
        )
        yerr_down = np.sqrt(
            (np.log10((GMsol * (M - eM2))/(R * Rsol)**2) - y)**2
            +
            (np.log10((GMsol * M)/((R+eR2) * Rsol)**2) - y)**2
        )

        y_err_avg = (yerr_up + yerr_down)/2
        x_err_avg = (xerr_up + xerr_down)/2
        total_sig = np.sqrt(y_err_avg**2 + x_err_avg**2)
        sig_discrepancy = np.abs(y - x)/total_sig 

        consistency_col = "log(g) consistent with M and R"
        df_check[consistency_col] = np.where(sig_discrepancy > 3, "deviation > 3 sigma", "within 3 sigma")

        yerr = np.array([yerr_down, yerr_up])
        xerr = np.array([xerr_down, xerr_up])

        logscale=False

    elif param == "L":
        x_col, y_col = "L", "L_from_SB"
        x_label = "Recorded L (Lsol)"
        y_label = "Calculated L from SB Law (Lsol)"
        x = df_check[x_col]
        print(f"Checking {len(df_check)} stars for L consistency")

        R = df_check["R"]
        Teff = df_check["Teff"]
        y = R ** 2 * (Teff / 5772) ** 4
        df_check[y_col] = y
        df_check["L_SB_+err"] = np.sqrt(
            (R**2 * ((Teff + df_check["eTeff1"]) / 5772) ** 4 - y) ** 2
            + ((R + df_check["eR1"]) ** 2 * (Teff / 5772) ** 4 - y) ** 2
        )
        df_check["L_SB_-err"] = np.sqrt(
            (R**2 * ((Teff - df_check["eTeff2"]) / 5772) ** 4 - y) ** 2
            + ((R - df_check["eR2"]) ** 2 * (Teff / 5772) ** 4 - y) ** 2
        )

        # compute distance from recorded Ls
        df_check["L_SB_avg_err"] = (df_check["L_SB_+err"] + df_check["L_SB_-err"]) / 2
        df_check["total_L_err"] = np.sqrt(
            df_check["L_SB_avg_err"] ** 2 + df_check["eL1"] ** 2
        )
        df_check["L_dist"] = y - x
        df_check["L_sig_distance"] = (
            np.abs(df_check["L_dist"]) / df_check["total_L_err"]
        )

        consistency_col = "L consistent with SB law"
        df_check[consistency_col] = np.where(df_check["L_sig_distance"] > 3, "deviation > 3 sigma", "within 3 sigma")
        yerr = np.array([df_check["L_SB_-err"], df_check["L_SB_+err"]])
        xerr = np.array([df_check["eL2"], df_check["eL1"]])

    plt.close()
    plt.figure()
    plt.errorbar(
        x, y,
        yerr=yerr,
        xerr=xerr,
        fmt='none',
        ecolor="gray",
        alpha=0.5,
        zorder=2,
    )
    sns.scatterplot(data=df_check, x=x_col, y=y_col, hue=consistency_col, alpha=0.5, zorder=3)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(
        [x.min(), x.max()],
        [x.min(), x.max()],
        linestyle="--",
        color="gray",
        zorder=1
    )
    if logscale:
        plt.xscale("log")
        plt.yscale("log")
    plt.savefig("BayestarML/data/figures/lum_check/" + savename)
    plt.show()

    df_bad_stars = df_check[df_check[consistency_col]=='deviation > 3 sigma']
    print(
        f"Removing {len(df_bad_stars)} stars that do not have {consistency_col} to 3 sigma:\n{df_bad_stars}"
    )
    print(len(df))
    df.drop(df_bad_stars.index, inplace=True)
    print(len(df))

    return df


def select_clean_data(
    df: pd.DataFrame,
    training_fs: list,
    targets: list,
    s_class: str = None,
    add_logvars: list = None,
    check_detached=True,
    check_consistency=True,
):
    """ """
    df = df.copy()
    if s_class:
        df = df[(df["class"] == s_class)]
        print(f"Working with {len(df)} " + s_class + " stars.")
    if check_detached:
        df = df[df["well_detached"].ne(False)]
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

    if check_consistency:
        df_allps = consistency_check(df_allps, "logg", "logg_from_M,R", s_class + "_logg_check.pdf")
        df_allps = consistency_check(df_allps, "L", "L_from_SB", s_class + "_lum_check.pdf")

    # make any vars log10 scale
    if add_logvars:
        df_allps = logomatic(df_allps, add_logvars)
        errs1 += [f"elog{var}1" for var in add_logvars]
        errs2 += [f"elog{var}2" for var in add_logvars]

    # make an avg symmetric err col for all vars, log or not
    df_final = add_symmetric_errs(df_allps, errs1, errs2)

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
        fig.savefig("BayestarML/data/figures/err_dists/" + savename)

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


def hr_plot(df, savename: str, hue: str = None, density_plot: bool = False, mass_plot: bool = False):
    """Plot logTeff (higher Teff to the left) against logL for stars in df
    """
    if "logTeff" not in df.columns:
        df = logomatic(df, ["Teff"])
    if "logL" not in df.columns:
        df = logomatic(df, ["L"])

    x = df["logTeff"]
    x_err = [df["elogTeff2"], df["elogTeff1"]]
    y = df["logL"]
    y_err = [df["elogL2"], df["elogL1"]]

    plt.close()
    fig, ax = plt.subplots(figsize=(8,6))

    colors = {
        "NASA Exoplanet Archive": "lightsteelblue",
        "Lamirel et al. (2026)": "mediumblue",
        "Expansion in this work": "red"
    }

    zorders = {
        "NASA Exoplanet Archive": 2,
        "Lamirel et al. (2026)": 4,
        "Expansion in this work": 4
    }

    if hue:
        for catalogue, group in df.groupby(hue):
            x = group["logTeff"]
            x_err = [group["elogTeff2"], group["elogTeff1"]]
            y = group["logL"]
            y_err = [group["elogL2"], group["elogL1"]]
            ax.errorbar(x, y, y_err, x_err, fmt='o', ms=5, capsize=1.5, alpha=0.5, ecolor='gray', color=colors[catalogue], mec='gray', label=catalogue, zorder=zorders[catalogue])
    if density_plot:
        stack = np.vstack([x, y])
        density = gaussian_kde(stack)(stack)
        idx = np.argsort(density)
        x, y, density = x[idx], y[idx], density[idx]
        scatter = ax.scatter(x, y, c=density, cmap='plasma', s=5, zorder=3)
        fig.colorbar(scatter, ax=ax, label="Density of Stars")
    elif mass_plot:
        scatter = ax.scatter(x, y, c=df["M"], cmap='plasma', s=5, zorder=3)
        fig.colorbar(scatter, ax=ax, label="Mass (Msol)")

    ax.xaxis.set_inverted(True)
    ax.set_xlabel(r"log($\mathrm{T_{eff}}$) [K]")
    ax.set_ylabel(r"log(L) [$\mathrm{L_{\odot}}$]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.savefig("BayestarML/data/figures/hr_diagrams/" + savename)
    plt.close()


def plot_feature_target(df: pd.DataFrame, savename: str, feature: str, target: str, density_plot: bool = False, bad_paretos=None):
    """Plot target as a function of feature - should be keys in df
    """
    x = df[feature]
    x_err = df["e" + feature]
    y = df[target]
    y_err = df["e" + target]
    fmt="o"
    fig, ax = plt.subplots(figsize=(8,6))
    if density_plot:
        stack = np.vstack([x, y])
        density = gaussian_kde(stack)(stack)
        idx = np.argsort(density)
        x, y, density = x[idx], y[idx], density[idx]

        scatter = ax.scatter(x, y, c=density, cmap='plasma', s=10, zorder=3)
        fig.colorbar(scatter, ax=ax, label="Density")
        fmt = "none"
    ax.errorbar(x, y, y_err, x_err, fmt=fmt, alpha=0.5, zorder=2)
    if bad_paretos:
        df_bad = df.iloc[bad_paretos]
        plt.plot(df_bad[feature], df_bad[target], 'rx', zorder=4)
        print(df_bad[[feature, "e"+feature]])
    ax.set_xlabel(feature)
    ax.set_ylabel(target + " (" + target[0] + "sol)")
    fig.savefig("BayestarML/data/figures/feature_target_figs/" + savename)
    plt.close()

# df = pd.read_csv("BayestarML/data/693ms.txt")
# training_fs = ["Teff", "logg", "FeH", "logL"]
# targets = ["M", "R"]
# dataset_key = "693ms"
# print(len(df))
# print(len(df)*0.8)
# (x_train, x_test, y_train, y_test) = return_train_test(df, training_fs, targets, dataset_key)
# df_train = pd.concat([x_train, y_train], axis=1)
# print(df_train)

# for f in training_fs:
#     plot_feature_target(df_train, "bad_paretos/paretos_GP_M_"+f+"_693ms.pdf", f, "M", bad_paretos=[57, 75, 139, 187, 216, 268, 275, 282, 323, 343, 349, 362, 430, 446, 499])


#HR PLOTTING

# df_2018 = pd.read_csv("BayestarML/data/training_databases/all6_2018_data.txt", sep='\t')
# df_2018 = df_2018[df_2018["class"]=="ms"]
# df_nasa = pd.read_csv("BayestarML/predict/prediction_datasets/NASAexop_archive_stars.txt", sep='\t')
# df_new = pd.read_csv("BayestarML/data/693ms.txt")
# df_new = logomatic(df_new, ["Teff"])
# df_all = pd.concat([df_nasa, df_2018, 
#                     df_new], 
#                     keys=["NASA Exoplanet Archive", "Lamirel et al. (2026)", 
#                           "Expansion in this work"],
#                     names=["Catalogue", None])
# df_all.reset_index()

# hr_plot(df_all, "max_nasa_new_hr_plot.pdf", hue="Catalogue")
