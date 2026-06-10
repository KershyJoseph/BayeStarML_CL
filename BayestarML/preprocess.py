#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 18:03:17 2025

@author: LamirelFamily
"""

"""Utility functions shared by many modules."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from constants import MU, SIGMA
from utils import get_dataset
from sklearn.model_selection import train_test_split

RANDOM_SEED = 5732 

def normalise_val(x: float | None, key: str) -> float:
    return np.nan if x is None else (x - MU[key]) / SIGMA[key]

def normalise_err(e: float | None, key: str) -> float:
    return np.nan if e is None else abs(e) / SIGMA[key]

def denormalise_val(y: np.ndarray, key: str) -> np.ndarray:
    return y * SIGMA[key] + MU[key]

def denormalise_err(y: np.ndarray, key: str) -> np.ndarray:
    return y * SIGMA[key]

def logomatic(df, add_logvars):
    """Add a log(var) column to df with bounds method 
    var should be string key of existing column in df
    """
    for var in add_logvars:
        invalids = (df["e"+var+"2"]>=df[var])
        print(f"< Removing {len(df[invalids])} star(s) with {var}(s) that couldn't be logged >")
        df = df[~invalids]
        df["log"+var] = np.log10(df[var])
        df["elog"+var+"1"] = np.log10(df[var] + df["e"+var+"1"]) - df["log"+var]
        df["elog"+var+"2"] = df["log"+var] - np.log10(df[var] - df["e"+var+"2"])
    return df

def add_symmetric_errs(df, errs1, errs2):
    """
    """
    df1 = df[errs1].copy()
    df2 = df[errs2].copy()
    df1.columns = errs2
    df_err = (df1 + df2) / 2
    df_err.columns = [err[:-1] for err in errs2]
    return pd.concat([df, df_err], axis=1)

def L_consistency_check(df, logscale=True):
    """
    """
    df_L_check = df[df["L_from_SB"]==0]
    df_L_check["L_SB"] = df_L_check["R"]**2 * (df_L_check["Teff"]/5772)**4
    R = df_L_check["R"]
    Teff = df_L_check["Teff"]
    df_L_check["L_SB_+err"] = np.sqrt(
        (R**2*((Teff+df_L_check["eTeff1"])/5772)**4 - df_L_check["L_SB"])**2 
        + ((R+df_L_check["eR1"])**2*(Teff/5772)**4 - df_L_check["L_SB"])**2 
    )
    df_L_check["L_SB_-err"] = np.sqrt(
        (R**2*((Teff-df_L_check["eTeff2"])/5772)**4 - df_L_check["L_SB"])**2 
        + ((R-df_L_check["eR2"])**2*(Teff/5772)**4 - df_L_check["L_SB"])**2 
    )

    #compute distance from recorded Ls
    df_L_check["L_SB_avg_err"] = (df_L_check["L_SB_+err"] + df_L_check["L_SB_-err"])/2
    df_L_check["total_L_err"] = np.sqrt(df_L_check["L_SB_avg_err"]**2 + df_L_check["eL1"]**2)
    df_L_check["L_dist"] = df_L_check["L_SB"]-df_L_check["L"]
    df_L_check["L_sig_distance"] = np.abs(df_L_check["L_dist"])/df_L_check["total_L_err"]
    df_bad_Ls = df_L_check[df_L_check["L_sig_distance"]>3]

    plt.figure()
    yerr = np.array([df_L_check["L_SB_-err"], df_L_check["L_SB_+err"]])
    xerr = np.array([df_L_check["eL2"], df_L_check["eL1"]])
    plt.errorbar(df_L_check["L"], df_L_check["L_SB"], #x,y,yerr,xerr
                yerr=yerr, xerr=xerr, fmt='go', ecolor='gray', alpha=0.4,
                zorder=1)
    yerr2 = np.array([df_bad_Ls["L_SB_-err"], df_bad_Ls["L_SB_+err"]])
    xerr2 = np.array([df_bad_Ls["eL2"], df_bad_Ls["eL1"]])
    plt.errorbar(df_bad_Ls["L"], df_bad_Ls["L_SB"],
                yerr=yerr2, xerr=xerr2, fmt='none', ecolor='red', alpha=0.5,
                zorder=2)
    sns.scatterplot(data=df_bad_Ls,x="L",y="L_SB",hue="source",alpha=0.5,zorder=3)
    plt.xlabel("L")
    plt.ylabel("L from SB")
    plt.plot([0, df_L_check["L"].max()], [0,df_L_check["L"].max()], linestyle='--', color='r')
    if logscale:
        plt.xscale("log")
        plt.yscale("log")
    plt.show()

    df.drop(df_bad_Ls.index, inplace=True)
    print(f"{len(df)} stars after checking L consistency with R and Teff to 3 sigma via SB law.")
    return df

def select_clean_data(df: pd.DataFrame,
                      training_fs: list, targets: list, s_class: str,
                      add_logvars: list = None,
                      check_detached=True,
                      L_check=True):
    """
    """
    if s_class:
        df = df[(df["class"]==s_class)]
        print(f"Working with {len(df)} "+s_class+" stars.")
    if check_detached:
        df = df[(df["well_detached"]!=False)]
        print(f"{len(df)} stars left after filtering those not from well-detached binaries.")

    all_params = training_fs + targets
    #check params are present based on whether both errors are
    errs1 = [f"e{param}1" for param in all_params] 
    errs2 = [f"e{param}2" for param in all_params]
    all_errs = errs1 + errs2
    df_allps = df[(df[all_errs].notna().all(axis=1)) &
                  (df[all_errs].gt(0).all(axis=1))]
    print(f"{len(df_allps)} stars left after checking all training features and targets present with err>0 for each.")

    if L_check:
        df_allps = L_consistency_check(df_allps)

    #make any vars log10 scale
    if add_logvars:
        df_allps_log = logomatic(df_allps, add_logvars)
        errs1 += [f"elog{var}1" for var in add_logvars]
        errs2 += [f"elog{var}2" for var in add_logvars]

    #make an avg symmetric err col for all vars, log or not
    df_final = add_symmetric_errs(df_allps_log, errs1, errs2)

    return df_final

def error_filter(df, abs_err_lims=None, percent_err_lims=None, plot_params=None):
    """Filter df based on specified error tolerances.
    """
    mask = pd.Series(True, index=df.index)
    if abs_err_lims:
        for evar, lim in abs_err_lims.items():
            mask &= (df[evar] <= lim)
    if percent_err_lims:
        for evar, p_lim in percent_err_lims.items():
            df["percent_"+evar] = 100 * df[evar]/df[evar[1:]]
            mask &= (df["percent_"+evar] <= p_lim)
    df_filtered = df[mask]
    print(f"{len(df_filtered)} stars left after error tolerance filtering.")

    if plot_params:
        plot_cols = int((len(plot_params)+1)/2)
        fig, ax = plt.subplots(2,plot_cols)
        i, j = 0, 0
        for param, spec in plot_params.items():
            #spec should be a list [nominal err limit to plot, units of value]
            k=int(j)
            err = "e"+param
            if spec[1] == "%": #make percentage errs have the right df key
                err = "percent_e"+param
            counts, _, _ = ax[i,k].hist(df_filtered[err], bins='auto')
            ax[i,k].vlines(spec[0],0,counts.max(),linestyle='--',color='r',label=str(spec[0])+spec[1])
            ax[i,k].set_title(param)
            ax[i,k].set_ylabel("Number")
            ax[i,k].set_xlabel(f"Error ({spec[1]})")
            ax[i,k].legend()
            #alternate i between 0 and 1
            i-=1
            i=abs(i)
            #step j up to length plot(cols) waiting once each time
            j+=1/2
        plt.tight_layout()
        plt.show()

    return df_filtered

def spreadomatic(df, var, hue=None):
    """Make a histogram for a given var (which should be one of df's keys)
    """
    plt.figure()
    sns.histplot(data=df, x=var, hue=hue)
    plt.ylabel("Number of stars")
    plt.show()

def return_train_test_edit(df, training_fs, targets):
    """
    """
    training_fs_errs = [f"e{f}" for f in training_fs]
    X = pd.concat([df[training_fs], df[training_fs_errs]], axis=1)
    target_errs = [f"e{t}" for t in targets]
    Y = pd.concat([df[targets], df[target_errs]], axis=1)

    # do split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=0.2,
                                                        random_state=RANDOM_SEED)

    #get MU, SIG, MIN and MAX from training set
    X_means = X_train[training_fs].mean()
    Y_means = Y_train[targets].mean()
    X_stds = X_train[training_fs].std()
    Y_stds = Y_train[targets].std()

    X_min = X_train[training_fs].min()
    Y_min = Y_train[targets].min()
    X_max = X_train[training_fs].max()
    Y_max = Y_train[targets].max()

    MU = pd.concat([X_means, Y_means]).to_dict()
    SIG = pd.concat([X_stds, Y_stds]).to_dict()
    MIN = pd.concat([X_min, Y_min]).to_dict()
    MAX = pd.concat([X_max, Y_max]).to_dict()


def normalise():
    x = None


def return_train_test(df, normalised=True, logL=False, logR=False):
    """
    ***Added a logL, logR option***

    Parameters
    ----------
    df : TYPE, pandas df
        DESCRIPTION. The default is df. All data.
    normalised : TYPE, bool
        DESCRIPTION. The default is True.

    Returns
    -------
    normalised or not normalised training and testing data. 
    Note that normalised and non normalised don't come in the same format
    For normalised: x_train, x_train_er, x_test, x_test_error,
    mass, emass, mass_test, emass_test
    For non normalised: X_train, X_test, Y_train, Y_test / where errors and
    data are combined

    if you want both just call twice

    """
    if logL == True:
        eL1 = "elogL1"
        eL2 = "elogL2"
        L = "logL"
        eL = "elogL" 
    else:
        eL1 = "eL1"
        eL2 = "eL2"
        L = "L"
        eL = "eL"

    if logR == True:
        eR1 = "elogR1"
        eR2 = "elogR2"
        R = "logR"
        eR = "elogR" 
    else:
        eR1 = "eR1"
        eR2 = "eR2"
        R = "R"
        eR = "eR"

    df1 = df[['eTeff1', 'elogg1', 'eFeH1', eL1, 'eM1', eR1]].copy()
    df2 = df[['eTeff2', 'elogg2', 'eFeH2', eL2, 'eM2', eR2]].copy()
    df2.columns = ['eTeff1', 'elogg1', 'eFeH1', eL1, 'eM1', eR1]

    # Mean error if non-symmetric
    X_error = (df1 + df2) / 2 

    X_error.columns = ['eTeff', 'elogg', 'eFeH', eL, 'eM', eR]

    X = pd.concat([df[['Teff', L, 'FeH', 'logg']],
                   X_error[['eTeff', 'elogg', 'eFeH', eL]]],
                  axis=1)
    Y = pd.concat([df['M'], X_error['eM'], df[R], X_error[eR]], axis=1)

    # do split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=0.2,
                                                        random_state=RANDOM_SEED)

    # Extract relevant columns for stellar mass prediction
    teff = X_train['Teff']
    logg = X_train['logg']
    met = X_train['FeH']
    lum = X_train[L]
    #print(lum)
    mass = Y_train["M"]
    rad = Y_train[R]

    # Compute means and standard deviations for standardization
    mteff = np.mean(teff)
    mlogg = np.mean(logg)
    mmet = np.mean(met)
    mlum = np.mean(lum)
    mtmass = np.mean(mass)
    mrad = np.mean(rad)

    print(mteff, mlogg, mmet, mlum, mtmass, mrad)

    steff = np.std(teff)
    slogg = np.std(logg)
    smet = np.std(met)
    slum = np.std(lum)
    smass = np.std(mass)
    srad = np.std(rad)
    
    print(steff, slogg, smet, slum, smass, srad)

    # Standardize inputs 
    teff = (teff - mteff) / steff
    logg = (logg - mlogg) / slogg
    met = (met - mmet) / smet
    lum = (lum - mlum) / slum
    mass = (mass - mtmass) / smass
    rad = (rad - mrad) / srad

    # Uncertainties for the inputs
    eteff = X_train['eTeff'] / steff
    elog = X_train['elogg'] / slogg
    emet = abs(X_train['eFeH']) / smet
    elum = X_train[eL] / slum  
    emass = Y_train['eM'] / smass
    erad = Y_train[eR] / srad

    x_train = pd.concat([teff, logg, met, lum], axis=1)
    x_train_er = pd.concat([eteff, elog, emet, elum], axis=1)

    teff_test = X_test['Teff']
    logg_test = X_test['logg']
    met_test = X_test['FeH']
    lum_test = X_test[L] 
    mass_test = Y_test['M']
    rad_test = Y_test[R]
     
    teff_test = (teff_test - mteff) / steff
    logg_test = (logg_test - mlogg) / slogg
    met_test = (met_test - mmet) / smet
    lum_test = (lum_test- mlum) / slum
    mass_test = (mass_test- mtmass) / smass
    rad_test = (rad_test - mrad) / srad

    x_test = pd.concat([teff_test, logg_test, met_test, lum_test], axis=1)

    eteff_test = X_test['eTeff'] / steff
    elog_test = X_test['elogg'] / slogg
    emet_test = abs(X_test['eFeH']) / smet
    elum_test = X_test[eL] / slum 
    emass_test = Y_test['eM'] / smass
    erad_test = Y_test[eR] / srad

    x_test_error = pd.concat([eteff_test, elog_test, emet_test, elum_test],
                             axis=1)

    if normalised == True:
        return x_train, x_train_er, x_test, x_test_error, mass, emass, mass_test, emass_test, rad, erad, rad_test, erad_test

    if normalised == False:
        return X_train, X_test, Y_train, Y_test

def prepare_pred4(filename, logL=False, logR=False):
    """
    ***Added logL option***
    Normalize input data and return DataFrames for normalized values and errors.

    Parameters:
    - teff, logg, FeH, l: Input values (can be scalars or arrays)
    - eteff, elogg, eFeH, el: Associated errors (can be scalars or arrays)
    - codeword: Value that indicates missing data (will be converted to NaN)

    Returns:
    - x_test: DataFrame with normalized values (columns: 'Teff', 'logg', 'FeH', 'L')
    - x_test_error: DataFrame with normalized errors (columns: 'eTeff', 'elogg', 'eFeH', 'eL')
    """
    if logL == True:
        L = "logL"
        eL = "elogL" 
    else:
        L = "L"
        eL = "eL"

    if logR == True:
        R = "logR"
        eR = "elogR" 
    else:
        R = "R"
        eR = "eR"

    X = pd.read_csv(filename, sep='\t')

    # Helper function to normalize and handle missing values
    def normalize(value, mean, std):
        if value is None:
            return np.nan
        return (np.array(value) - mean) / std

    # Helper function to normalize errors (absolute value)
    def normalize_error(error, std):
        if error is None:
            return np.nan
        return abs(np.array(error)) / std

    # Normalize each parameter and its error
    norm_data = {
        'Teff': normalize(X['Teff'], MU['Teff'], SIGMA['Teff']),
        'logg': normalize(X['logg'], MU['logg'], SIGMA['logg']),
        'FeH': normalize(X['FeH'], MU['FeH'], SIGMA['FeH']),
        L: normalize(X[L], MU[L], SIGMA[L])
    }

    error_data = {
        'eTeff': normalize_error(X['eTeff'], SIGMA['Teff']),
        'elogg': normalize_error(X['elogg'], SIGMA['logg']),
        'eFeH': normalize_error(X['eFeH'], SIGMA['FeH']),
        eL: normalize_error(X[eL], SIGMA[L])
    }

    # For scalar inputs, we need to create a single-row DataFrame
    if (not hasattr(X['Teff'], '__len__') or isinstance(X['Teff'], str)) and X['Teff'] is not None:
        x_test = pd.DataFrame(norm_data, index=[0])
        x_test_error = pd.DataFrame(error_data, index=[0])
    else:
        x_test = pd.DataFrame(norm_data)
        x_test_error = pd.DataFrame(error_data)

    return x_test, x_test_error

def prepare_pred3(filename):
    """
    Normalize input data and return DataFrames for normalized values and errors.

    Parameters:
    - teff, logg, FeH, l: Input values (can be scalars or arrays)
    - eteff, elogg, eFeH, el: Associated errors (can be scalars or arrays)
    - codeword: Value that indicates missing data (will be converted to NaN)

    Returns:
    - x_test: DataFrame with normalized values (columns: 'Teff', 'logg', 'FeH', 'L')
    - x_test_error: DataFrame with normalized errors (columns: 'eTeff', 'elogg', 'eFeH', 'eL')
    """

    X = pd.read_csv(filename)
    df = get_dataset('Datasets/data_sample_mass_radius.txt', 'MS')
    mteff, mlogg, mmet, mlum, mtmass, steff, slogg, smet, slum, smass = return_norm(df)

    # Helper function to normalize and handle missing values
    def normalize(value, mean, std):
        if value is None:
            return np.nan
        return (np.array(value) - mean) / std

    # Helper function to normalize errors (absolute value)
    def normalize_error(error, std):
        if error is None:
            return np.nan
        return abs(np.array(error)) / std

    # Normalize each parameter and its error
    norm_data = {
        'Teff': normalize(X['Teff'], mteff, steff),
        'logg': normalize(X['logg'], mlogg, slogg),
        'FeH': normalize(X['FeH'], mmet, smet)
    }

    error_data = {
        'eTeff': normalize_error(X['eTeff'], steff),
        'elogg': normalize_error(X['elogg'], slogg),
        'eFeH': normalize_error(X['eFeH'], smet)
    }

    if (not hasattr(X['Teff'], '__len__') or isinstance(X['Teff'], str)) and X['Teff'] is not None:
        x_test = pd.DataFrame(norm_data, index=[0])
        x_test_error = pd.DataFrame(error_data, index=[0])
    else:
        x_test = pd.DataFrame(norm_data)
        x_test_error = pd.DataFrame(error_data)

    return x_test, x_test_error

# def return_norm(df, logL=False):
#     """
#     ***Added a logL option***

#     Compute normalization statistics for stellar parameters and their errors.

#     Extracts stellar feature columns and their associated asymmetric measurement
#     uncertainties, computes symmetric mean errors, splits the dataset into
#     training and test sets, and calculates the mean and standard deviation
#     of each input and target variable for normalization.

#     Parameters
#     ----------
#     df : pandas.DataFrame
#         Input DataFrame containing stellar parameters (`Teff`, `logg`, `FeH`, `L`, `M`)
#         and their asymmetric uncertainties (`eX1`, `eX2` for lower/upper errors).

#     Returns
#     -------
#     tuple
#         (mteff, mlogg, mmet, mlum, mtmass, steff, slogg, smet, slum, smass)
#         Mean and standard deviation for each variable, in the order:
#         effective temperature, surface gravity, metallicity, luminosity, and mass.
#     """
#     if logL == True:
#         eL1 = "elogL1"
#         eL2 = "elogL2"
#         L = "logL"
#         eL = "elogL" 
#     else:
#         eL1 = "eL1"
#         eL2 = "eL2"
#         L = "L"
#         eL = "eL"

#     df1 = df[['eTeff1', 'elogg1', 'eFeH1', eL1, 'eM1']].copy()
#     df2 = df[['eTeff2', 'elogg2', 'eFeH2', eL2, 'eM2']].copy()
#     df2.columns = ['eTeff1', 'elogg1', 'eFeH1', eL1, 'eM1']

#     # Mean error if non-symmetric
#     X_error = (df1 + df2) / 2 

#     X_error.columns = ['eTeff', 'elogg', 'eFeH', eL, 'eM']

#     X = pd.concat([df[['Teff', L, 'FeH', 'logg']],
#                    X_error[['eTeff', 'elogg', 'eFeH', eL]]],
#                   axis=1)
#     Y = pd.concat([df['M'], X_error['eM']], axis=1)
    
#     # do split
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
#                                                         test_size=0.2,
#                                                         random_state=RANDOM_SEED)

#     # Extract relevant columns for stellar mass prediction
#     teff = X_train['Teff']
#     logg = X_train['logg']
#     met = X_train['FeH']
#     lum = X_train[L]
#     mass = Y_train["M"]

#     # Compute means and standard deviations for standardization
#     mteff = np.mean(teff)
#     mlogg = np.mean(logg)
#     mmet = np.mean(met)
#     mlum = np.mean(lum)
#     mtmass = np.mean(mass)

#     steff = np.std(teff)
#     slogg = np.std(logg)
#     smet = np.std(met)
#     slum = np.std(lum)
#     smass = np.std(mass)
    
#     return mteff, mlogg, mmet, mlum, mtmass, steff, slogg, smet, slum, smass