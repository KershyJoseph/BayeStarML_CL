"""
See dist of params in old set
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def spreadomatic(df, var, save_path, hue=None, xlabel=None):
    """Make a histogram for a given var (which should be one of df's keys)
    """
    plt.figure()
    sns.histplot(data=df, x=var, hue=hue)
    if xlabel:
        plt.xlabel(xlabel)
    plt.ylabel("Number of stars")
    plt.savefig(save_path)

def logomatic(df, var):
    """Add a log(var) column to df with bounds method 
    var should be string key of existing column in df
    """
    invalids = (df["e"+var+"2"]>=df[var])
    print(f"< Removing {len(df[invalids])} invalids >")
    df = df[~invalids]
    df["log"+var] = np.log10(df[var])
    df["elog"+var+"1"] = np.log10(df[var] + df["e"+var+"1"]) - df["log"+var]
    df["elog"+var+"2"] = df["log"+var] - np.log10(df[var] - df["e"+var+"2"])
    return df

df = pd.read_csv("Datasets/datos_tot_v20180517_adapted.txt", sep="\t")
print(len(df))
df_all6 = df[(df[["M", "R", "Teff", "L", "FeH", "logg"]].notna().all(axis=1))]
print(len(df_all6))

df_all6 = logomatic(df_all6, "Teff")
df_all6 = logomatic(df_all6, "L")
df_all6.to_csv("Datasets/all6_2018_data.txt", sep="\t", index=False, na_rep="NA")

for var in ["M", "R"]:
    spreadomatic(df_all6, var, "Datasets/2018spreads/"+var+"_2018spread.pdf")
