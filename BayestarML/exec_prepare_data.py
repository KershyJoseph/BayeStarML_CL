"""Executable to prepare a given database file for training on the BHS model
"""

import pandas as pd
from preprocess import select_clean_data, error_filter, spreadomatic, return_train_test

#choose database to select data from for training
filename = "datos_todos_v20261905.txt"
df = pd.read_csv("data/"+filename, sep='\t', comment="#")

#make sure all column names match what BHS expects

#choose which stellar class to train on, and which target and training features. Add columns for log10 value of some variables, and average symmetrical errors for all.
training_fs = ["Teff", "logg", "FeH", "L"]
targets = ["M", "R"]
s_class = "MS"
add_logvars = ["Teff", "L"]
df = select_clean_data(df, training_fs, targets, s_class, add_logvars, L_check=False)
#re-write training and targets in case of log switch
training_fs = ["Teff", "logg", "FeH", "logL"]
targets = ["M", "R"]

#filter by error limits and plot error distributions
abs_err_lims = {
    "elogL": 0.5
}
percent_err_lims = {
    "eM": 100,
    "eR": 100
}
plot_params = {
    "M" : [7, "%"],
    "R" : [7, "%"],
    "logL" : [0.05, "dex"],
    "Teff" : [100, "K"],
    "logg" : [0.05, "dex"],
    "FeH" : [0.15, "dex"]
}
df = error_filter(df,
                  abs_err_lims=abs_err_lims,
                  percent_err_lims=percent_err_lims,
                  plot_params=None)

#look at spread in target variables and decide whether or not to cut training range
target_lim = 2
df = df[df["M"]<=target_lim]
if target_lim == None:
    for var in targets:
        repeat=True
        spreadomatic(df, var)
        while repeat:
            lim0 = float(input(f"What lower limit do you want to put on {var}?\n"))
            lim1 = float(input(f"What upper limit do you want to put on {var}?\n"))
            df = df[(df[var]>=lim0) & (df[var]<=lim1)]
            print(f"{len(df)} stars left after trimming {var} range.")
            spreadomatic(df, var)
            repeat = input(f"Continue checking {var} spread? (yes/no)\n") == "yes"

#get MUs and SIGs for normalisation of each param, and write to constants.py, as well as MIN and MAX
return_train_test(df, training_fs, targets)