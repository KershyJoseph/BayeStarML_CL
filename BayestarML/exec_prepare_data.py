"""Executable to prepare a given database file for training on the BHS model
"""

import pandas as pd
from preprocess import select_clean_data, error_filter

#choose database to select data from for training
filename = "datos_todos_v20261905.txt"
df = pd.read_csv("data/"+filename, sep='\t', comment="#")

#make sure all column names match what BHS expects

#choose which stellar class to train on, and which target and training features. Add columns for log10 value of some variables, and average symmetrical errors for all.
training_fs = ["Teff", "logg", "FeH", "L"]
targets = ["M", "R"]
s_class = "MS"
add_logvars = ["Teff", "L"]
df = select_clean_data(df, training_fs, targets, s_class, add_logvars)

#filter by error limits and plot error distributions
abs_err_tols = {
    "elogL": 0.5
}
percent_err_tols = {
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
                  abs_err_tols=abs_err_tols,
                  percent_err_tols=percent_err_tols,
                  plot_params=plot_params)


