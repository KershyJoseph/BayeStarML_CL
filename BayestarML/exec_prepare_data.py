"""Executable to prepare a given database file for training on the BHS model
"""

import pandas as pd
from preprocess import select_clean_data

#choose database to select data from for training
filename = "datos_todos_v20261905.txt"
df = pd.read_csv("data/"+filename, sep='\t', comment="#")

#make sure all column names match what BHS expects

#choose which stellar class to train on, and which target and training features. Add average symmetrical errors.
training_fs = ["Teff", "logg", "FeH", "L"]
targets = ["M", "R"]
s_class = "MS"
add_logvars = ["Teff", "L"]
df = select_clean_data(df, training_fs, targets, s_class, add_logvars)

with pd.option_context("display.max_columns", None):
    print(df)

#compute average symmetrical error for each param and filter by error limits