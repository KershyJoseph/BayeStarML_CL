"""
JK 22/04/26
"""
import pandas as pd
import numpy as np

df = pd.read_csv("Datasets/datos_todos_20260417.txt", sep="\t", comment="#")

#cheeky tiny MS sample to practice on
check_params = ["eM1", "eR1", "elogg1", "eL1", "eFe/H1", "eTeff1"]
df_good_MS = df[(df["class"]=="MS") & 
                (df["well_detached"]==True) &
                (df[check_params].notna().all(axis=1))]

sample_list = np.arange(0, len(df_good_MS), int(len(df_good_MS)/70))
df_MS_sample = df_good_MS.iloc[sample_list]

df_MS_sample.to_csv("Datasets/MS_sample.txt", index=False, na_rep="NA", sep="\t")
