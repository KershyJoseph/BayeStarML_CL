"""
JK 22/04/26
"""
import pandas as pd
import numpy as np

df = pd.read_csv("DataExploring/datos_todos_20260417.txt", sep="\t", comment="#")

check_params = ["eM1", "eR1", "elogg1", "eL1", "eFe/H1", "eTeff1"]
df_good_MS = df[(df["class"]=="MS") & 
                (df["well_detached"]!=False) &
                (df[check_params].notna().all(axis=1))]

df_good_MS.to_csv("DataExploring/good_MS.txt", index=False, na_rep="NA", sep="\t")

def make_MS_sample(N):
    """Cheeky tiny MS sample to practice on. N is roughly how many stars.
    """
    sample_list = np.arange(0, len(df_good_MS), int(len(df_good_MS)/N))
    df_MS_sample = df_good_MS.iloc[sample_list]

    df_MS_sample.to_csv("DataExploring/MS_sample_"+str(len(df_MS_sample))+".txt",
                        index=False, na_rep="NA", sep="\t")

make_MS_sample(30)
