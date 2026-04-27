"""
JK 22/04/26
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("datos_todos_20260417.txt", sep="\t", comment="#")

check_params = ["eM1", "eR1", "elogg1", "eL1", "eFe/H1", "eTeff1"]
df_good_MS = df[(df["class"]=="MS") & 
                (df["well_detached"]!=False) &
                (df[check_params].notna().all(axis=1))]

def make_MS_sample(N):
    """Cheeky tiny MS sample to practice on. N is roughly how many stars.
    """
    sample_list = np.arange(0, len(df_good_MS), int(len(df_good_MS)/N))
    df_MS_sample = df_good_MS.iloc[sample_list]

    df_MS_sample.to_csv("MS_sample_"+str(len(df_MS_sample))+".txt",
                        index=False, na_rep="NA", sep="\t")

#histogram of types
types = df_good_MS["type"].str[0]
df_EBs = df_good_MS[(df_good_MS["mode"]=="EB") | (df_good_MS["mode"]=="TEB")]
types_EBs = df_EBs["type"].str[0]
x = pd.Series(types).value_counts(dropna=False)
print(x["type"])
pd.Series(types_EBs).value_counts(dropna=False)
# plt.xlabel("Spectral Type")
# plt.ylabel("Frequency")

