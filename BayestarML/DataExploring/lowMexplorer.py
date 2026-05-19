"""
JK 14/05/26
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_all = pd.read_csv("DataExploring/datos_todos_v20261905.txt", sep="\t", comment="#")

#diagnostics
df_low = df_all[(df_all["M"]<0.8) & (df_all["class"]=="MS")]
print(f"There are {len(df_low)} MS stars below 0.8M.")

df_low_d = df_low[(df_low["well_detached"]!=False)]
print(f"Of which {len(df_low_d)} are well-detached ({len(df_low_d[df_low_d["database"]==3])} are new and {len(df_low_d[df_low_d["database"]==2])} are modified).")

MRL = ["eM1", "eM2", "eL1", "eL2", "eR1", "eR2"]
df_low_d_3 = df_low_d[df_low_d[MRL].notna().all(axis=1)]
print(f"Of which {len(df_low_d_3)} have mass, radius and L with errors.")

df_low_d_4 = df_low_d_3[df_low_d_3["Fe/H"].notna()]
df_low_d_MH = df_low_d_3[(df_low_d_3["Fe/H"].isna()) & (df_low_d_3["M/H"].notna())]
print(f"Of which {len(df_low_d_4)} have Fe/H and {len(df_low_d_MH)} don't have Fe/H but have M/H.")

df_low_d_5 = df_low_d_4[df_low_d_4["Teff"].notna()]
print(f"Of which {len(df_low_d_5)} have Teff.")

df_low_d_6 = df_low_d_5[df_low_d_4["logg"].notna()]
print(f"Of which {len(df_low_d_6)} have logg.")

df_low_d_6["percent_eL"] = 100 * df_low_d_6["eL1"] + df_low_d_6["eL2"] / (2 * df_low_d_6["L"])
df_low_d_6_Lerr = df_low_d_6[df_low_d_6["percent_eL"]<=50]
print(f"Of which {len(df_low_d_6_Lerr)} have avg L err less than or equal to 50%.")

TLF = ["eTeff1", "eTeff2", "elogg1", "elogg2", "eFe/H1", "eFe/H2"]
all6 = [*MRL, *TLF]
df_no_nas = df_low_d_6_Lerr[df_low_d_6_Lerr[all6].notna().all(axis=1)]
print(f"Of which {len(df_no_nas)} don't have NA err values.")

df_new_good = df_no_nas[df_no_nas["database"]!=1]
print(f"Of which {len(df_new_good)} are new/revised.")

with pd.option_context("display.max_rows", None, "display.max_columns", None):
    print(df_new_good)
