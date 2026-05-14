"""
JK 14/05/26
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_all = pd.read_csv("DataExploring/datos_todos_v20261105.txt", sep="\t", comment="#")

#diagnostics
df_low = df_all[(df_all["M"]<0.8) & (df_all["class"]=="MS")]
print(f"There are {len(df_low)} MS stars below 0.8M.")

df_low_d = df_low[(df_low["well_detached"]!=False)]
print(f"Of which {len(df_low_d)} are well-detached.")

MRL = ["eM1", "eM2", "eL1", "eL2", "eR1", "eR2"]
df_low_d_3 = df_low_d[df_low_d[MRL].notna().all(axis=1)]
print(f"Of which {len(df_low_d_3)} have mass, radius and L with errors.")