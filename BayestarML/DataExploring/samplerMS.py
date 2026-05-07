"""
JK 22/04/26
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("DataExploring/datos_todos_v20260505.txt", sep="\t", comment="#")

check_params = ["eM1", "eR1", "elogg1", "eL1", "eFe/H1", "eTeff1"]
df_all6_MS = df[(df["class"]=="MS") & 
                (df["well_detached"]!=False) &
                (df[check_params].notna().all(axis=1))]

#file for MS stars with all 6 params and all binary systems well-detached
df_all6_MS.to_csv("DataExploring/all6_MS.txt", index=False, na_rep="NA", sep="\t")
print("All 6 MS: ", len(df_all6_MS))

#adapted from Max
#get mean errors for non-symmetric ones
df1 = df_all6_MS[['eTeff1', 'elogg1', 'eFe/H1', 'eL1', 'eM1', 'eR1']].copy()
df2 = df_all6_MS[['eTeff2', 'elogg2', 'eFe/H2', 'eL2', 'eM2', 'eR2']].copy()
df2.columns = ['eTeff1', 'elogg1', 'eFe/H1', 'eL1', 'eM1', 'eR1']
df_err = (df1 + df2) / 2
#get percentage errors for M, R, L
df_err["percent_eL"] = 100 * df_err["eL1"] / df_all6_MS["L"]
df_err["percent_eM"] = 100 * df_err["eM1"] / df_all6_MS["M"]
df_err["percent_eR"] = 100 * df_err["eR1"] / df_all6_MS["R"]
df_err["percent_eTeff"] = 100 * df_err["eTeff1"] / df_all6_MS["Teff"]
#make mask
err_mask = (df_err["percent_eL"]<=50) #& (df_err["percent_eR"]<=25)# & (df_err["percent_eM"]<=7) & (df_err["eTeff1"]<=100) & (df_err["elogg1"]<=0.05) & (df_err["eFe/H1"]<=0.15)


df_good_MS = df_all6_MS[err_mask]
print("All 6 MS, err cleaned: ", len(df_good_MS))
df_good_errs = df_err[err_mask]
df_good_MS.to_csv("DataExploring/good_MS.txt", index=False, na_rep="NA", sep="\t")

#see err dist
fig, ax = plt.subplots(2,3)
ax[0,0].hist(df_good_errs["percent_eL"], bins='fd')
ax[0,0].set_title("L percent err")

ax[0,1].hist(df_good_errs["percent_eR"], bins='fd')
ax[0,1].set_title("R percent err")

ax[0,2].hist(df_good_errs["percent_eM"], bins='fd')
ax[0,2].set_title("M percent err")

ax[1,0].hist(df_good_errs["eTeff1"], bins='fd')
ax[1,0].set_title("Teff err")

ax[1,1].hist(df_good_errs["elogg1"], bins='fd')
ax[1,1].set_title("logg err")

ax[1,2].hist(df_good_errs["eFe/H1"], bins='fd')
ax[1,2].set_title("Fe/H err")

plt.tight_layout()
plt.savefig("DataExploring/db_new_err_dists.png")

def make_MS_sample(N):
    """Cheeky tiny MS sample to practice on. N is roughly how many stars.
    """
    sample_list = np.arange(0, len(df_good_MS), int(len(df_good_MS)/N))
    df_MS_sample = df_good_MS.iloc[sample_list]

    df_MS_sample.to_csv("DataExploring/MS_sample_"+str(len(df_MS_sample))+".txt",
                        index=False, na_rep="NA", sep="\t")
