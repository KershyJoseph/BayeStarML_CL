"""
JK 22/04/26
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("DataExploring/datos_todos_v20260505.txt", sep="\t", comment="#")

check_params = ["eM1", "eR1", "elogg1", "eL1", "eFe/H1", "eTeff1"]
df_all6_RGB = df[(df["class"]=="RGB") & 
                (df["well_detached"]!=False) &
                (df[check_params].notna().all(axis=1))]

df_all6_RGB.to_csv("DataExploring/all6_RGB.txt", index=False, na_rep="NA", sep="\t")

#adapted from Max
#get mean errors for non-symmetric ones
df1 = df_all6_RGB[['eTeff1', 'elogg1', 'eFe/H1', 'eL1', 'eM1', 'eR1']].copy()
df2 = df_all6_RGB[['eTeff2', 'elogg2', 'eFe/H2', 'eL2', 'eM2', 'eR2']].copy()
df2.columns = ['eTeff1', 'elogg1', 'eFe/H1', 'eL1', 'eM1', 'eR1']
df_err = (df1 + df2) / 2
#get percentage errors for M, R, L
df_err["percent_eL"] = 100 * df_err["eL1"] / df_all6_RGB["L"]
df_err["percent_eM"] = 100 * df_err["eM1"] / df_all6_RGB["M"]
df_err["percent_eR"] = 100 * df_err["eR1"] / df_all6_RGB["R"]
df_err["percent_eTeff"] = 100 * df_err["eTeff1"] / df_all6_RGB["Teff"]
#make mask
err_mask = (df_err["percent_eL"]<=1000) & (df_err["percent_eR"]<=25)# & (df_err["percent_eM"]<=7) & (df_err["eTeff1"]<=100) & (df_err["elogg1"]<=0.05) & (df_err["eFe/H1"]<=0.15)


df_good_RGB = df_all6_RGB[err_mask]
df_good_errs = df_err[err_mask]
df_good_RGB.to_csv("DataExploring/good_RGB.txt", index=False, na_rep="NA", sep="\t")

#see err dist
fig, ax = plt.subplots(2,3)

ax[0,0].hist(df_good_errs["percent_eM"], bins='auto')
ax[0,0].vlines(7,0,4200,linestyle='--',color='r',label="7%")
ax[0,0].set_title("M")
ax[0,0].set_ylabel("Number")
ax[0,0].set_xlabel("% Error")
ax[0,0].legend()

ax[0,1].hist(df_good_errs["percent_eR"], bins='auto')
ax[0,1].vlines(7,0,3500,linestyle='--',color='r',label="7%")
ax[0,1].set_title("R")
ax[0,1].set_xlabel("% Error")
ax[0,1].legend()

ax[0,2].hist(df_good_errs["percent_eL"], bins='auto')
ax[0,2].vlines(10,0,3000,linestyle='--',color='r',label="10%")
ax[0,2].set_title("L")
ax[0,2].set_xlabel("% Error")
ax[0,2].legend()

ax[1,0].hist(df_good_errs["eTeff1"], bins='auto')
ax[1,0].vlines(100,0,1800,linestyle='--',color='r',label="100K")
ax[1,0].set_title("T$_{eff}$") #how to make not italic...
ax[1,0].set_ylabel("Number")
ax[1,0].set_xlabel("Error (K)")
ax[1,0].legend()

ax[1,1].hist(df_good_errs["elogg1"], bins='auto')
ax[1,1].vlines(0.05,0,4500,linestyle='--',color='r',label="0.05dex")
ax[1,1].set_title("log(g)")
ax[1,1].set_xlabel("Error (dex)")
ax[1,1].legend()

ax[1,2].hist(df_good_errs["eFe/H1"], bins='auto')
ax[1,2].vlines(.15,0,6200,linestyle='--',color='r',label="0.15dex")
ax[1,2].set_title("Fe/H")
ax[1,2].set_xlabel("Error (dex)")
ax[1,2].legend()

plt.tight_layout()
plt.savefig("DataExploring/db_new_err_distsRGB.pdf")

#consistency checks...
df_L_check = df[df["L_from_SB"]==0]
df_L_check["L_SB"] = 
