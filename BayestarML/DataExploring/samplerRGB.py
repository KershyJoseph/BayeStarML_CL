"""
JK 22/04/26
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def logomatic(df, var):
    """Add a log(var) column to df with bounds method 
    var should be string key of existing column in df
    """
    df["log"+var] = np.log10(df[var])
    df["elog"+var+"1"] = np.log10(df[var] + df["e"+var+"1"]) - df["log"+var]
    df["elog"+var+"2"] = df["log"+var] - np.log10(df[var] - df["e"+var+"2"])
    return None

def err_maskomatic(df, L=.5):
    """Return an error mask on a df with given limits. And df of average errors.
    L - percent L err limit
    """
    #adapted from Max
    #get mean errors for non-symmetric ones
    df1 = df[
        ['eTeff1', 'elogg1', 'eFeH1', 'eL1', 'elogL1', 'eM1', 'eR1', 'elogR1']
        ].copy()
    df2 = df[
        ['eTeff2', 'elogg2', 'eFeH2', 'eL2', 'elogL2', 'eM2', 'eR2', 'elogR2']
        ].copy()
    df1.columns = ['eTeff', 'elogg', 'eFeH', 'eL', 'elogL', 'eM', 'eR', 'elogR']
    df2.columns = ['eTeff', 'elogg', 'eFeH', 'eL', 'elogL', 'eM', 'eR', 'elogR']
    df_err = (df1 + df2) / 2
    #get percentage errors for M, R, L
    df_err["percent_eL"] = 100 * df_err["eL"] / df["L"]
    df_err["percent_eM"] = 100 * df_err["eM"] / df["M"]
    df_err["percent_eR"] = 100 * df_err["eR"] / df["R"]
    df_err["percent_eTeff"] = 100 * df_err["eTeff"] / df["Teff"]

    #make mask
    err_mask = (df_err["elogL"]<=L)# & (df_err["percent_eR"]<=7) & (df_err["elogg1"]<=0.05) & (df_err["percent_eTeff"]<=5) & (df_err["eFeH1"]<=0.2)

    return df_err, err_mask

df = pd.read_csv("DataExploring/datos_todos_v20261905.txt", sep="\t", comment="#")

check_params1 = ["eM1", "eR1", "elogg1", "eL1", "eFeH1", "eTeff1"]
check_params2 = ["eM2", "eR2", "elogg2", "eL2", "eFeH2", "eTeff2"]

df_all6_RGB = df[(df["class"]=="RGB") & 
                (df["well_detached"]!=False) &
                (df[check_params1].notna().all(axis=1)) &
                (df[check_params2].notna().all(axis=1)) &
                (df[check_params1].gt(0).any(axis=1)) &
                (df[check_params2].gt(0).any(axis=1))]

df_all6_RGB.to_csv("DataExploring/all6_RGB.txt", index=False, na_rep="NA", sep="\t")
print("All 6 RGB:", len(df_all6_RGB))

#consistency checks...
#get SB Ls and errs
df_L_check = df_all6_RGB[df_all6_RGB["L_from_SB"]==0]
df_L_check["L_SB"] = df_L_check["R"]**2 * (df_L_check["Teff"]/5772)**4

R = df_L_check["R"]
Teff = df_L_check["Teff"]

df_L_check["L_SB_+err"] = np.sqrt(
    (R**2*((Teff+df_L_check["eTeff1"])/5772)**4 - df_L_check["L_SB"])**2 
    + ((R+df_L_check["eR1"])**2*(Teff/5772)**4 - df_L_check["L_SB"])**2 
)
df_L_check["L_SB_-err"] = np.sqrt(
    (R**2*((Teff-df_L_check["eTeff2"])/5772)**4 - df_L_check["L_SB"])**2 
    + ((R-df_L_check["eR2"])**2*(Teff/5772)**4 - df_L_check["L_SB"])**2 
)

#compute distance from recorded Ls
df_L_check["L_SB_avg_err"] = (df_L_check["L_SB_+err"] + df_L_check["L_SB_-err"])/2
df_L_check["total_L_err"] = np.sqrt(df_L_check["L_SB_avg_err"]**2 + df_L_check["eL1"]**2)
df_L_check["L_dist"] = df_L_check["L_SB"]-df_L_check["L"]
df_L_check["L_sig_distance"] = np.abs(df_L_check["L_dist"])/df_L_check["total_L_err"]
df_bad_Ls = df_L_check[df_L_check["L_sig_distance"]>3]

plt.figure()
yerr = np.array([df_L_check["L_SB_-err"], df_L_check["L_SB_+err"]])
xerr = np.array([df_L_check["eL2"], df_L_check["eL1"]])
plt.errorbar(df_L_check["L"], df_L_check["L_SB"], #x,y,yerr,xerr
             yerr=yerr, xerr=xerr, fmt='bo', ecolor='gray', alpha=0.5)
yerr2 = np.array([df_bad_Ls["L_SB_-err"], df_bad_Ls["L_SB_+err"]])
xerr2 = np.array([df_bad_Ls["eL2"], df_bad_Ls["eL1"]])
plt.errorbar(df_bad_Ls["L"], df_bad_Ls["L_SB"],
             yerr=yerr2, xerr=xerr2, fmt='ro', ecolor='orange', alpha=0.5)
plt.xlabel("L")
plt.ylabel("L from SB")
plt.plot([0, df_L_check["L"].max()], [0,df_L_check["L"].max()], linestyle='--', color='r')
#plt.xscale("log")
#plt.yscale("log")
plt.savefig("DataExploring/RGB_L_check.pdf")

print("Non-physical Ls, assuming R and Teff are stellar: ", len(df_bad_Ls))

df_all6_RGB.drop(df_bad_Ls.index, inplace=True)
print("Error filtered and physical sense L filtered RGB stars: ", len(df_good_RGB))

#add logL col
df_good_RGB["logL"] = np.log10(df_good_RGB["L"])
df_good_RGB["elogL1"] = np.log10(df_good_RGB["L"] + df_good_RGB["eL1"]) - df_good_RGB["logL"]
df_good_RGB["elogL2"] = df_good_RGB["logL"] - np.log10(df_good_RGB["L"] - df_good_RGB["eL2"])

#add logR col
df_good_RGB["logR"] = np.log10(df_good_RGB["R"])
df_good_RGB["elogR1"] = np.log10(df_good_RGB["R"] + df_good_RGB["eR1"]) - df_good_RGB["logR"]
df_good_RGB["elogR2"] = df_good_RGB["logR"] - np.log10(df_good_RGB["R"] - df_good_RGB["eR2"])

#remove areas of sparse training data?
df_good_RGB = df_good_RGB[(df_good_RGB["M"]<=2.5)]
print("RGB stars with outliers over 2.5 Msol removed: ", len(df_good_RGB))

#see what spread is like in variables
for col in ["M", "R", "logR", "logg", "logL", "L", "FeH", "Teff"]:
    print("--------",col,"---------")
    print("Min. - ", df_good_RGB[col].min())
    print("Max. - ", df_good_RGB[col].max())
    print("Mean - ", df_good_RGB[col].mean())
    print("Std - ", df_good_RGB[col].std())

    plt.figure()
    sns.histplot(data=df_good_RGB, x=col)
    plt.xlabel(col)
    plt.ylabel("Number of stars")
    plt.savefig("DataExploring/"+col+"_RGB_spread.pdf")
    plt.close()

#err filering

#see err dist
fig, ax = plt.subplots(2,3)

ax[0,0].hist(df_good_errs["percent_eM"], bins='auto')
ax[0,0].vlines(7,0,3200,linestyle='--',color='r',label="7%")
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
ax[0,2].vlines(10,0,1000,linestyle='--',color='r',label="10%")
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

ax[1,2].hist(df_good_errs["eFeH1"], bins='auto')
ax[1,2].vlines(.15,0,6200,linestyle='--',color='r',label="0.15dex")
ax[1,2].set_title("FeH")
ax[1,2].set_xlabel("Error (dex)")
ax[1,2].legend()

plt.tight_layout()
plt.savefig("DataExploring/db_new_err_distsRGB.pdf")


df_good_RGB.to_csv("DataExploring/good_RGB.txt", index=False, na_rep="NA", sep="\t")
