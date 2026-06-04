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
    invalids = (df["e"+var+"2"]>=df[var])
    print(f"< Removing {len(df[invalids])} invalids >")
    df = df[~invalids]
    df["log"+var] = np.log10(df[var])
    df["elog"+var+"1"] = np.log10(df[var] + df["e"+var+"1"]) - df["log"+var]
    df["elog"+var+"2"] = df["log"+var] - np.log10(df[var] - df["e"+var+"2"])
    return df

def err_maskomatic(df):
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
    err_mask = (df_err["elogL"]<=0.05) & (df_err["percent_eR"]<=7) & (df_err["elogg"]<=0.05) & (df_err["eFeH"]<=0.1) & (df_err["percent_eM"]<=7) & (df_err["eTeff"]<=100)

    return df_err, err_mask

#see what spread is like in variables
def spreadomatic(cols):
    for col in cols:
        print("--------",col,"---------")
        print("Min. - ", df_good_RGB[col].min())
        print("Max. - ", df_good_RGB[col].max())
        print("Mean - ", df_good_RGB[col].mean())
        print("Std - ", df_good_RGB[col].std())

        plt.figure()
        sns.histplot(data=df_good_RGB, x=col)
        plt.xlabel(col)
        plt.ylabel("Number of stars")
        plt.savefig("DataExploring/goodRGBspreads/"+col+"_RGB_spread.pdf")
        plt.close()



df = pd.read_csv("DataExploring/datos_todos_v20261905.txt", sep="\t", comment="#")

check_params1 = ["eM1", "eR1", "elogg1", "eL1", "eFeH1", "eTeff1"]
check_params2 = ["eM2", "eR2", "elogg2", "eL2", "eFeH2", "eTeff2"]

df_all6_RGB = df[(df["class"]=="RGB") & 
                (df["well_detached"]!=False) &
                (df[check_params1].notna().all(axis=1)) &
                (df[check_params2].notna().all(axis=1)) &
                (df[check_params1].gt(0).any(axis=1)) &
                (df[check_params2].gt(0).any(axis=1))]

print("All 6 RGB:", len(df_all6_RGB))

#add 50K systematic error in quadrature to all Sch-St Teff errs, as taken from APOGEE https://iopscience.iop.org/article/10.3847/1538-4357/ac4891
old_Teff = df_all6_RGB.loc[df["source"]=="Schonhut-Stasik24", ["eTeff1", "eTeff2"]]
df_all6_RGB.loc[df["source"]=="Schonhut-Stasik24", ["eTeff1", "eTeff2"]] = np.sqrt(old_Teff**2 + 50**2)

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
             yerr=yerr, xerr=xerr, fmt='go', ecolor='gray', alpha=0.4,
             zorder=1)
yerr2 = np.array([df_bad_Ls["L_SB_-err"], df_bad_Ls["L_SB_+err"]])
xerr2 = np.array([df_bad_Ls["eL2"], df_bad_Ls["eL1"]])
plt.errorbar(df_bad_Ls["L"], df_bad_Ls["L_SB"],
             yerr=yerr2, xerr=xerr2, fmt='none', ecolor='red', alpha=0.5,
             zorder=2)
sns.scatterplot(data=df_bad_Ls,x="L",y="L_SB",hue="source",alpha=0.5,zorder=3)
plt.xlabel("L")
plt.ylabel("L from SB")
plt.plot([0, df_L_check["L"].max()], [0,df_L_check["L"].max()], linestyle='--', color='r')
#plt.xscale("log")
#plt.yscale("log")
plt.savefig("DataExploring/random_plots/RGB_L_check.pdf")
plt.close()

print("Non-physical Ls, assuming R and Teff are stellar: ", len(df_bad_Ls))

df_all6_RGB.drop(df_bad_Ls.index, inplace=True)
print("All 6 and physical sense L filtered RGB stars: ", len(df_all6_RGB))

#add logL, logR. logTeff col
df_all6_RGB = logomatic(df_all6_RGB, "L")
df_all6_RGB = logomatic(df_all6_RGB, "R")
df_all6_RGB = logomatic(df_all6_RGB, "Teff")

#remove areas of sparse training data?
df_good_RGB = df_all6_RGB[(df_all6_RGB["M"]<=2.5)]
print("Outliers over 2.5 Msol removed: ", len(df_good_RGB))

#err filering
df_err, err_mask = err_maskomatic(df_good_RGB)
df_good_RGB = df_good_RGB[err_mask]
df_good_errs = df_err[err_mask]
print(f"After error filtering, {len(df_good_RGB)} stars.")

#see err dist
fig, ax = plt.subplots(2,3)

ax[0,0].hist(df_good_errs["percent_eM"], bins='auto')
ax[0,0].vlines(7,0,3200,linestyle='--',color='r',label="7%")
ax[0,0].set_title("M")
ax[0,0].set_ylabel("Number")
ax[0,0].set_xlabel("% Error")
ax[0,0].legend()

ax[0,1].hist(df_good_errs["percent_eR"], bins='auto')
#ax[0,1].vlines(7,0,3500,linestyle='--',color='r',label="7%")
ax[0,1].set_title("R")
ax[0,1].set_xlabel("% Error")
ax[0,1].legend()

ax[0,2].hist(df_good_errs["elogL"], bins='auto')
#ax[0,2].vlines(0.5,0,1000,linestyle='--',color='r',label="0.5")
ax[0,2].set_title("elogL")
ax[0,2].set_xlabel("Error (dex)")
ax[0,2].legend()

ax[1,0].hist(df_good_errs["eTeff"], bins='auto')
ax[1,0].vlines(100,0,1800,linestyle='--',color='r',label="100K")
ax[1,0].set_title("T$_{eff}$") #how to make not italic...
ax[1,0].set_ylabel("Number")
ax[1,0].set_xlabel("Error (K)")
ax[1,0].legend()

ax[1,1].hist(df_good_errs["elogg"], bins='auto')
ax[1,1].vlines(0.05,0,4500,linestyle='--',color='r',label="0.05dex")
ax[1,1].set_title("log(g)")
ax[1,1].set_xlabel("Error (dex)")
ax[1,1].legend()

ax[1,2].hist(df_good_errs["eFeH"], bins='auto')
#ax[1,2].vlines(.15,0,6200,linestyle='--',color='r',label="0.15dex")
ax[1,2].set_title("FeH")
ax[1,2].set_xlabel("Error (dex)")
ax[1,2].legend()

plt.tight_layout()
plt.savefig("DataExploring/db_new_err_distsRGB.pdf")


df_good_RGB=df_good_RGB[df_good_RGB["logTeff"]<3.74]
print(f"Removed that one Yildiz funny - now {len(df_good_RGB)} stars")

spreadomatic(["M", "R", "logR", "logg", "logL", "L", "FeH", "Teff"])

df_all6_RGB.set_index("ID", inplace=True)
df_good_RGB.to_csv("DataExploring/good_RGB.txt", na_rep="NA", sep="\t")
