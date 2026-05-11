"""
JK 22/04/26
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("DataExploring/datos_todos_v20261105.txt", sep="\t", comment="#")

check_params1 = ["eM1", "eR1", "elogg1", "eL1", "eFe/H1", "eTeff1"]
check_params2 = ["eM2", "eR2", "elogg2", "eL2", "eFe/H2", "eTeff2"]
df_all6_MS = df[(df["class"]=="MS") & 
                (df["well_detached"]!=False) &
                (df[check_params1].notna().all(axis=1)) &
                (df[check_params1].gt(0).all(axis=1)) &
                (df[check_params2].notna().all(axis=1)) &
                (df[check_params2].gt(0).all(axis=1))]

#file for MS stars with all 6 params and all binary systems well-detached
df_all6_MS.to_csv("DataExploring/all6_MS.txt", index=False, na_rep="NA", sep="\t")
print("All 6 MS: ", len(df_all6_MS))

#adapted from Max
#get mean errors for non-symmetric ones
df1 = df_all6_MS[
    ['eTeff1', 'elogg1', 'eFe/H1', 'eL1', 'eM1', 'eR1']
    ].copy()
df2 = df_all6_MS[
    ['eTeff2', 'elogg2', 'eFe/H2', 'eL2', 'eM2', 'eR2']
    ].copy()
df2.columns = ['eTeff1', 'elogg1', 'eFe/H1', 'eL1', 'eM1', 'eR1']
df_err = (df1 + df2) / 2
#get percentage errors for M, R, L
df_err["percent_eL"] = 100 * df_err["eL1"] / df_all6_MS["L"]
df_err["percent_eM"] = 100 * df_err["eM1"] / df_all6_MS["M"]
df_err["percent_eR"] = 100 * df_err["eR1"] / df_all6_MS["R"]
df_err["percent_eTeff"] = 100 * df_err["eTeff1"] / df_all6_MS["Teff"]
df_err["percent_elogg"] = 100 * df_err["elogg1"] / df_all6_MS["logg"]
df_err["percent_eFe/H"] = 100 * df_err["eFe/H1"] / df_all6_MS["Fe/H"] #some are /0!

#make mask
err_mask = (df_err["percent_eL"]<=50)# & (df_err["percent_eR"]<=7) & (df_err["elogg1"]<=0.05) & (df_err["percent_eTeff"]<=5) & (df_err["eFe/H1"]<=0.2)

df_good_MS = df_all6_MS[err_mask]
print("All 6 MS, err cleaned: ", len(df_good_MS))
df_good_errs = df_err[err_mask]
df_good_MS.to_csv("DataExploring/good_MS.txt", index=False, na_rep="NA", sep="\t")

#make file with only stars with data that I did not fill via SB or M,R
df_no_fills = df_good_MS[(df_good_MS["L_from_SB"]!=1) &
                         (df_good_MS["logg_from_M,R"]!=1)]
df_no_fills.to_csv("DataExploring/good_MS_no_fills.txt", 
                   index=False, na_rep="NA", sep="\t")
print("All 6 MS, err cleaned, no fills: ", len(df_no_fills))

#Try logL and logTeff columns - added with bounds method
df_good_MS["logTeff"] = np.log10(df_good_MS["Teff"])
df_good_MS["elogTeff1"] = np.log10(df_good_MS["logTeff"] + df_good_MS["eTeff1"]) - df_good_MS["logTeff"]
df_good_MS["elogTeff2"] = df_good_MS["logTeff"] - np.log10(df_good_MS["logTeff"] - df_good_MS["eTeff2"])

df_good_MS["logL"] = np.log10(df_good_MS["L"])
df_good_MS["elogL1"] = np.log10(df_good_MS["logL"] + df_good_MS["eL1"]) - df_good_MS["logL"]
df_good_MS["elogL2"] = df_good_MS["logL"] - np.log10(df_good_MS["logL"] - df_good_MS["eL2"])

#see how logTeff and logL errors look
fig2, ax2 = plt.subplots(1,2)

ax2[0].plot(df_good_MS["elogTeff1"], df_good_MS["elogTeff2"], 'o')
ax2[0].set_xlabel("logTeff +err")
ax2[0].set_ylabel("logTeff -err")

ax2[1].plot(df_good_MS["elogL1"], df_good_MS["elogL2"], 'o')
ax2[1].set_xlabel("logL +err")
ax2[1].set_ylabel("logL -err")

plt.tight_layout()
plt.savefig("DataExploring/logTeff_logL_errs.pdf")

#------------------------------
#see err dist
fig, ax = plt.subplots(2,4)

ax[0,0].hist(df_good_errs["percent_eM"], bins='auto')
ax[0,0].vlines(7,0,150,linestyle='--',color='r',label="7%")
ax[0,0].set_title("M")
ax[0,0].set_ylabel("Number")
ax[0,0].set_xlabel("% Error")
#ax[0,0].legend()

ax[0,1].hist(df_good_errs["percent_eR"], bins='auto')
ax[0,1].vlines(7,0,100,linestyle='--',color='r',label="7%")
ax[0,1].set_title("R")
ax[0,1].set_xlabel("% Error")
#ax[0,1].legend()

ax[0,2].hist(df_good_errs["percent_eL"], bins='auto')
ax[0,2].vlines(10,0,250,linestyle='--',color='r',label="10%")
ax[0,2].set_title("L")
ax[0,2].set_xlabel("% Error")
#ax[0,2].legend()

ax[1,0].hist(df_good_errs["percent_eTeff"], bins='auto')
#ax[1,0].vlines(100,0,220,linestyle='--',color='r',label="100K")
ax[1,0].set_title("T$_{eff}$") #how to make not italic...
ax[1,0].set_ylabel("Number")
ax[1,0].set_xlabel("% Error")
#ax[1,0].legend()

ax[1,1].hist(df_good_errs["elogg1"], bins='auto')
#ax[1,1].vlines(0.05,0,220,linestyle='--',color='r',label="0.05dex")
ax[1,1].set_title("log(g)")
ax[1,1].set_xlabel("Error (dex)")
#ax[1,1].legend()

ax[1,2].hist(df_good_errs["eFe/H1"], bins='auto')
#ax[1,2].vlines(.15,0,250,linestyle='--',color='r',label="0.15dex")
ax[1,2].set_title("Fe/H")
ax[1,2].set_xlabel("Error (dex)")
#ax[1,2].legend()

# ax[0,3].hist(df_good_errs["elogL1"], bins='auto')
# #ax[0,3].vlines(10,0,250,linestyle='--',color='r',label="10%")
# ax[0,3].set_title("logL")
# ax[0,3].set_xlabel("Error (logLsol)")
# #ax[0,3].legend()

# ax[1,3].hist(df_good_errs["elogTeff1"], bins='auto')
# #ax[1,3].vlines(100,0,220,linestyle='--',color='r',label="100K")
# ax[1,3].set_title("logT$_{eff}$") #how to make not italic...
# ax[1,3].set_xlabel("Error (logK)")
# #ax[1,3].legend()

plt.tight_layout()
plt.savefig("DataExploring/db_new_err_dists.pdf")


#consistency checks...
#---------------------
#get SB Ls and errs
df_L_check = df_good_MS[df_good_MS["L_from_SB"]==0]
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
#total_err = np.sqrt(df)

plt.figure()
yerr = np.array([df_L_check["L_SB_-err"], df_L_check["L_SB_+err"]])
xerr = np.array([df_L_check["eL2"], df_L_check["eL1"]])
plt.errorbar(df_L_check["L"], df_L_check["L_SB"], #x,y,yerr,xerr
             yerr=yerr, xerr=xerr, fmt='bo', ecolor='gray', alpha=0.5)
plt.xlabel("L")
plt.ylabel("L from SB")
plt.plot([0, df_L_check["L"].max()], [0,df_L_check["L"].max()], linestyle='--', color='r')
plt.xscale("log")
plt.yscale("log")
plt.savefig("DataExploring/MS_L_check.pdf")



def make_MS_sample(N):
    """Cheeky tiny MS sample to practice on. N is roughly how many stars.
    """
    sample_list = np.arange(0, len(df_good_MS), int(len(df_good_MS)/N))
    df_MS_sample = df_good_MS.iloc[sample_list]

    df_MS_sample.to_csv("DataExploring/MS_sample_"+str(len(df_MS_sample))+".txt",
                        index=False, na_rep="NA", sep="\t")

def diagnostics(df, name):
    print("Diagnostics on ", name)

    print("Old stars: ", len(df[(df["database"]==1)]))

    print("New stars: ", len(df[(df["database"]!=1)]), "out of ", len(df))

    print("New range stars: ", len(df[(df["M"]<=0.8) | (df["M"]>=1.4)]), "out of ", len(df))

    print("New stars AND new range stars: ", len(df[((df["M"]<=0.8) | (df["M"]>=1.4)) & (df["database"]!=1)]), "out of ", len(df))


