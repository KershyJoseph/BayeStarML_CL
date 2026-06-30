"""JK 03/06/2026
Convert NASA exoplanet archive data into format used for Bayestar
"""
import pandas as pd
import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt

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

df = pd.read_csv("BayestarML/predict/prediction_datasets/PS_2026.06.03_04.08.32.tab", sep="\t", comment="#")
print(f"Starting number of stars: {len(df)}")

col_names_change = {
    "st_teff": "Teff",
    "st_tefferr1": "eTeff1",
    "st_tefferr2": "eTeff2",
    "st_rad": "R",
    "st_raderr1": "eR1",
    "st_raderr2": "eR2",
    "st_mass": "M",
    "st_masserr1": "eM1",
    "st_masserr2": "eM2",
    "st_met": "FeH",
    "st_meterr1": "eFeH1",
    "st_meterr2": "eFeH2",
    "st_lum": "logL",
    "st_lumerr1": "elogL1",
    "st_lumerr2": "elogL2",
    "st_logg": "logg",
    "st_loggerr1": "elogg1",
    "st_loggerr2": "elogg2"
}
df.rename(columns=col_names_change, inplace=True)

#make any -ve errs +ve
errs = ['eTeff1', 'elogg1', 'eFeH1', 'elogL1', 'eM1', 'eR1', 'eTeff2', 'elogg2', 'eFeH2', 'elogL2', 'eM2', 'eR2']
df[errs] = df[errs].abs()

#creat df_all6: only stars with no limits flag, and all 6 params with errs
params = ["M", "R", "logg", "logL", "FeH", "Teff"]
limits_flag = ["st_tefflim", "st_radlim", "st_masslim", "st_metlim", "st_lumlim", "st_logglim"]
df_all6 = df[((df[limits_flag]==0).all(axis=1)) &
             (df[errs].notna().all(axis=1)) &
             (df[params].notna().all(axis=1))]
print(f"All 6 stars: {len(df_all6)}")

df_all6 = logomatic(df_all6, "Teff")

#remove duplicates! (Same star for many planets)
df_all6.drop_duplicates(inplace=True, subset=["Teff", "R"]) #subset anything but row id

#add MS/RG distinction 
rg_vertices = np.array([
    [3.8, 1.4],
    [3.7, 4.0],
    [3.5, 3.0],
    [3.65, -0.5],
    [3.67, -0.3],
])
rg_mesh = Path(rg_vertices)
hr_points = df_all6[["logTeff", "logL"]].values
rg_mask = rg_mesh.contains_points(hr_points)

x = df_all6["logTeff"]
y = df_all6["logL"]
plt.plot(x, y, 'x')
plt.plot(x[rg_mask], y[rg_mask], 'o', alpha=0.5)
plt.gca().xaxis.set_inverted(True)
#plt.show()

print(str(len(df_all6))+" stars total")
print(str(len(df_all6[~rg_mask]))+" ms stars")
print(str(len(df_all6[rg_mask]))+" rgb stars")

df_all6 = df_all6[~rg_mask]

df_all6.to_csv("BayestarML/predict/prediction_datasets/NASAexop_archive_stars_all6ms.txt", sep='\t', na_rep='NA', index=False)

