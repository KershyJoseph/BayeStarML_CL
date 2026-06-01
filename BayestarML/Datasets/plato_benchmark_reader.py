"""JK 21/05/26
File to read Plato Benchmark Stars from Maxted
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def logomatic_sym(df, var:str):
    """Edit for symmetric errs in linear space

    Add a log(var) column to df with bounds method 
    var should be string key of existing column in df
    """
    df["log"+var] = np.log10(df[var])
    elog1 = np.log10(df[var] + df["e"+var]) - df["log"+var]
    elog2 = df["log"+var] - np.log10(df[var] - df["e"+var])
    df["elog"+var] = (elog1 + elog2)/2 #avg err

    return None

def statomatic(col):
    print("--------",col,"---------")
    print("Min. - ", df_plato_goodMS_new[col].min())
    print("Max. - ", df_plato_goodMS_new[col].max())
    print("Mean - ", df_plato_goodMS_new[col].mean())
    print("Std - ", df_plato_goodMS_new[col].std())

col_specs = [
    (0, 32), #ID
    (33, 35), #component if multiple system
    (65, 68), #Simbad obj type
    (124, 132), #radius
    (133, 140), #radius err
    (203, 210), #logg
    (211, 218), #logg err
    (242, 250), #mass
    (251, 259), #mass err
    (322, 332), #L
    (333, 342), #L err
    (411, 418), #Teff
    (419, 425), #Teff err
    (690, 696), #FeH
    (697, 703), #FeH err
]

col_names = ["ID", "component", "obj_type", "R", "eR", "logg", "elogg", "M", "eM", "L", "eL", "Teff", "eTeff", "FeH", "eFeH"]

plato_df = pd.read_fwf("Datasets/benchmark_stars_20260420.dat", colspecs=col_specs, names=col_names)
print(plato_df)
print("Total stars: ", len(plato_df))

check_params = ["eM", "eR", "elogg", "eL", "eFeH", "eTeff"]

df_plato_all6 = plato_df[(plato_df[check_params].notna().all(axis=1)) & 
                         (plato_df[check_params].gt(0).any(axis=1))]
print("Stars with all 6 params and mass: ", len(df_plato_all6)) #(all that have mass have radius)

#filter for MS stars based on logg and Teff
df_plato_all6_MS = df_plato_all6[((df_plato_all6["Teff"]<=6700) & (df_plato_all6["logg"]>=4.2)) |
                                 ((df_plato_all6["Teff"]>6700) & (df_plato_all6["logg"]>=4.2) & (df_plato_all6["M"]<=2.2))]
print("MS stars with all 6: ", len(df_plato_all6_MS))
#could verify this with a quick HR plot...

df_plato_goodMS = df_plato_all6_MS[["ID", "component", "M", "eM", "R", "eR", "Teff", "eTeff",
                                   "L", "eL", "logg", "elogg", "FeH", "eFeH"]]

#check for duplicates in my database
df_us = pd.read_csv("DataExploring/datos_todos_v20261905.txt", sep="\t", comment="#")
df_p = df_plato_goodMS.copy()

#bunch of edits to make strings match
df_us["ID"] = df_us["ID"].str.replace('_', ' ')
df_us["ID"] = df_us["ID"].str.replace(' A', '')
df_us["ID"] = df_us["ID"].str.replace(' B', '')
df_p["ID"] = df_p["ID"].str.replace(' A', '')
df_p["ID"] = df_p["ID"].str.replace(' B', '')
df_p["ID"] = df_p["ID"].str.replace('V* ', '')
df_p["ID"] = df_p["ID"].str.replace('*  ', '')
df_p["ID"] = df_p["ID"].str.replace('* ', '')
df_p["ID"] = df_p["ID"].str.replace('HD   ', 'HD')
df_p["ID"] = df_p["ID"].str.replace('HD  ', 'HD')
df_p["ID"] = df_p["ID"].str.replace('HD ', 'HD')
df_p["ID"] = df_p["ID"].str.replace('TYC  ', 'TYC ')
df_us["ID"] = df_us["ID"].str.replace('* ', '')

dup_mask = df_p["ID"].isin(df_us["ID"]) #True for matches
df_plato_goodMS_new = df_plato_goodMS[~dup_mask]
print("New plato stars: ", len(df_plato_goodMS_new))
print(df_plato_goodMS_new["ID"].unique())
#print(df_us["ID"])

#add logL col
logomatic_sym(df_plato_goodMS_new, "L")

df_plato_goodMS_new.to_csv("Datasets/plato_data.txt", sep='\t', index=False)

#Compare with my data
feature = "logL"
target = "M"

plt.figure()
lbl = "Our Data"
for df in [df_us, df_plato_goodMS_new]:
    x = df[feature]
    x_err = df["e"+feature]
    y = df[target]
    y_err = df["e"+target]
    plt.errorbar(x, y, y_err, x_err, fmt='o', alpha=0.3, label=lbl)
    lbl = "Plato Data"
plt.legend()
plt.xlabel(feature)
plt.ylabel(target+" ("+target[0]+"sol)")
plt.savefig("Datasets/"+feature+"_"+target+"_Plato_Us.pdf")
plt.close()
