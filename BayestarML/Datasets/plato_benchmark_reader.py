"""JK 21/05/26
File to read Plato Benchmark Stars from Maxted
"""
import pandas as pd

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
    (690, 696), #Fe/H
    (697, 703), #Fe/H err
]

col_names = ["ID", "component", "obj_type", "R", "eR", "logg", "elogg", "M", "eM", "L", "eL", "Teff", "eTeff", "Fe/H", "eFe/H"]

plato_df = pd.read_fwf("Datasets/benchmark_stars_20260420.dat", colspecs=col_specs, names=col_names)
print(plato_df)
print("Total stars: ", len(plato_df))

check_params = ["eM", "eR", "elogg", "eL", "eFe/H", "eTeff"]

df_plato_all6 = plato_df[(plato_df[check_params].notna().all(axis=1)) & 
                         (plato_df[check_params].gt(0).any(axis=1))]
print("Stars with all 6 params and mass: ", len(df_plato_all6)) #(all that have mass have radius)

#filter for MS stars based on logg and Teff
df_plato_all6_MS = df_plato_all6[((df_plato_all6["Teff"]<=6700) & (df_plato_all6["logg"]>=4.2)) |
                                 ((df_plato_all6["Teff"]>6700) & (df_plato_all6["logg"]>=4.2) & (df_plato_all6["M"]<=2.2))]
print("MS stars with all 6: ", len(df_plato_all6_MS))
#could verify this with a quick HR plot...

df_plato_goodMS = df_plato_all6_MS[["ID", "component", "M", "eM", "R", "eR", "Teff", "eTeff",
                                   "L", "eL", "logg", "elogg", "Fe/H", "eFe/H"]]

#check for duplicates in my database
df_us = pd.read_csv("DataExploring/datos_todos_v20261905.txt", sep="\t", comment="#")
df_us["ID"] = df_us["ID"].str.replace('_', ' ')
df_us["ID"] = df_us["ID"].str.replace(' A', '')
df_us["ID"] = df_us["ID"].str.replace(' B', '')
df_plato_goodMS["ID"] = df_plato_goodMS["ID"].str.replace(' A', '')
df_plato_goodMS["ID"] = df_plato_goodMS["ID"].str.replace(' B', '')
df_plato_goodMS["ID"] = df_plato_goodMS["ID"].str.replace('V* ', '')
df_plato_goodMS["ID"] = df_plato_goodMS["ID"].str.replace('*  ', '')
df_plato_goodMS["ID"] = df_plato_goodMS["ID"].str.replace('* ', '')
df_us["ID"] = df_us["ID"].str.replace('* ', '')

dup_mask = df_plato_goodMS["ID"].isin(df_us["ID"]) #True for matches
df_plato_goodMS_new = df_plato_goodMS[~dup_mask]
print("New plato stars: ", len(df_plato_goodMS_new))
print(df_plato_goodMS_new["ID"])
#print(df_us["ID"])

df_plato_goodMS_new.to_csv("Datasets/plato_data.txt", sep='\t')
