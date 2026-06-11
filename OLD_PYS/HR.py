"""JK 03/06/2026
Plot HR diagram for different dfs
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def HRplot(df,savename:str,hue:str=None):
    """Plot logTeff (higher Teff to the left) against logL for stars in df
    """
    x=df["logTeff"]
    x_err=[df["elogTeff2"], df["elogTeff1"]]
    y=df["logL"]
    y_err=[df["elogL2"], df["elogL1"]]

    fig,ax = plt.subplots()

    fmt='o'
    if hue:
        sns.scatterplot(data=df, x='logTeff', y='logL', hue=hue, ax=ax, zorder=3, alpha=0.8)
        fmt='none'
    ax.errorbar(x,y,y_err,x_err,fmt=fmt,ecolor='grey',alpha=0.5,zorder=2)
    ax.xaxis.set_inverted(True)
    ax.set_xlabel("log[ Teff (K) ]")
    ax.set_ylabel("log[ L (Lsol) ]")

    fig.savefig("DataExploring/HRds/"+savename)

df_goodMS = pd.read_csv("DataExploring/good_MS.txt", sep='\t')
df_goodMS.set_index("ID", inplace=True)
HRplot(df_goodMS, "HRgoodMS700.pdf")

df_RGB = pd.read_csv("DataExploring/good_RGB.txt", sep="\t")
df_RGB.set_index("ID", inplace=True)
df_MS_RGB = pd.concat([df_goodMS, df_RGB], axis=0)
HRplot(df_MS_RGB, "HR_goodMS700_RGB5816.pdf", hue='class')

df_NASA = pd.read_csv("Datasets/NASAexop_archive_stars.txt", sep="\t")
HRplot(df_NASA, "HR_NASAexop_stars.pdf")

df_old_data = pd.read_csv("Datasets/all6_2018_data.txt", sep="\t")
HRplot(df_old_data, "HR2018data.pdf", hue="class")

df_all_current = pd.read_csv("DataExploring/good_RGB.txt", sep="\t", comment="#")
#to be continued

df_plato = pd.read_csv("Datasets/plato_data.txt", sep="\t")
HRplot(df_plato, "HRplato.pdf")

df_plato["class"] = "PLATO"
df_plato_goodMS = pd.concat([df_plato, df_goodMS], axis=0)
HRplot(df_plato_goodMS, "HRplato_goodMS.pdf", hue='class')
