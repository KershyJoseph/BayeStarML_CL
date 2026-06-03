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

df = pd.read_csv("DataExploring/good_MS.txt", sep='\t')
df.set_index("ID", inplace=True)
HRplot(df, "HRgoodMS700.pdf")

df_RGB = pd.read_csv("DataExploring/good_RGB.txt", sep="\t")
df_RGB.set_index("ID", inplace=True)
df_MS_RGB = pd.concat([df, df_RGB], axis=0)
HRplot(df_MS_RGB, "HR_goodMS700_RGB5816.pdf", hue='class')

df_NASA = pd.read_csv("Datasets/NASAexop_archive_stars.txt", sep="\t")
HRplot(df_NASA, "HR_NASAexop_stars.pdf")

df_old_data = pd.read_csv("Datasets/all6_2018_data.txt", sep="\t")
HRplot(df_old_data, "HR2018data.pdf", hue="class")
