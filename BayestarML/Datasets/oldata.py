"""
See dist of params in old set
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def spreadomatic(df, var, save_path, hue=None, xlabel=None):
    """Make a histogram for a given var (which should be one of df's keys)
    """
    plt.figure()
    sns.histplot(data=df, x=var, hue=hue)
    if xlabel:
        plt.xlabel(xlabel)
    plt.ylabel("Number of stars")
    plt.savefig(save_path)

df = pd.read_csv("Datasets/datos_tot_v20180517_adapted.txt", sep="\t")
print(len(df))
df_all6 = df[df[["M", "R", "Teff", "L", "FeH", "logg"]].notna().all(axis=1)]
print(len(df_all6))

for var in ["M", "R"]:
    spreadomatic(df_all6, var, "Datasets/2018spreads/"+var+"_2018spread.pdf")
