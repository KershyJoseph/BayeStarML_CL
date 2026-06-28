"""
JK 27/04/26
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("BayestarML/data/693ms.txt", sep=',')
df.set_index("ID", inplace=True)

#histogram of detection modes
# plt.figure()
# modes = df["mode"]
# pd.Series(modes).value_counts(dropna=False).plot(kind='bar')
# plt.xlabel("Detection Mode")
# plt.ylabel("Frequency")
# plt.savefig("Side_quests/DataExploring/modes_hist_MS.pdf")

# #histogram of spectral types
# plt.figure()
# df["type"] = df["type"].str[0] #bit dodgy as one entry for example is F/G
# df["mode"] = df["mode"].replace("TEB", "EB")
# df["mode"] = df["mode"].replace("A/I", "A")
# sns.countplot(data=df, x='type', hue='mode', alpha=0.7)
# plt.xlabel("Spectral Type")
# plt.ylabel("Frequency")
# plt.savefig("Side_quests/DataExploring/types_hist_MS.pdf")

#spread of data in mass range on goodMS
def plot_target_spread(df, target, multiple="stack"):
    """Function to get histogram of mass spread with different databases shown
    db_name should be a string for creating file name
    target is "R" or "M"
    """
    plt.figure()
    labels = {1: "Old", 2: "Revised", 3: "New"}
    df["database"] = df["database"].map(labels)
    sns.histplot(data=df, x=target, hue="database", multiple=multiple) 
    plt.xlabel(target+" (Msol)")
    plt.ylabel("Number of stars")
    plt.show()

def plot_feature_target(df:pd.DataFrame, dataset_key:str, feature:str, target:str):
    """Plot target as a function of feature - should be keys in df
    """
    plt.figure()
    x = df[feature]
    x_err = df["e"+feature]#+"1"]
    y = df[target]
    y_err = df["e"+target]
    plt.errorbar(x, y, y_err, x_err, fmt='o', alpha=0.3)
    # plt.plot(np.log10(1.193), 0.499, 'rx')
    # plt.plot(np.log10(0.08), 0.566, 'rx')
    plt.xlabel(feature)
    plt.ylabel(target+" ("+target[0]+"sol)")
    plt.savefig("figures/feature_target_figs/"+dataset_key+"_"+feature+"_"+target+".pdf")
    plt.close()

def diagnostics(df, name):
    print("Diagnostics on ", name)

    print(df["mode"].value_counts())

    print("Old stars: ", len(df[(df["database"]==1)]))

    print("New (and revised) stars: ", len(df[(df["database"]!=1)]), "out of ", len(df))

    print("New, new stars: ", len(df[(df["database"]==3)]), "out of ", len(df))

    print("New range stars: ", len(df[(df["M"]<=0.8) | (df["M"]>=1.4)]), "out of ", len(df))

    print("New stars AND new range stars: ", len(df[((df["M"]<=0.8) | (df["M"]>=1.4)) & (df["database"]!=1)]), "out of ", len(df))

    print("Low mass stars: ", len(df[(df["M"]<=0.8)]), "out of ", len(df))

    print("New/revised stars AND low mass stars: ", len(df[(df["M"]<=0.8) & (df["database"]!=1)]), "out of ", len(df))

diagnostics(df, "693ms")