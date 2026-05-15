"""
JK 27/04/26
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("DataExploring/good_MS.txt", sep='\t')
df.set_index("ID", inplace=True)

#histogram of detection modes
plt.figure()
modes = df["mode"]
pd.Series(modes).value_counts(dropna=False).plot(kind='bar')
plt.xlabel("Detection Mode")
plt.ylabel("Frequency")
plt.savefig("DataExploring/modes_hist_MS.pdf")

#histogram of spectral types
plt.figure()
df["type"] = df["type"].str[0] #bit dodgy as one entry for example is F/G
df["mode"] = df["mode"].replace("TEB", "EB")
df["mode"] = df["mode"].replace("A/I", "A")
sns.countplot(data=df, x='type', hue='mode', alpha=0.7)
plt.xlabel("Spectral Type")
plt.ylabel("Frequency")
plt.savefig("DataExploring/types_hist_MS.pdf")

#spread of data in mass range on goodMS
def plot_mass_spread(df, name, multiple="layer"):
    """Function to get histogram of mass spread with different databases shown
    'name' should be a string for file name
    """
    plt.figure()
    labels = {1: "Old", 2: "Revised", 3: "New"}
    df["database"] = df["database"].map(labels)
    sns.histplot(data=df, x="M", hue="database", multiple=multiple) 
    plt.xlabel("Mass (Msol)")
    plt.ylabel("Number of stars")
    plt.savefig("DataExploring/mass_spread_"+name+".pdf")

df_strict = pd.read_csv("DataExploring/strict_MS.txt", sep="\t", comment="#")
plot_mass_spread(df_strict, "strictMS")

plot_mass_spread(df, "goodMS", multiple="stack")
