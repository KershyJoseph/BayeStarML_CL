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
plt.figure()
labels = {1: "Old", 2: "Revised", 3: "New"}
df["database"] = df["database"]
sns.histplot(data=df, x="M", hue="database", multiple="layer") 
plt.xlabel("Mass (Msol)")
plt.ylabel("Number of stars")
plt.savefig("DataExploring/mass_spread.pdf")
