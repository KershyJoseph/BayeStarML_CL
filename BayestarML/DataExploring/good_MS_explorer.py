"""
JK 27/04/26
"""
import pandas as pd
import matplotlib.pyplot as plt
from preprocess import return_train_test
import seaborn as sns

df = pd.read_csv("DataExploring/good_MS.txt", sep='\t')

#histogram of detection modes
modes = df["mode"]
pd.Series(modes).value_counts(dropna=False).plot(kind='bar')
plt.xlabel("Detection Mode")
plt.ylabel("Frequency")
plt.savefig("modes_hist_MS.pdf")
plt.show()

#histogram of spectral types
df["type"] = df["type"].str[0] #bit dodgy as one entry for example is F/G
sns.countplot(data=df, x='type', hue='mode', alpha=0.7)
plt.xticks(rotation=45)
plt.xlabel("Spectral Type")
plt.ylabel("Frequency")
plt.show()

#see test and train data
X_train, X_test, Y_train, Y_test = return_train_test(df, normalised=False)
