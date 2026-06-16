import pandas as pd
import numpy as np

df = pd.read_csv("Datasets/estrellas_anfitrionas.txt", sep="\t")

df1 = df[
    ['eTeff1', 'elogg1', 'eFeH1', 'elogL1']
    ].copy()
df2 = df[
    ['eTeff2', 'elogg2', 'eFeH2', 'elogL2']
    ].copy()
df1.columns = ['eTeff', 'elogg', 'eFeH', 'elogL']
df2.columns = ['eTeff', 'elogg', 'eFeH', 'elogL']
df_err = (df1 + df2) / 2

df = pd.concat([df, df_err], axis=1)

df.to_csv("Datasets/estrellas_anfitrionas.txt", sep="\t", index=False)
