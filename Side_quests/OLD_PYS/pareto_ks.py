
import pandas as pd
from BayestarML.src.data_utils import logomatic
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

def hr_plot(df, df_bad, savename: str, hue: str = None, density_plot: bool = False, mass_plot: bool = False):
    """Plot logTeff (higher Teff to the left) against logL for stars in df
    """
    if "logTeff" not in df.columns:
        df = logomatic(df, ["Teff"])
    if "logL" not in df.columns:
        df = logomatic(df, ["L"])

    if "logTeff" not in df_bad.columns:
        df_bad = logomatic(df_bad, ["Teff"])
    if "logL" not in df_bad.columns:
        df_bad = logomatic(df_bad, ["L"])

    x = df["logTeff"]
    x_err = [df["elogTeff2"], df["elogTeff1"]]
    y = df["logL"]
    y_err = [df["elogL2"], df["elogL1"]]

    plt.close()
    fig, ax = plt.subplots(figsize=(8,6))

    colors = {
        "NASA Exoplanet Archive": "lightsteelblue",
        "Lamirel et al. (2026)": "mediumblue",
        "Expansion in this work": "red"
    }

    zorders = {
        "NASA Exoplanet Archive": 2,
        "Lamirel et al. (2026)": 4,
        "Expansion in this work": 4
    }

    if hue:
        for catalogue, group in df.groupby(hue):
            x = group["logTeff"]
            x_err = [group["elogTeff2"], group["elogTeff1"]]
            y = group["logL"]
            y_err = [group["elogL2"], group["elogL1"]]
            ax.errorbar(x, y, y_err, x_err, fmt='o', ms=5, capsize=1.5, alpha=0.5, ecolor='gray', color=colors[catalogue], mec='gray', label=catalogue, zorder=zorders[catalogue])
    if density_plot:
        stack = np.vstack([x, y])
        density = gaussian_kde(stack)(stack)
        idx = np.argsort(density)
        x, y, density = x[idx], y[idx], density[idx]
        scatter = ax.scatter(x, y, c=density, cmap='plasma', s=5, zorder=3)
        fig.colorbar(scatter, ax=ax, label="Density of Stars")
    elif mass_plot:
        scatter = ax.scatter(x, y, c=df["M"], cmap='plasma', s=5, zorder=3)
        fig.colorbar(scatter, ax=ax, label="Mass (Msol)")

    ax.plot(df_bad["logTeff"], df_bad["logL"], 'x', color='orange', ms=10, zorder=5, mew = 3)
    ax.xaxis.set_inverted(True)
    ax.set_xlabel(r"log($\mathrm{T_{eff}}$) [K]")
    ax.set_ylabel(r"log(L) [$\mathrm{L_{\odot}}$]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.savefig("BayestarML/data/figures/hr_diagrams/" + savename)
    plt.close()


ms693_train = pd.read_csv("BayestarML/data/693ms_norm_train.txt", sep=',')
ms693 = pd.read_csv("BayestarML/data/693ms.txt", sep=',')

#ms
gp_m_ms = {
    "bad": [75, 139, 220, 268, 349, 430, 503],
    "very bad": [57, 187, 323, 362, 396]
}

gp_r_ms = {
    "bad": [22, 75, 139, 160, 205, 268, 343, 396, 446, 447, 510],
    "very bad": [57, 187, 216, 323, 499, 503]
}

nn_r_ms = {
    "bad": [50, 106, 127, 199, 200, 236, 371, 428],
    "very bad": [38, 146]}

def print_bad_stars(bads, name):
    bad_ids = ms693_train["ID"].iloc[bads]
    df_bad = ms693.loc[ms693["ID"].isin(bad_ids)]
    print(name,"\n",ms693.loc[ms693["ID"].isin(bad_ids)])
    return df_bad

df_2018 = pd.read_csv("BayestarML/data/training_databases/all6_2018_data.txt", sep='\t')
df_2018 = df_2018[df_2018["class"]=="ms"]
df_nasa = pd.read_csv("BayestarML/predict/prediction_datasets/NASAexop_archive_stars_all6.txt", sep='\t')
df_new = pd.read_csv("BayestarML/data/693ms.txt")
df_new = logomatic(df_new, ["Teff"])
df_all = pd.concat([df_nasa, df_2018, 
                    df_new], 
                    keys=["NASA Exoplanet Archive", "Lamirel et al. (2026)", 
                          "Expansion in this work"],
                    names=["Catalogue", None])
df_all.reset_index()


df_bad = print_bad_stars(
    list(set(gp_m_ms["very bad"]) | set(gp_r_ms["very bad"])), "V bad GP MS"
)

hr_plot(df_all, df_bad, "max_nasa_new_hr_plot_M_vbad_paretos.pdf", hue="Catalogue")

print(df_bad["FeH"], df_bad["eFeH1"])