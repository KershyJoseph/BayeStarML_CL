import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Outputs/MS/mass testing/GP_mass_ADVI_trials_data.txt", sep="\t")

x = df["Inducing_Points"]

plt.figure()

plt.plot(x, df["MARD"], 'x', label="MARD")
#plt.plot(x, df["elpd_loo"], '+', label="elpd_loo")
plt.plot(x, df["pareto_k_bad(%)"], 'o', label="bad pareto k (%)")
plt.legend()
plt.xlabel("Number inducing points")

plt.savefig("Outputs/MS/mass testing/GP_mass_ADVI_trial_fig.pdf")