"""Script to tune model hyperparameters with BO
JK 01/06/2025
"""

from preprocess import get_dataset, return_train_test, denormalise_val
from exec_trainer import Dataset
from models import hbnn
from utils import train, mard
from pred_sampling import posterior_predictive_GP, sample_post_pred_HBNN_para
import optuna
import arviz as az
import matplotlib.pyplot as plt
import time
import pandas as pd

def objective(trial, data, min, max, score:str, draw=1000, chains=4, target_accept=0.95):
    nodes = trial.suggest_int("nodes", min, max)

    model = hbnn.HBNN_M4(data.x_train, data.mass_train, data.x_train_er, data.emass_train, nodes)
    trace = train(model, draw=draw, chains=chains, target_accept=target_accept,
                  nuts_sampler="nutpie")

    if score=="elpd_loo":
        elpd_loo = az.loo(trace).elpd_loo
        return elpd_loo
    elif score=="MARD":
        pred, lpd = sample_post_pred_HBNN_para(trace, data.x_test, data.x_test_err, nodes, 4, "Mass")
        means = pred.mean(0)
        return mard(data.unorm_mass, means)

def best_scores_at_trial_n(score, all_scores):
    best_scores = []
    best_score = all_scores.iloc[0]
    for score in all_scores:
        if score=="MARD":
            new_best = min(best_score, score)
        elif score=="elpd_loo":
            new_best = max(best_score, score)
        best_scores.append(new_best)
        best_score = new_best
    return best_scores


if __name__ == '__main__':
    start_time = time.perf_counter()

    min=2
    max=64
    n_startup=3
    n=7
    score="MARD"

    df_train = get_dataset('DataExploring/good_MS.txt', logL=True)

    (x_train, x_train_er, x_test, x_test_err, mass_train, emass_train,
    mass_test, emass_test, rad_train, erad_train, rad_test, erad_test
    ) = return_train_test(df_train, logL=True)

    dataset = Dataset(
        x_train = x_train[['Teff', 'logg', 'FeH', 'logL']],
        x_train_er = x_train_er[['eTeff', 'elogg', 'eFeH', 'elogL']],
        x_test = x_test[['Teff', 'logg', 'FeH', 'logL']],
        x_test_err = x_test_err[['eTeff', 'elogg', 'eFeH', 'elogL']],

        rad_train=rad_train,
        erad_train=erad_train,
        rad_test=rad_test,
        erad_test=erad_test,
        mass_train=mass_train,
        emass_train=emass_train,
        mass_test=mass_test,
        emass_test=emass_test,

        unorm_mass = denormalise_val(mass_test, 'Mass'),
        unorm_radius = denormalise_val(rad_test, 'Radius')
        )

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(n_startup_trials=n_startup))
    study.optimize(lambda trial: objective(trial, dataset, min, max, score=score),
                   n_trials=n)
    df_results = study.trials_dataframe()
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df_results)
    df_results = df_results[df_results["state"]=="COMPLETE"] #just in case
    plt.figure()
    plt.plot(df_results["number"], df_results["value"], 'bo', label=score+"at trial n")
    running_bests = best_scores_at_trial_n(score="MARD", all_scores=df_results["value"])
    plt.plot(df_results["number"], running_bests, 'b-', label="Best "+score+" at trial n")
    plt.xlabel("Trial number")
    plt.ylabel(score)
    plt.legend()
    plt.grid(linestyle="--", alpha=0.5)
    plt.savefig("Outputs700MS/Tuning/NNmass_"+score+"_"+str(n)+"_"+str(min)+"_"+str(max)+".pdf")

    print(f"""
    After {n} trials for nodes in range [{min},{max}]:

    Best Number Nodes: {study.best_params}
    Best ELPD-LOO: {study.best_value}

    Wall clock time: {(time.perf_counter()-start_time)/60} mins")
    """
    )
