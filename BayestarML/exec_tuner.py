"""Script to tune model hyperparameters with BO
JK 01/06/2025
"""

from preprocess import get_dataset, return_train_test, denormalise_val
from exec_trainer import Dataset
from models import hbnn
from utils import train
import optuna
import arviz as az

def objective(trial, data, draw=1000, chains=4, target_accept=0.95):
    nodes = trial.suggest_int("nodes", 2, 16)

    model = hbnn.HBNN_M4(data.x_train, mass_train, data.x_train_er, data.emass_train, nodes)
    trace = train(model, draw=draw, chains=chains, target_accept=target_accept)

    elpd_loo = az.loo(trace).elpd_loo
    return elpd_loo

if __name__ == '__main__':

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

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=3, data=dataset)

    print("Best Number Nodes:", study.best_params_)
    print("Best ELPD-LOO:", study.best_value)