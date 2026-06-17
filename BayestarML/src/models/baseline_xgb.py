"""JK 16/05/2026
Get baseline accuracy on mass prediction with an Extreme Gradient Boosting model
Based on tutorial here:
https://machinelearningmastery.com/xgboost-for-regression/
[Last accessed 16/05/2026]
"""

import json

import numpy as np
import optuna
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor


# bayesian optimisation
class Objective:
    """ """

    def __init__(self, X, y):
        """Instantiate with training data X, y"""
        self.X = X
        self.y = y

    def __call__(self, trial):
        """Return MARD for different iterations of below params"""
        params = {
            "objective": "reg:squarederror",  # objective is minimise squared error
            "eval_metric": "mape",  # mean absolute percentage error (basically MARD) - what gets printed
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }

        # evaluate with k-fold cross-validation
        cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=99)
        fold_mards = []
        for train_idxs, test_idxs in cv.split(self.X, self.y):
            X_train, X_test = self.X.iloc[train_idxs], self.X.iloc[test_idxs]
            y_train, y_test = self.y.iloc[train_idxs], self.y.iloc[test_idxs]

            model = XGBRegressor(**params, random_state=99)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            eps = 1e-8  # could get rid to be fair as I know all masses and radii in data are > 0...
            fold_mard = np.mean(100 * np.abs(y_test - preds) / (y_test + eps))
            fold_mards.append(fold_mard)

        return np.mean(fold_mards)  # mean MARD across folds


# run study
def studyrunner(X, y, savefile=None):
    """ """
    sampler = optuna.samplers.TPESampler(n_startup_trials=15, multivariate=True)
    study = optuna.create_study(sampler=sampler, direction="minimize")
    objective = Objective(X, y)

    study.optimize(objective, n_trials=100, show_progress_bar=True)

    # print/save results
    results = {"best_mard": study.best_value, "best_params": study.best_params}
    print("\n\n", results)
    if savefile:
        with open("BayestarML/data/xgb_results/" + savefile, "w") as f:
            json.dump(results, f, indent=4)


# make predictions on Plato data and see what MARD is like
# def platopredder():
#     df_plato = read_csv("Datasets/plato_data.txt", sep='\t')
#     X_plato, y_plato = df_plato[training_fs], df_plato["M"]

#     best_params = {'learning_rate': 0.013233351056378049, 'n_estimators': 633, 'max_depth': 6, 'min_child_weight': 1, 'reg_lambda': 0.004561346827040905, 'reg_alpha': 0.005860201359441429, 'subsample': 0.623075537754977}
#     model = xgbRegressor(**best_params, random_state=99)
#     model.fit(X,y)

#     plato_preds = model.predict(X_plato)
#     plato_xgb_mard = np.mean(100*np.abs(y_plato-plato_preds)/(y_plato))
#     print(f"Plato mard: {plato_xgb_mard:.3f}%")


# RESULTS/////////////////////////////////////////////////////

# OPTUNA TPE-SAMPLER BO -------------------------

# MASS - goodMS700
# Best MARD: 5.778
# For best params: {'learning_rate': 0.013233351056378049, 'n_estimators': 633, 'max_depth': 6, 'min_child_weight': 1, 'reg_lambda': 0.004561346827040905, 'reg_alpha': 0.005860201359441429, 'subsample': 0.623075537754977}

# No FeH
# Best MARD: 7.061
# For best params: {'learning_rate': 0.019438711953432122, 'n_estimators': 491, 'max_depth': 6, 'min_child_weight': 1,'reg_lambda': 1.2165147000281793, 'reg_alpha': 0.0018771614299610306, 'subsample': 0.5307806524705606}

# PLATO
# Plato mard: 6.213% (compared to 5.8%)

# PLATO NO FeH
# Plato mard: 7.939% (compared to 7% mean from k-fold cross v)

# 2018 Data just MS L not logL
# Best MARD: 3.953 (compared to Max MARDs 3.9 3.5 3.7 3.5)
# For best params: {'learning_rate': 0.011311727518843268, 'n_estimators': 724, 'max_depth': 5, 'min_child_weight': 2, 'reg_lambda': 0.001615497456041067, 'reg_alpha': 0.031499983937579, 'subsample': 0.5503081652739142}


# RADIUS - goodMS700
# Best MARD: 3.659
# For best params: {'learning_rate': 0.012932424554637587, 'n_estimators': 823, 'max_depth': 6, 'min_child_weight': 1, 'reg_lambda': 0.004519051077834074, 'reg_alpha': 0.0024557881422537934, 'subsample': 0.5495253440342718}


# GRID SEARCH -----------------------------

# MASS

# ---- Best params GoodMS ----
# {'colsample_bytree': 1, 'eta': 0.1, 'max_depth': 4, 'n_estimators': 200, 'reg_lambda': 1, 'subsample': 1}
# -+-+-+-+-+-+-+-+-+-+-
# MARD result across all cross-validations: 5.51 +/- 0.01 %

# ---- Best params 2018Data ----
# Attempt 1
# {'colsample_bytree': 1, 'eta': 0.1, 'max_depth': 3, 'n_estimators': 400, 'reg_lambda': 10, 'subsample': 1}
# -+-+-+-+-+-+-+-+-+-+-
# MARD result across all cross-validations: 5.30 +/- 0.01 %
# Attempt 2
# {'colsample_bytree': 1, 'eta': 0.01, 'max_depth': 4, 'n_estimators': 800, 'reg_lambda': 1, 'subsample': 0.8}
# -+-+-+-+-+-+-+-+-+-+-
# MARD result across all cross-validations: 5.27 +/- 0.01 %

# RADIUS

# ---- Best params GoodMS ----
# {'colsample_bytree': 1, 'eta': 0.1, 'max_depth': 4, 'n_estimators': 400, 'reg_lambda': 1, 'subsample': 0.8}
# -+-+-+-+-+-+-+-+-+-+-
# MARD result across all cross-validations: 3.73 +/- 0.01 %

# ---- Best params 2018data ----
# {'colsample_bytree': 1, 'eta': 0.01, 'max_depth': 4, 'n_estimators': 800, 'reg_lambda': 1, 'subsample': 0.8}
# -+-+-+-+-+-+-+-+-+-+-
# MARD result across all cross-validations: 5.27 +/- 0.01 %
