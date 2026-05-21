"""JK 16/05/2026
Get baseline accuracy on mass prediction with an Extreme Gradient Boosting model
Based on tutorial here:
https://machinelearningmastery.com/xgboost-for-regression/
[Last accessed 16/05/2026]
"""

from pandas import read_csv
from xgboost import XGBRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from numpy import absolute

#load data
df = read_csv("DataExploring/datos_todos_v20261905.txt", sep="\t", comment="#")
training_fs = ["Teff", "Fe/H", "L", "logg"]
X, y = df[training_fs], df["M"]

#make model
model = XGBRegressor(random_state=42)

#evaluate with repeated k-fold cross-validation
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

#search for the best model params
param_grid = {
    'eta': [0.3, 0.2, 0.1, 0.01],
    'n_estimators': [20, 50, 100, 200, 400],
    'max_depth': [2, 3, 4],
    'reg_lambda': [1, 10], #to stop overfitting. Default is 1.
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_absolute_percentage_error',
    cv=cv,
    n_jobs=-1
)
grid_search.fit(X,y)

#print results
best_params = grid_search.best_params_
best_score = absolute(grid_search.best_score_)*100 #to make it a percent (bad naming)
best_index = grid_search.best_index_
best_score_std = grid_search.cv_results_["std_test_score"][best_index]

print(f"---- Best params ----\n{best_params}")
print("-+-+-+-+-+-+-+-+-+-+-")
print(f"MARD result across all cross-validations: {best_score:.2f} +/- {best_score_std:.2f} %")

#RESULTS/////////////////////////////////////////////////////

#MASS

# ---- Best params GoodMS ----
# {'colsample_bytree': 0.8, 'eta': 0.1, 'max_depth': 4, 'n_estimators': 100, 'reg_lambda': 1, 'subsample': 0.8}
# -+-+-+-+-+-+-+-+-+-+-
# MARD result across all cross-validations: 5.46 +/- 0.01 %

# ---- Best params 2018Data ---- Needs redoing
# {'colsample_bytree': 1, 'eta': 0.1, 'max_depth': 3, 'n_estimators': 400, 'reg_lambda': 10, 'subsample': 1}
# -+-+-+-+-+-+-+-+-+-+-
# MARD result across all cross-validations: 5.30 +/- 0.01 %

#RADIUS 

# ---- Best params GoodMS ---- 
# {'colsample_bytree': 0.8, 'eta': 0.1, 'max_depth': 4, 'n_estimators': 200, 'reg_lambda': 1, 'subsample': 0.8}
# -+-+-+-+-+-+-+-+-+-+-
# MARD result across all cross-validations: 3.68 +/- 0.01 %


