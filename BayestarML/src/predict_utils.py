import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors
from BayestarML.src.data_utils import select_clean_data, normalise


def check_feature_extrapolation(x_train, x_pred, k=10, percentile=95):
    """
    Compute Convex Hull check and average distance to k-nearest neighbours.
    Doesn't take into account errors in x_train or x_pred!
    """
    #convex hull check
    triangulation = Delaunay(x_train)
    is_inside = triangulation.find_simplex(x_pred) >= 0 #find_simplex returns index of simplex point or -1 if x_pred point not in convex hull

    #avg distance of each training point to nearest k training points
    nn_t = NearestNeighbors(n_neighbors=k+1) #+1 to ignore distance to self
    nn_t.fit(x_train)
    distances_t, _ = nn_t.kneighbors(x_train) #distances shape (n_train, k)
    avg_dists_t = np.mean(distances_t[:, 1:], axis=1) #skip first column where nearest point is same point
    dist_threshold = np.percentile(avg_dists_t, percentile)

    #avg distance of each pred point to nearest k training points
    nn_p = NearestNeighbors(n_neighbors=k)
    nn_p.fit(x_train)
    distances, _ = nn_p.kneighbors(x_pred) #distances shape (n_pred, k)
    avg_dists_p = np.mean(distances, axis=1) #shape (n_pred,)

    interpolation_mask = is_inside & (avg_dists_p < dist_threshold)

    return interpolation_mask

def prepare_pred_data(
    filename: str, training_dataset_key: str, features: list, add_log_vars: list = None, extrapolate=False
):
    """
    Normalize input data and return DataFrames for normalized values and errors.
    Check all input data within training ranges.

    Parameters:
    - teff, logg, FeH, l: Input values (can be scalars or arrays)
    - eteff, elogg, eFeH, el: Associated errors (can be scalars or arrays)

    Returns:
    - x_test: DataFrame with normalized values (columns: 'Teff', 'logg', 'FeH', 'L')
    - x_test_error: DataFrame with normalized errors (columns: 'eTeff', 'elogg', 'eFeH', 'eL')
    """
    df = pd.read_csv("BayestarML/predict/prediction_datasets/" + filename, sep="\t")
    # filter to stars with all training features present with err. Add symmetric err column and log vars if needed.
    df_clean = select_clean_data(
        df,
        features,
        targets=[],
        add_logvars=add_log_vars,
        check_detached=False,
        lum_check=False,
    )

    # normalise input data
    df_norm = normalise(df_clean, None, training_dataset_key, x_only=True)

    x, x_er = df_norm[features], df_norm[[f"e{f}" for f in features]]  # might need modifying for scalar inputs?

    #check extrapolation
    interp_mask = check_feature_extrapolation(
        x_train=pd.read_csv("BayestarML/data/"+training_dataset_key+"_norm_train.txt"),
        x_pred=x
    )
    print(f"{len(x[~interp_mask])} stars marked as extrapolating from training database.")

    return x, x_er, interp_mask

def get_bhs_weights(w_draws, y_draws):
    """Save a df of the weights BHS assigns to each model for each test value
    """
    mean_ws = w_draws.mean(0) #shape (N_test, K)

    bart_ws = mean_ws[:,0]
    hbnn_ws = mean_ws[:,1]
    gp_ws = mean_ws[:,2]

    #preds = 


# with open("BayestarML/data/" + training_dataset_key + "_constants.json", "r") as f:
#         constants = json.load(f)
#         x_constants = constants["training_fs"]

#     # check predicting within feature training ranges
#     for f in features:
#         MIN = x_constants["MIN"][f]
#         MAX = x_constants["MAX"][f]
#         RANGE = MAX - MIN
#         x_new = x[
#             (x[f] >= MIN + 0.025 * RANGE) & (x[f] <= MAX - 0.025 * RANGE)
#         ]  # keep middle 95%
#         if len(x) != len(x_new):
#             print(f"Star(s) outside middle 95% of {f} training range:\n",
#                   pd.concat([x, x_new]).drop_duplicates(keep=False))
#         if extrapolate:
#             continue # don't update x and thus remove extrapolating points
#         print(
#             f"Removing {len(x) - len(x_new)} stars with {f} inputs outside middle 95% of {f} training range."
#         )
#         x = x_new