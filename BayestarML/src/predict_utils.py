import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors
from BayestarML.src.data_utils import select_clean_data, normalise, consistency_check, lineamatic
from BayestarML.src.train_utils import load_data
import json
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

def plot_convex(x_train):
    hull = ConvexHull(x_train)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # 2. Extract and format the 3D surface facets for plotting
    faces = []
    for simplex in hull.simplices:
        # Get the 3D coordinates for the vertices of this facet
        facet_vertices = x_train[simplex]
        faces.append(facet_vertices)

    # Create a 3D polygon collection for the hull shell
    hull_surface = Poly3DCollection(faces, alpha=0.15, edgecolor="k", linewidths=0.5)
    hull_surface.set_facecolor("cyan")  # Gives the hull a translucent "glass" look
    ax.add_collection3d(hull_surface)

    ax.plot(7000, -2, 4, 'rx')
    ax.set_xlabel(r"$T_{eff}$ (K)")
    ax.set_ylabel(r"$\log L$ [L$_{\odot}$]")
    ax.set_zlabel(r"$\log g$ (dex)")
    ax.legend()

    # Dynamic viewing angle adjustment so it looks good out of the box
    ax.view_init(elev=20, azim=45)

    plt.show()
    plt.savefig("convex_hull.pdf")


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
    filename: str, training_dataset_key: str, features: list, target: str, add_log_vars: list = None, check_consistency=False, symmetric_errs=False, y_comp=True
):
    """Prepare normalised data for predictions and check feature extrapolation.
    Pass filename=="test_set" for predictions on test set.
    """
    data, _ = load_data(training_dataset_key, target)
    if filename == "test_set": #prepare test set for prediction
        x = data["x_test"]
        x_er = data["x_test_err"]
        y_compare = data["unorm_y_test"]
        y_comp_er = data["unorm_y_test_err"]
        star_id = data["test_ID"]

    else: #prepare new data for prediction
        df = pd.read_csv("BayestarML/predict/prediction_datasets/" + filename, sep="\t")
        # filter to stars with all training features present with err. Add symmetric err column and log vars if needed.
        ts = []
        if check_consistency:
            ts = ["M", "R"] #for logg and L check

        if "L" not in features:
            df = lineamatic(df, add_linvars=["logL"])
            features.append("L")

        df_clean = select_clean_data(
            df,
            features,
            targets=ts,
            add_logvars=add_log_vars,
            check_detached=False,
            check_consistency=False,
            symmetric_errs=symmetric_errs
        )

        if add_log_vars:
            for var in add_log_vars:
                for list in [features, ts]:
                    if var in list:
                        list.remove(var)
                        list.append("log" + var)

        if check_consistency:
            df_clean = consistency_check(df_clean, "logg", "predict/prediction_datasets/con_check/"+filename[:-4]+"_logg_"+target+".pdf", symmetric_errs=True)
            df_clean = consistency_check(df_clean, "L", "predict/prediction_datasets/con_check/"+filename[:-4]+"_L_"+target+".pdf", symmetric_errs=True)
        features.remove("L")

        # normalise input data
        x_unorm = df_clean[features+[f"e{f}" for f in features]]
        if y_comp:
            y_compare, y_comp_er = df_clean[target], df_clean[f"e{target}"]
        x_norm = normalise(x_unorm, None, training_dataset_key, x_only=True)
        x, x_er = x_norm[features], x_norm[[f"e{f}" for f in features]]  # might need modifying for scalar inputs?
        star_id = x_norm.index

    #check feature extrapolation
    interp_mask = check_feature_extrapolation(data["x_train"], x)
    print(f"{len(x[~interp_mask])} stars marked as extrapolating from training database inputs.")

    if y_comp:
        #check target extrapolation
        with open("BayestarML/data/" + training_dataset_key + "_constants.json", "r") as f:
            constants = json.load(f)
        y_min = constants["targets"]["MIN"][target]
        y_max = constants["targets"]["MAX"][target]
        t_mask = (y_compare > y_min) & (y_compare < y_max)
        if filename != "test_set":
            print(f"Removing {len(x[~t_mask])} stars for being outside target training range: {df_clean[~t_mask]}")

        star_id, x, x_er, interp_mask = star_id[t_mask], x[t_mask], x_er[t_mask], interp_mask[t_mask]
        y_compare, y_comp_er = y_compare[t_mask], y_comp_er[t_mask]

        return star_id, x, x_er, y_compare, y_comp_er, interp_mask
    
    else:
        return star_id, x, x_er, interp_mask


def get_bhs_weights(w_draws, star_id, savename:str):
    """Save a df of the weights BHS assigns to each model for each test star
    """
    mean_ws = w_draws.mean(0) #shape (N_test, K)

    bart_ws = mean_ws[:,0]
    hbnn_ws = mean_ws[:,1]
    gp_ws = mean_ws[:,2]

    df_weights = pd.DataFrame({
        "ID": star_id,
        "BART_weight": bart_ws,
        "HBNN_weight": hbnn_ws,
        "GP_weight": gp_ws
    })

    df_weights.to_csv("BayestarML/predict/outputs_bhs/"+savename, index=None)

    return df_weights


def plot_bhs_weights(ws, target, compare_file, savename):
    """
    """
    ws.set_index("ID", inplace=True)
    all_masses = pd.read_csv(compare_file, usecols=[target, "ID"], sep='\t')
    all_masses.set_index("ID", inplace=True)
    masses = all_masses.loc[ws.index]

    df_plot = pd.concat([ws, masses], axis=1)
    df_plot.sort_values(target, inplace=True)

    fig, ax = plt.subplots(figsize=(8,6))
    for w in ["BART_weight", "HBNN_weight", "GP_weight"]:
        ax.plot(df_plot[target], df_plot[w], label=w)
    ax.set_xlabel(target+" ("+target+"sol)")
    ax.set_ylabel("BHS base model weight")
    ax.legend()
    fig.savefig("BayestarML/predict/outputs_bhs/"+savename)



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