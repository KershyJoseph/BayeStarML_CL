"""Executable to train HBNN or GP
"""

import time
import psutil
import pymc as pm
from BayestarML.src.pred_sampling import (
    SIMPLE_sample_post_pred_HBNN_para,
    posterior_predictive_GP,
    sample_post_pred_HBNN_para,
)
from BayestarML.src.train_utils import get_results, train, load_data
from BayestarML.src.predict_utils import check_feature_extrapolation
from BayestarML.src.models import gp, hbnn


def train_GP(
    dataset_key,
    target,
    M_mean,
    M_var,
    draws=1000,
    target_accept=0.95,
    chains=4,
    advi=False,
    nutpie=False,
):
    """Function to train GP"""
    hyperp_str = (
        "GP_"
        + target
        + "_"
        + dataset_key
        + "_"
        + str(M_mean)
        + "_"
        + str(M_var)
        + "_"
        + str(draws)
        + "_"
        + str(target_accept)
    )
    outputs_folder_path = "BayestarML/train/outputs" + dataset_key + "/GP_" + target
    nuts_sampler = "pymc"
    if nutpie:
        hyperp_str += "NUTPIE"
        nuts_sampler = "nutpie"

    data, train_dim = load_data(dataset_key, target)

    model, μ_gp, lg_σ_gp, μ_trace, var_trace, Xu, Xu_er = (
        gp.sparse_fully_heteroscedastic_gp(
            data["x_train"], data["x_train_err"], data["y_train"], M_mean, M_var
        )
    )

    if advi:
        approx = pm.fit(n=40000, method="advi", model=model, progressbar=True)
        trace = approx.sample(1000)
        print("ELBO:\n", approx.hist)
        trace.extend(pm.compute_log_likelihood(trace, model=model, var_names="y"))
        # trace.to_netcdf(...)
    else:
        trace = train(
            model,
            outputs_folder_path + "/" + hyperp_str + ".nc",
            draw=draws,
            chains=chains,
            target_accept=target_accept,
            nuts_sampler=nuts_sampler,
        )

    pred, _ = posterior_predictive_GP(
        model,
        μ_gp,
        lg_σ_gp,
        μ_trace,
        var_trace,
        trace,
        data["x_test"],
        data["x_test_err"],
        Xu,
        Xu_er,
        train_dim,
        target,
        dataset_key,
    )

    interp_mask = check_feature_extrapolation(data["x_train"], data["x_test"])
    get_results(pred, data, interp_mask, outputs_folder_path, dataset_key, target, hyperp_str)


def train_NN_1layer(
    dataset_key,
    target,
    n_hidden,
    draws=1000,
    target_accept=0.95,
    chains=4,
    advi=False,
    nutpie=False,
):
    """Function to train HBNN"""
    hyperp_str = (
        "NN_1layer_"
        + target
        + "_"
        + dataset_key
        + "_"
        + str(n_hidden)
        + "_"
        + str(draws)
        + "_"
        + str(target_accept)
    )
    outputs_folder_path = "BayestarML/train/outputs" + dataset_key + "/NN_" + target
    nuts_sampler = "pymc"
    if nutpie:
        hyperp_str += "NUTPIE"
        nuts_sampler = "nutpie"

    data, train_dim = load_data(dataset_key, target)

    model = hbnn.HBNN_M4_simpler(
        data["x_train"],
        data["y_train"],
        data["x_train_err"],
        data["y_train_err"],
        n_hidden,
    )

    if advi:
        approx = pm.fit(n=100000, method="advi", model=model, progressbar=True)
        trace = approx.sample(1000)
        print("ELBO:\n", approx.hist)
        trace.extend(pm.compute_log_likelihood(trace, model=model, var_names="y"))
        # trace.to_netcdf(...)
    else:
        trace = train(
            model,
            outputs_folder_path + "/" + hyperp_str + ".nc",
            draw=draws,
            chains=chains,
            target_accept=target_accept,
            nuts_sampler=nuts_sampler,
        )

    pred, _ = SIMPLE_sample_post_pred_HBNN_para(
        trace,
        data["x_test"],
        data["x_test_err"],
        n_hidden,
        train_dim,
        target,
        dataset_key,
    )

    interp_mask = check_feature_extrapolation(data["x_train"], data["x_test"])
    get_results(pred, data, interp_mask, outputs_folder_path, dataset_key, target, hyperp_str)


def train_NN(
    dataset_key,
    target,
    n_hidden,
    draws=1000,
    target_accept=0.95,
    chains=4,
    advi=False,
    nutpie=False,
):
    """Function to train HBNN"""
    hyperp_str = (
        "NN_"
        + target
        + "_"
        + dataset_key
        + "_"
        + str(n_hidden)
        + "_"
        + str(draws)
        + "_"
        + str(target_accept)
    )
    outputs_folder_path = "BayestarML/train/outputs" + dataset_key + "/NN_" + target
    nuts_sampler = "pymc"
    if nutpie:
        hyperp_str += "NUTPIE_init_mean"
        nuts_sampler = "nutpie"

    data, train_dim = load_data(dataset_key, target)

    model = hbnn.HBNN_M4(
        data["x_train"],
        data["y_train"],
        data["x_train_err"],
        data["y_train_err"],
        n_hidden,
    )

    if advi:
        approx = pm.fit(n=100000, method="advi", model=model, progressbar=True)
        trace = approx.sample(1000)
        print("ELBO:\n", approx.hist)
        trace.extend(pm.compute_log_likelihood(trace, model=model, var_names="y"))
        # trace.to_netcdf(...)
    else:
        trace = train(
            model,
            outputs_folder_path + "/" + hyperp_str + ".nc",
            draw=draws,
            chains=chains,
            target_accept=target_accept,
            nuts_sampler=nuts_sampler,
        )

    pred, _ = sample_post_pred_HBNN_para(
        trace,
        data["x_test"],
        data["x_test_err"],
        n_hidden,
        train_dim,
        target,
        dataset_key,
    )

    interp_mask = check_feature_extrapolation(data["x_train"], data["x_test"])
    get_results(pred, data, interp_mask, outputs_folder_path, dataset_key, target, hyperp_str)


if __name__ == "__main__":
    print("<><><><><<><<><><><>><<><<><<><<<><><><<><<><><><><>")

    process = psutil.Process()
    start_time_CPU = time.process_time()
    start_time_wall = time.perf_counter()

    print("5438rgb M NN, 1 layer. .02/.05 in/out ws and biases. 0.08er. 8_2000")
    train_NN_1layer(
        "5438rgb", "M", 8, draws=2000
    )

    end_time_CPU = time.process_time()
    mem1 = process.memory_info().rss / 1024**2
    print(f"Peak Memory: {mem1:.2f} MB")
    print(f"CPU time accumulated: {(end_time_CPU - start_time_CPU):.5f} s")
    print(f"Total wall clock time: {time.perf_counter() - start_time_wall:.5f} s")

    print("><><><><><><><><><><><><><><><><><><><><><><><><><><")

    # start_time_CPU2 = time.process_time()
    # start_time2 = time.time()

    #print("RGB M NN. 1 layer. 0.05bias, 0.1er, 0.05He w_in_1, 0.1He w_1_out, normal sampler. 4_2000")
    # train_NN_1layer(
    #     "5438rgb", "M", 4, draws=2000
    # )

    # end_time_CPU2 = time.process_time()

    # mem2 = process.memory_info().rss / 1024**2
    # print(f"Peak Memory: {(mem2-mem1):.2f} MB")
    # print(f"CPU time used: {(end_time_CPU2-start_time_CPU2):.5f} s")
    # print(f"Total run time: {time.time()-start_time2:.5f} s")

    # print("><><><><><><><><><><><><><><><><><><><><><><><><><><")

    # start_time_CPU3 = time.process_time()
    # start_time3 = time.time()

    # print("GP - radius - RGB stars. 100, 30, 1000. 20TD still.")
    # radius_train_GP(datasetRGB, 100, 30, "outputs5438rgb", 1000, target_accept=0.95, sclass="RGB")

    # end_time_CPU3 = time.process_time()

    # mem3 = process.memory_info().rss / 1024**2
    # print(f"Peak Memory: {(mem3-mem2):.2f} MB")
    # print(f"CPU time used: {(end_time_CPU3-start_time_CPU3):.5f} s")
    # print(f"Total run time: {time.time()-start_time3:.5f} s")

    # print("><><><><><><><><><><><><><><><><><><><><><><><><><><")

    # start_time_CPU4 = time.process_time()
    # start_time4 = time.time()

    # print("GP - mass - RGB stars. 100, 30, 1000. 20TD still.")
    # mass_train_GP(datasetRGB, 100, 30, "outputs5438rgb", 1000, target_accept=0.95, sclass="RGB")

    # end_time_CPU4 = time.process_time()

    # mem4 = process.memory_info().rss / 1024**2
    # print(f"Peak Memory: {(mem4-mem3):.2f} MB")
    # print(f"CPU time used: {(end_time_CPU4-start_time_CPU4):.5f} s")
    # print(f"Total run time: {time.time()-start_time4:.5f} s")

    # print("><><><><><><><><><><><><><><><><><><><><><><><><><")
    print("Salve Regina")
