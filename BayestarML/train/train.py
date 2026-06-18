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
        "NN_0.5He_"
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

    print("700ms R NN. 0.5*He priors. 0.2 non-centred er prior. 0.05 bias priors. NUTPIE with init_means. 8_2000")
    train_NN(
        "700ms", "R", 8, draws=2000, nutpie=True
    )

    end_time_CPU = time.process_time()
    mem1 = process.memory_info().rss / 1024**2
    print(f"Peak Memory: {mem1:.2f} MB")
    print(f"CPU time accumulated: {(end_time_CPU - start_time_CPU):.5f} s")
    print(f"Total wall clock time: {time.perf_counter() - start_time_wall:.5f} s")

    print("><><><><><><><><><><><><><><><><><><><><><><><><><><")

    # start_time_CPU2 = time.process_time()
    # start_time2 = time.time()

    # print("RGB logR NN. 0.01He priors. 8_1400_0.95")
    # train_NN(
    #     "5438rgb", "logR", 8, draws=1400, nutpie=True
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


# The reason you are hitting PyMC warnings about "old GP objects" is because you've coded up a brilliant, custom implementation of a Sparse Heteroscedastic GP from scratch, rather than using PyMC's built-in pm.gp high-level classes.

# Because you aren't using their native pm.gp.Marginal or pm.gp.Latent wrapper classes to hold your models, PyMC's pm.sample_posterior_predictive doesn't inherently understand what your SparseLatent class is doing. When you reconstruct f_mu_pred and log_var_pred_latent inside the with gp_model: block during prediction, PyMC scans the context, notices that new random variables are being hooked up to older PyTensor computational graph fragments (μ_trace and var_trace), and panics with a warning thinking you've leaked an old model's state.

# Passing gp_trace around explicitly was a great instinct—and it works mathematically—but it triggers PyMC's internal guardrails because the graph is being modified dynamically after sampling.

# Here is the cleanest way to completely silence that warning and make your architecture robust, without completely rewriting your custom logic.

# The Fix: Decouple PPC from the Model Context
# When you use your own custom math for predictive conditionals, you don't actually need to inject those predictive variables back into the training gp_model context for sample_posterior_predictive.

# Instead, you can compile your conditional math directly into a clean, standalone PyTensor Function. This approach is faster, completely bypasses PyMC's graph tracking warnings, and treats your posterior predictive step as a pure mathematical mapping.

# Here is how you update your posterior_predictive_GP function:

# Python
# import pytensor

# def posterior_predictive_GP(
#     gp_model, mu_gp, log_var_gp, μ_trace, var_trace, trace,
#     X_new_raw, X_er_new_raw, Xu, Xu_var,
#     n_param, target,
#     var_cols_x=(0,1),
#     var_cols_xerr=(0,1),
#     random_seed=42,
# ):
#     lpd_GP = find_pointwise_loo(trace)
#     X_new_raw = np.asarray(X_new_raw, float)
#     X_er_new_raw = np.asarray(X_er_new_raw, float)
#     N_new = X_new_raw.shape[0]

#     # 1. We create a fresh, isolated model container just for compiling the prediction math
#     with pm.Model() as pred_model:
#         X_mu_obs = pm.Data("X_mu_obs", X_new_raw)
#         X_var_obs = pm.Data("X_var_obs", X_var_new_raw)

#         # Re-handle your missingness masks locally
#         mask_mu = ~np.isfinite(X_new_raw)
#         X_var_new_raw = np.hstack([X_new_raw[:, var_cols_x], X_er_new_raw[:, var_cols_xerr]])
#         mask_var = ~np.isfinite(X_var_new_raw)
#         D_var = X_var_new_raw.shape[1]

#         X_mu_latent = pm.Normal("X_new_latent", mu=0.0, sigma=1.0, shape=(N_new, n_param))
#         X_var_latent = pm.Normal("X_var_new_latent", mu=0.0, sigma=1.0, shape=(N_new, D_var))

#         X_new = tt.where(mask_mu, X_mu_latent, X_mu_obs)
#         X_var_new = tt.where(mask_var, X_var_latent, X_var_obs)

#         # Calculate math conditionals using your classes
#         f_mu_pred = mu_gp.conditional_marginal("f_mu_pred", X_new, Xu, gp_trace=μ_trace)
#         log_var_pred_latent = log_var_gp.conditional_marginal(
#             "log_var_pred_latent", X_var_new, Xu_var, gp_trace=var_trace
#         )

#         # Pull parameters safely via the original model graph names
#         log_var_pred = gp_model["alpha_log_var"] + log_var_pred_latent
#         sigma_pred = pm.math.exp(0.5 * log_var_pred)

#     # 2. Compile an explicit PyTensor function.
#     # This takes your posterior sample inputs and maps them directly to your outputs.
#     # We find all free parameters required by the graph (ls, eta, alpha_log_var, etc.)
#     input_rvs = [v for v in pred_model.free_RVs if v not in [X_mu_latent, X_var_latent]]
#     # Add the parent variables from your training model that are required
#     input_rvs.extend([gp_model["alpha_log_var"]])

#     # Compile the mathematical graph directly
#     print("Compiling predictive math graph...")
#     predict_fn = pytensor.function(inputs=input_rvs, outputs=[f_mu_pred, sigma_pred])

#     # 3. Vectorized Evaluation over the Trace
#     # Instead of calling sample_posterior_predictive, loop or map over your posteriors safely
#     posterior_samples = trace.posterior.stack(sample=("chain", "draw"))

#     y_draws_list = []
#     rng = np.random.default_rng(random_seed)

#     for i in range(len(posterior_samples.sample)):
#         sample = posterior_samples.isel(sample=i)

#         # Extract values from trace to pass to the compiled graph
#         feed_dict = {}
#         for r_var in input_rvs:
#             feed_dict[r_var] = sample[r_var.name].values

#         # Compute mu and sigma for this specific posterior draw
#         mu_val, sigma_val = predict_fn(**feed_dict)

#         # Sample your final observed y analytically
#         y_sample = rng.normal(loc=mu_val, scale=sigma_val)
#         y_draws_list.append(y_sample)

#     y_draws = np.array(y_draws_list) # Shape: (samples, N_test)

#     return denormalise_val(y_draws, target), lpd_GP
# Why this solves your issue completely
# No More Warning Contexts: By isolating the prediction step inside with pm.Model() as pred_model:, you prevent PyMC from scanning your original training context and seeing "mutated" or "redefined" GP operations.

# Deterministic Inputs: Compiling via pytensor.function converts your custom conditional logic directly into a fast C/JAX compiled execution loop.

# Bypasses the Bug entirely: PyMC's internal tracking triggers warning flags whenever custom classes reference external tensors. Converting it to an explicit function execution makes it immune to these namespace checks.
