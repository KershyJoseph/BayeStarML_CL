#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 15:48:37 2025

@author: LamirelFamily
"""
import numpy as np
import pytensor.tensor as tt
import pymc as pm
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans


class SparseLatent:
    """
    Sparse latent Gaussian process representation for scalable GP models.

    Implements a low-rank GP approximation using inducing points, where the 
    latent function is represented via a smaller set of inducing variables. 
    Provides methods to define the GP prior and compute conditional predictions.

    Parameters
    ----------
    cov_func : callable
        Covariance function (kernel) that computes covariance matrices between inputs.
    """
    def __init__(self, cov_func):
        self.cov = cov_func

    def prior(self, name, X, Xu):
        """
        Define the GP prior using inducing points.

        Constructs the Cholesky factorization of the inducing-point covariance,
        defines latent variables in the rotated space, and computes the GP prior 
        mean for the observed inputs.

        Parameters
        ----------
        name : str
            Base name for PyMC random variables.
        X : tensor
            Training input locations (N × D).
        Xu : tensor
            Inducing point locations (M × D).

        Returns
        -------
        tensor
            GP prior mean evaluated at X.
        """
        Kuu = self.cov(Xu)
        self.L = tt.slinalg.cholesky(pm.gp.util.stabilize(Kuu))

        self.v = pm.Normal(f"u_rotated_{name}", mu=0.0, sigma=1.0, shape=len(Xu))
        self.u = pm.Deterministic(f"u_{name}", tt.dot(self.L, self.v))

        Kfu = self.cov(X, Xu)
        self.Kuiu = tt.slinalg.solve_triangular(
            self.L.T, tt.slinalg.solve_triangular(self.L, self.u, lower=True),
            lower=False
        )
        self.mu = pm.Deterministic(f"mu_{name}", tt.dot(Kfu, self.Kuiu))
        return self.mu

    def conditional(self, name, Xnew, Xu):
        """
        Compute the full conditional predictive distribution at new inputs.

        Parameters
        ----------
        name : str
            Name of the predictive random variable.
        Xnew : tensor
            Test input locations (N* × D).
        Xu : tensor
            Inducing point locations (M × D).

        Returns
        -------
        pm.MvNormal
            Multivariate normal random variable representing GP predictions.
        """
        Ksu = self.cov(Xnew, Xu)
        mus = tt.dot(Ksu, self.Kuiu)
        tmp = tt.slinalg.solve_triangular(self.L, Ksu.T, lower=True)
        Qss = tt.dot(tmp.T, tmp)  # Qss = tt.dot(tt.dot(Ksu, tt.nlinalg.pinv(Kuu)), Ksu.T)
        Kss = self.cov(Xnew)
        Lss = tt.slinalg.cholesky(pm.gp.util.stabilize(Kss - Qss))
        mu_pred = pm.MvNormal(name, mu=mus, chol=Lss, shape=Xnew.eval().shape[0])
        return mu_pred
    
    def conditional_marginal(self, name, Xnew, Xu, jitter=1e-6):
        """
        Compute marginal predictive means and variances at new inputs.

        Returns a Normal random variable for each test point with the GP
        predictive mean and marginal uncertainty. More efficient than the
        full multivariate conditional (O(M·N*) instead of O(N*³)).

        Parameters
        ----------
        name : str
            Name of the predictive random variable.
        Xnew : tensor
            Test input locations (N* × D).
        Xu : tensor
            Inducing point locations (M × D).
        jitter : float, optional
            Numerical stabilizer added to small variances. Default is 1e-6.

        Returns
        -------
        pm.Normal
            PyMC Normal random variable representing marginal predictions.
        """
        Ksu = self.cov(Xnew, Xu)
        # predictive mean:  μ* = K_{x*,u}  K_uu⁻¹  u
        mu_pred = tt.dot(Ksu, self.Kuiu)             

        # marginal variance
        tmp = tt.slinalg.solve_triangular(self.L, Ksu.T, lower=True)
        Kss_diag = self.cov(Xnew, diag=True)         
        var_pred = Kss_diag - tt.sum(tmp**2, axis=0) + jitter
        sigma_pred = tt.sqrt(var_pred)

        return pm.Normal(name, mu=mu_pred, sigma=sigma_pred,
                         shape=Xnew.shape[0])
    
def get_ℓ_prior(points):
    """
    Estimate mean and standard deviation for an InverseGamma prior on the GP length scale.

    Computes empirical lower and upper bounds on the distances between input points
    and derives corresponding parameters for a weakly informative InverseGamma prior.

    Parameters
    ----------
    points : array-like
        One-dimensional array of input coordinates.

    Returns
    -------
    tuple of float
        Mean and standard deviation for the InverseGamma prior on the length scale.
    """
    distances = pdist(points[:, None])
    distinct = distances != 0
    ℓ_l = distances[distinct].min() if sum(distinct) > 0 else 0.1
    ℓ_u = distances[distinct].max() if sum(distinct) > 0 else 1
    ℓ_σ = max(0.1, (ℓ_u - ℓ_l) / 6)
    ℓ_μ = ℓ_l + 3 * ℓ_σ
    return ℓ_μ, ℓ_σ

def _farthest_point_sampling(X, M, seed=0):
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    n = len(X)
    idx = [rng.integers(n)]
    d2 = np.full(n, np.inf)
    for _ in range(1, M):
        d2 = np.minimum(d2, np.sum((X - X[idx[-1]])**2, axis=1))
        idx.append(int(np.argmax(d2)))
    return X[idx]

def make_inducing_points(
    X, X_er=None, M=None, method="kmeans",
    add_bounds=True, weight_by_error=True, seed=0
):
    """
    Assumes X is already standardised (per-feature).
    """
    X = np.asarray(X, dtype=float)
    N, D = X.shape
    if M is None:
        M = max(50, min(N // 10, 800))

    # error-based weights (optional)
    sample_weight = None
    if X_er is not None and weight_by_error:
        er = np.asarray(X_er, dtype=float)
        var = (er**2).mean(axis=1) + 1e-12
        sample_weight = 1.0 / var

    if method == "kmeans":
        km = KMeans(n_clusters=M, n_init=10, random_state=seed)
        km.fit(X, sample_weight=sample_weight)
        Xu = km.cluster_centers_
    else:  # 'fps'
        Xu = _farthest_point_sampling(X, M, seed=seed)

    if add_bounds:
        lb = X.min(axis=0, keepdims=True)
        ub = X.max(axis=0, keepdims=True)
        Xu = np.vstack([lb, Xu, ub])

    return Xu.astype(float)

def make_Xu_er(X_er, M=60, method="kmeans", add_bounds=True, standardise=True, seed=0):
    Xe = np.asarray(X_er, float)
    if standardise:
        mu, sd = Xe.mean(0, keepdims=True), Xe.std(0, keepdims=True) + 1e-12
        Z = (Xe - mu)/sd
    else:
        Z = Xe

    if method == "kmeans":
        km = KMeans(n_clusters=M, n_init=10, random_state=seed)
        km.fit(Z)
        Zu = km.cluster_centers_

    Xu_er = Zu * (sd if standardise else 1) + (mu if standardise else 0)

    if add_bounds:
        lb, ub = Xe.min(0, keepdims=True), Xe.max(0, keepdims=True)
        Xu_er = np.vstack([lb, Xu_er, ub])

    return Xu_er.astype(float)

def sparse_fully_heteroscedastic_gp(
    X, X_err, y,
    M_mean=60,
    M_var=60,
    seed=0,
):
    """
    X      : (N, D) inputs
    X_err  : (N, D) input errors (same D, or D_err different if you want)
    y      : (N,) targets
    """

    X = np.asarray(X, float)
    X_err = np.asarray(X_err, float)
    y = np.asarray(y, float)
    N, D = X.shape
    D_err = X_err.shape[1]

    # Inducing points for mean GP
    Xu = make_inducing_points(X, X_er=X_err, M=M_mean,
                              method="kmeans",
                              add_bounds=True,
                              weight_by_error=True,
                              seed=seed)
    
    X_var = np.hstack([X[:,:2], X_err[:,:2]])  # use only Teff and logg to model log variance
    # X_var = X_err 
    Xu_var = make_inducing_points(X_var, M=M_var,
                                  method="kmeans",
                                  add_bounds=True,
                                  weight_by_error=False,  # maybe off here
                                  seed=seed)
    
    with pm.Model() as model:
        X_mu  = pm.Data("X_mu",  X)
        # X_er  = pm.ConstantData("X_er",  X_err)

        # -------- mean GP: ARD kernel over X --------
        # Build per-dimension ℓ priors, then pack into a vector
        ls_mu_list = []
        ls_sd_list = []
        for d in range(D):
            μ_d, σ_d = get_ℓ_prior(X[:, d])
            ls_mu_list.append(μ_d)
            ls_sd_list.append(σ_d)
        
        ls_mu_vec = np.array(ls_mu_list)
        ls_sd_vec = np.array(ls_sd_list)
        ls = pm.InverseGamma("ls", mu=ls_mu_vec, sigma=ls_sd_vec, shape=D)
        eta = pm.Gamma("eta", alpha=2, beta=1)

        cov_mean = eta**2 * pm.gp.cov.ExpQuad(input_dim=D, ls=ls) \
                   + pm.gp.cov.WhiteNoise(sigma=1e-5)

        μ_gp = SparseLatent(cov_mean)
        μ_f  = μ_gp.prior("μ", X_mu, Xu)

        X_var_data = pm.Data("X_var", X_var)
        D_var = X_var.shape[1]

        # Priors for ARD lengthscales over concatenated space
        ls_v_mu_list, ls_v_sd_list = [], []
        for d in range(D_var):
            μ_d, σ_d = get_ℓ_prior(X_var[:, d])
            ls_v_mu_list.append(μ_d)
            ls_v_sd_list.append(σ_d)

        ls_v_mu_vec = np.array(ls_v_mu_list)
        ls_v_sd_vec = np.array(ls_v_sd_list)

        ls_v  = pm.InverseGamma("ls_var", mu=ls_v_mu_vec, sigma=ls_v_sd_vec, shape=D_var)
        # eta_v = pm.LogNormal("eta_var", mu=np.log(0.2), sigma=0.35)
        eta_v = pm.Gamma("eta_var", alpha=2, beta=1)

        cov_var = eta_v**2 * pm.gp.cov.ExpQuad(input_dim=D_var, ls=ls_v) \
                  + pm.gp.cov.WhiteNoise(sigma=1e-5)
        
        alpha_log_var = pm.Normal("alpha_log_var", mu=0.0, sigma=1.0)
        log_var_gp = SparseLatent(cov_var)
        log_var_latent  = log_var_gp.prior("log_var_f", X_var_data, Xu_var)
        log_var = pm.Deterministic("log_var", alpha_log_var + log_var_latent)
        σ_f     = pm.Deterministic("σ_f", pm.math.exp(0.5 * log_var))

        # -------- likelihood --------
        y_obs = pm.Normal("y", mu=μ_f, sigma=σ_f, observed=y)

    return model, μ_gp, log_var_gp, Xu, Xu_var