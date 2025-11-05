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

def GP_4(x_train, x_train_er, y_train):
    
    """
    Construct a 4-dimensional sparse latent Gaussian process model with input uncertainties.

    Builds a hierarchical GP where both the predictive mean (μ) and the log noise variance (σ)
    are modeled as sparse latent Gaussian processes. Each input feature and its uncertainty
    has an individual kernel with InverseGamma-Gamma hyperpriors. 

    Parameters
    ----------
    x_train : pandas.DataFrame
        Training inputs with 4 features (e.g., Teff, logg, Meta, L).
    x_train_er : pandas.DataFrame
        Corresponding measurement errors for each input feature.
    y_train : array-like
        Observed target values.

    Returns
    -------
    tuple
        (gp_model, μ_gp, lg_σ_gp, Xu, Xu_er)
        - gp_model : pm.Model
            The constructed PyMC Gaussian process model.
        - μ_gp : SparseLatent
            Sparse latent GP modeling the predictive mean.
        - lg_σ_gp : SparseLatent
            Sparse latent GP modeling the log noise variance.
        - Xu : ndarray
            Inducing points for the mean process.
        - Xu_er : ndarray
            Inducing points for the uncertainty process.
    """
    
    Xu = np.array(x_train)[1::39]
    Xu_er = np.array(x_train_er)[1::39]
    
    with pm.Model() as gp_model:
        X_mu = pm.MutableData("X_mu", x_train)
        X_er = pm.MutableData("X_er", x_train_er.copy())
        Y = pm.MutableData('y_data', y_train)

        
        ls_logg_mu, ls_logg_sd = get_ℓ_prior(np.array(x_train['logg']))
        ls_teff_mu, ls_teff_sd = get_ℓ_prior(np.array(x_train['Teff']))
        ls_meta_mu, ls_meta_sd = get_ℓ_prior(np.array(x_train['Meta']))
        ls_L_mu, ls_L_sd = get_ℓ_prior(np.array(x_train['L']))
        
        ls_logg = pm.InverseGamma("ls_logg", mu=ls_logg_mu, sigma=ls_logg_sd) 
        eta_logg = pm.Gamma("eta_logg", alpha=2, beta=1) 
        cov_logg = eta_logg**2 * pm.gp.cov.ExpQuad(input_dim=4, ls=ls_logg, active_dims=[1]) + pm.gp.cov.WhiteNoise(sigma=1e-5)
        
        ls_teff = pm.InverseGamma("ls_teff", mu=ls_teff_mu, sigma=ls_teff_sd) 
        eta_teff = pm.Gamma("eta_teff", alpha=2, beta=1) 
        cov_teff = eta_teff**2 * pm.gp.cov.ExpQuad(input_dim=4, ls=ls_teff, active_dims=[0]) + pm.gp.cov.WhiteNoise(sigma=1e-5)
        
        ls_L = pm.InverseGamma("ls_L", mu=ls_L_mu, sigma=ls_L_sd) 
        eta_L = pm.Gamma("eta_L", alpha=2, beta=1) 
        cov_L = eta_L**2 * pm.gp.cov.ExpQuad(input_dim=4, ls=ls_L, active_dims=[3]) + pm.gp.cov.WhiteNoise(sigma=1e-5)
        
        ls_meta = pm.InverseGamma("ls_meta", mu=ls_meta_mu, sigma=ls_meta_sd) 
        eta_meta = pm.Gamma("eta_meta", alpha=2, beta=1) 
        cov_meta = eta_meta**2 * pm.gp.cov.ExpQuad(input_dim=4, ls=ls_meta, active_dims=[2]) + pm.gp.cov.WhiteNoise(sigma=1e-5)
        
        cov1 = cov_teff * cov_logg * cov_meta * cov_L
        
        μ_gp = SparseLatent(cov1)
        μ_f = μ_gp.prior("μ", X_mu, Xu)
        
        ls_elogg_mu, ls_elogg_sd = get_ℓ_prior(np.array(x_train_er['elogg']))
        ls_eteff_mu, ls_eteff_sd = get_ℓ_prior(np.array(x_train_er['eTeff']))
        ls_eL_mu, ls_eL_sd = get_ℓ_prior(np.array(x_train_er['eL']))
        ls_emeta_mu, ls_emeta_sd = get_ℓ_prior(np.array(x_train_er['eMeta']))
        
        ls_elogg = pm.InverseGamma("ls_elogg", mu=ls_elogg_mu, sigma=ls_elogg_sd) 
        eta_elogg = pm.Gamma("eta_elogg", alpha=2, beta=1) 
        cov_elogg = eta_elogg**2 * pm.gp.cov.ExpQuad(input_dim=4, ls=ls_elogg, active_dims=[1]) + pm.gp.cov.WhiteNoise(sigma=1e-5)
        
        ls_eteff = pm.InverseGamma("ls_eteff", mu=ls_eteff_mu, sigma=ls_eteff_sd) 
        eta_eteff = pm.Gamma("eta_eteff", alpha=2, beta=1) 
        cov_eteff = eta_eteff**2 * pm.gp.cov.ExpQuad(input_dim=4, ls=ls_eteff, active_dims=[0]) + pm.gp.cov.WhiteNoise(sigma=1e-5)
        
        ls_eL = pm.InverseGamma("ls_eL", mu=ls_eL_mu, sigma=ls_eL_sd) 
        eta_eL = pm.Gamma("eta_eL", alpha=2, beta=1) 
        cov_eL = eta_eL**2 * pm.gp.cov.ExpQuad(input_dim=4, ls=ls_eL, active_dims=[3]) + pm.gp.cov.WhiteNoise(sigma=1e-5)
        
        ls_emeta = pm.InverseGamma("ls_emeta", mu=ls_emeta_mu, sigma=ls_emeta_sd) 
        eta_emeta = pm.Gamma("eta_emeta", alpha=2, beta=1) 
        cov_emeta = eta_emeta**2 * pm.gp.cov.ExpQuad(input_dim=4, ls=ls_emeta, active_dims=[2]) + pm.gp.cov.WhiteNoise(sigma=1e-5)
        
        cov2 = cov_eteff * cov_elogg * cov_emeta * cov_eL
        
        lg_σ_gp = SparseLatent(cov2)

        lg_σ_f = lg_σ_gp.prior("lg_σ_f", X_er, Xu_er)
        σ_f = pm.Deterministic("σ_f", pm.math.exp(lg_σ_f))

        y = pm.Normal("y", mu=μ_f, sigma=σ_f, observed=Y)
        
        return gp_model, μ_gp, lg_σ_gp, Xu, Xu_er
    
def GP_3(x_train, x_train_er, y_train):
    """
    Construct a 3-dimensional sparse latent Gaussian process model with input uncertainties.

    Similar to `GP_4`, but for inputs with three physical features (e.g., Teff, logg, Meta).
    Models both the GP mean and log noise variance with sparse latent Gaussian processes,
    incorporating input measurement errors into the covariance structure.

    Parameters
    ----------
    x_train : pandas.DataFrame
        Training inputs with 3 features.
    x_train_er : pandas.DataFrame
        Corresponding measurement errors for each input feature.
    y_train : array-like
        Observed target values.

    Returns
    -------
    tuple
        (gp_model, μ_gp, lg_σ_gp, Xu, Xu_er)
        - gp_model : pm.Model
            The constructed PyMC Gaussian process model.
        - μ_gp : SparseLatent
            Sparse latent GP modeling the predictive mean.
        - lg_σ_gp : SparseLatent
            Sparse latent GP modeling the log noise variance.
        - Xu : ndarray
            Inducing points for the mean process.
        - Xu_er : ndarray
            Inducing points for the uncertainty process.
    """
    
    Xu = np.array(x_train)[1::39]
    Xu_er = np.array(x_train_er)[1::39]
    
    with pm.Model() as gp_model:
        X_mu = pm.ConstantData("X_mu", x_train)
        X_er = pm.ConstantData("X_er", x_train_er.copy())
        
        ls_logg_mu, ls_logg_sd = get_ℓ_prior(np.array(x_train['logg']))
        ls_teff_mu, ls_teff_sd = get_ℓ_prior(np.array(x_train['Teff']))
        ls_meta_mu, ls_meta_sd = get_ℓ_prior(np.array(x_train['Meta']))
        
        ls_logg = pm.InverseGamma("ls_logg", mu=ls_logg_mu, sigma=ls_logg_sd) 
        eta_logg = pm.Gamma("eta_logg", alpha=2, beta=1) 
        cov_logg = eta_logg**2 * pm.gp.cov.ExpQuad(input_dim=3, ls=ls_logg, active_dims=[1]) + pm.gp.cov.WhiteNoise(sigma=1e-5)
        
        ls_teff = pm.InverseGamma("ls_teff", mu=ls_teff_mu, sigma=ls_teff_sd) 
        eta_teff = pm.Gamma("eta_teff", alpha=2, beta=1) 
        cov_teff = eta_teff**2 * pm.gp.cov.ExpQuad(input_dim=3, ls=ls_teff, active_dims=[0]) + pm.gp.cov.WhiteNoise(sigma=1e-5)
        
        
        ls_meta = pm.InverseGamma("ls_meta", mu=ls_meta_mu, sigma=ls_meta_sd) 
        eta_meta = pm.Gamma("eta_meta", alpha=2, beta=1) 
        cov_meta = eta_meta**2 * pm.gp.cov.ExpQuad(input_dim=3, ls=ls_meta, active_dims=[2]) + pm.gp.cov.WhiteNoise(sigma=1e-5)
        
        cov1 = cov_teff * cov_logg * cov_meta 
        
        μ_gp = SparseLatent(cov1)
        μ_f = μ_gp.prior("μ", X_mu, Xu)
        
        ls_elogg_mu, ls_elogg_sd = get_ℓ_prior(np.array(x_train_er['elogg']))
        ls_eteff_mu, ls_eteff_sd = get_ℓ_prior(np.array(x_train_er['eTeff']))
        ls_emeta_mu, ls_emeta_sd = get_ℓ_prior(np.array(x_train_er['eMeta']))
        
        ls_elogg = pm.InverseGamma("ls_elogg", mu=ls_elogg_mu, sigma=ls_elogg_sd) 
        eta_elogg = pm.Gamma("eta_elogg", alpha=2, beta=1) 
        cov_elogg = eta_elogg**2 * pm.gp.cov.ExpQuad(input_dim=3, ls=ls_elogg, active_dims=[1]) + pm.gp.cov.WhiteNoise(sigma=1e-5)
        
        ls_eteff = pm.InverseGamma("ls_eteff", mu=ls_eteff_mu, sigma=ls_eteff_sd) 
        eta_eteff = pm.Gamma("eta_eteff", alpha=2, beta=1) 
        cov_eteff = eta_eteff**2 * pm.gp.cov.ExpQuad(input_dim=3, ls=ls_eteff, active_dims=[0]) + pm.gp.cov.WhiteNoise(sigma=1e-5)
        
        
        ls_emeta = pm.InverseGamma("ls_emeta", mu=ls_emeta_mu, sigma=ls_emeta_sd) 
        eta_emeta = pm.Gamma("eta_emeta", alpha=2, beta=1) 
        cov_emeta = eta_emeta**2 * pm.gp.cov.ExpQuad(input_dim=3, ls=ls_emeta, active_dims=[2]) + pm.gp.cov.WhiteNoise(sigma=1e-5)
        
        cov2 = cov_eteff * cov_elogg * cov_emeta 
        
        lg_σ_gp = SparseLatent(cov2)
        lg_σ_f = lg_σ_gp.prior("lg_σ_f", X_er, Xu_er)
        σ_f = pm.Deterministic("σ_f", pm.math.exp(lg_σ_f))

        y = pm.Normal("y", mu=μ_f, sigma=σ_f, observed=y_train)
        
        return gp_model, μ_gp, lg_σ_gp, Xu, Xu_er