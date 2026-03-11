Probabilistic Machine Learning Framework for Stellar Mass and Radius Determination in Exoplanet Host Stars

This repository contains the code accompanying the article:

"A Probabilistic Machine Learning Framework for Stellar Mass and Radius Determination in Exoplanet Host Stars"
(manuscript in preparation)

The project implements a Bayesian machine learning framework for predicting stellar masses and radii of exoplanet host stars while fully propagating observational uncertainties and model variance.

Overview

Accurate stellar parameters are essential for determining the physical properties of exoplanets. However, most host stars lack direct measurements of mass and radius.

This repository provides a probabilistic framework that predicts stellar masses and radii from commonly available stellar observables. The method combines multiple probabilistic models through Bayesian hierarchical stacking, producing calibrated predictive distributions that account for both measurement uncertainties and model uncertainty.

The framework integrates three complementary probabilistic models:

Bayesian Additive Regression Trees (BART)

Heteroscedastic Sparse Gaussian Process (GP)

Hierarchical Bayesian Neural Network (HBNN)

These models are combined using Bayesian hierarchical stacking, allowing the final predictions to adaptively weight the strengths of each model across parameter space.

Method Summary

The framework incorporates several key probabilistic elements:

Propagation of measurement uncertainties in stellar observables

Explicit modelling of heteroscedastic predictive variance

Hierarchical structure in the neural network to improve calibration

Sparse Gaussian processes to scale to larger datasets

Bayesian stacking to combine predictive distributions

The result is a set of posterior predictive distributions for stellar mass and radius, rather than single deterministic predictions.

Results (Summary)

On a held-out test set the framework achieves:

Parameter	Mean Absolute Relative Distance
Stellar Mass	3.5%
Stellar Radius	2.1%

When applied to ~2400 exoplanet host stars:

Predictions differ from literature values by 2.7% (MARD) for both mass and radius.

Theoretical consistency improves by a factor of three as measured by the scatter in Δlog g.

Residual analysis reveals interpretable astrophysical trends, including:

Mild underestimation for cool dwarfs

Positive offsets near the main sequence turnoff

These effects likely arise from sparse training coverage and priors in isochrone-based catalogues.
