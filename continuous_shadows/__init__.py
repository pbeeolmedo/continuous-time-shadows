"""
continuous_shadows — Continuous-Time Classical Shadow Tomography

A research-grade Python library accompanying the MSc thesis
"Continuous-Time Classical Shadows" (University of Copenhagen, 2026). It
supports both the discrete (Pauli) and continuous-time (stochastic
Hamiltonian) unitary ensembles, with a closed-form eigenvalue engine, a
numerical superoperator engine for validation, and the finite-time
(Goldilocks) variance optimisation.

Modules
-------
core         : Pauli matrices, kets, projectors, vectorisation, QubitSystem
analytical   : Closed-form eigenvalue formulas for the measurement channel
ensembles    : Unitary ensembles (Pauli, stochastic Hamiltonian)
shadows      : Shadow experiment generation and median-of-means estimation
lindblad     : Superoperator algebra and numerical M-matrix (validation engine)
optimisation : Finite-time optimisation (Goldilocks formula, variance multiplier)

Quick Start
-----------
>>> from continuous_shadows import P_0, SIGMA_Z, PauliEnsemble
>>> from continuous_shadows import ShadowExperiment, ShadowEstimator
>>>
>>> exp = ShadowExperiment(P_0, num_runs=1000, ensemble=PauliEnsemble())
>>> df = exp.run()
>>> df = exp.add_snapshots()
>>>
>>> est = ShadowEstimator(df, P_0)
>>> estimate, true_val, error = est.estimate(SIGMA_Z)

Author: Pablo Bee Olmedo. Advisor: Daniel Stilck Franca.
"""

__version__ = "2.1.0"
__author__ = "Pablo Bee Olmedo"

# Core primitives
from .core import (
    QubitSystem,
    I2, I4, H2, S2,
    SIGMA_X, SIGMA_Y, SIGMA_Z,
    PAULI_DICT,
    KET_0, KET_1, KET_PLUS, KET_MINUS, KET_PLUS_I, KET_MINUS_I,
    P_0, P_1, P_PLUS, P_MINUS, P_PLUS_I, P_MINUS_I,
    vectorise, unvectorise,
    density_matrix_to_bloch_vector,
    bloch_vector_to_density_matrix,
)

# Analytical eigenvalue engine
from .analytical import (
    shadow_decay_rate,
    time_to_u, u_to_time,
    lambda_X, lambda_Y, lambda_Z,
    inv_lambda_X, inv_lambda_Y, inv_lambda_Z,
    eigenvalues, eigenvalue_budget,
    analytical_M_matrix, analytical_M_inverse,
    analytical_inverse_channel,
    decompose_pauli, recompose_pauli,
    mixing_matrix,
    pauli_coefficients,
    shadow_norm_pauli, shadow_norm_composite,
)

# Ensembles
from .ensembles import (
    UnitaryEnsemble,
    PauliEnsemble,
    StochasticHamiltonianEnsemble,
    create_ensemble,
)

# Shadow simulation and estimation
from .shadows import (
    ShadowExperiment,
    ShadowEstimator,
    run_shadow_experiment,
    flatten_matrix_columns,
)

# Numerical superoperator engine (validation)
from .lindblad import (
    commutator_superoperator,
    dissipator_superoperator,
    generalised_commutator_superoperator,
    liouvillian_hamiltonian,
    liouvillian_two_copy,
    evolution_propagator,
    evolve_state,
    evolve_trajectory,
    compute_M_matrix,
    compute_M_matrix_analytical,
    validate_analytical_eigenvalues,
    compute_spectral_gap,
)

# Finite-time optimisation
from .optimisation import (
    goldilocks_u_star, goldilocks_time,
    variance_multiplier, variance_multiplier_minimum,
    composite_variance, composite_variance_optimal_u,
    weighted_goldilocks_u_star,
)

__all__ = [
    "__version__",
    # Core
    "QubitSystem",
    "I2", "I4", "H2", "S2",
    "SIGMA_X", "SIGMA_Y", "SIGMA_Z",
    "PAULI_DICT",
    "KET_0", "KET_1", "KET_PLUS", "KET_MINUS", "KET_PLUS_I", "KET_MINUS_I",
    "P_0", "P_1", "P_PLUS", "P_MINUS", "P_PLUS_I", "P_MINUS_I",
    "vectorise", "unvectorise",
    "density_matrix_to_bloch_vector", "bloch_vector_to_density_matrix",
    # Analytical
    "shadow_decay_rate", "time_to_u", "u_to_time",
    "lambda_X", "lambda_Y", "lambda_Z",
    "inv_lambda_X", "inv_lambda_Y", "inv_lambda_Z",
    "eigenvalues", "eigenvalue_budget",
    "analytical_M_matrix", "analytical_M_inverse",
    "analytical_inverse_channel",
    "decompose_pauli", "recompose_pauli",
    "mixing_matrix", "pauli_coefficients",
    "shadow_norm_pauli", "shadow_norm_composite",
    # Ensembles
    "UnitaryEnsemble", "PauliEnsemble", "StochasticHamiltonianEnsemble",
    "create_ensemble",
    # Shadows
    "ShadowExperiment", "ShadowEstimator",
    "run_shadow_experiment", "flatten_matrix_columns",
    # Lindblad
    "commutator_superoperator", "dissipator_superoperator",
    "generalised_commutator_superoperator",
    "liouvillian_hamiltonian", "liouvillian_two_copy",
    "evolution_propagator", "evolve_state", "evolve_trajectory",
    "compute_M_matrix", "compute_M_matrix_analytical",
    "validate_analytical_eigenvalues",
    "compute_spectral_gap",
    # Optimisation
    "goldilocks_u_star", "goldilocks_time",
    "variance_multiplier", "variance_multiplier_minimum",
    "composite_variance", "composite_variance_optimal_u",
    "weighted_goldilocks_u_star",
]
