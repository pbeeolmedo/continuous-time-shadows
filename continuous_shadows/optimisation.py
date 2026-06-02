"""
optimisation.py — Finite-Time Optimisation for Continuous-Time Classical Shadows

This module implements the Goldilocks formula and the associated
variance-multiplier utilities from the thesis.

Key results:
  - Goldilocks formula:   u* = max(0, (2m−k) / (2(m+k)))
  - Variance multiplier:  V_{k,m}(u) = 3^{k+m} / [(1−u)^k (1+2u)^m]

Author: Pablo Bee Olmedo
"""

import numpy as np
from typing import Union, Tuple, Dict

from .analytical import (
    u_to_time,
    inv_lambda_X,
    inv_lambda_Z,
)

ArrayLike = Union[float, np.ndarray]


# =============================================================================
# Goldilocks Formula (General Pauli Strings)
# =============================================================================

def goldilocks_u_star(k_eq: int, m_polar: int) -> float:
    """
    Compute the Goldilocks optimal decay parameter u*.

    For an observable with k equatorial (X/Y) and m polar (Z) operators:

        u* = max(0, (2m − k) / (2(m + k)))

    Three regimes:
      - Pure polar (k=0): u* = 1, measure immediately
      - Equatorial-dominated (k ≥ 2m): u* = 0, need full 2-design
      - Mixed Goldilocks (0 < k < 2m): finite-time beats asymptotic!

    Parameters
    ----------
    k_eq : int
        Number of equatorial (X/Y) entries in the Pauli string
    m_polar : int
        Number of polar (Z) entries in the Pauli string

    Returns
    -------
    float
        Optimal decay parameter u*
    """
    if k_eq + m_polar == 0:
        return 0.0  # Identity: any time works
    numerator = 2 * m_polar - k_eq
    denominator = 2 * (m_polar + k_eq)
    return max(0.0, numerator / denominator)


def goldilocks_time(
    k_eq: int,
    m_polar: int,
    gamma_sh: float,
) -> float:
    """
    Convert the Goldilocks u* to a physical optimal time.

        t* = −ln(u*) / γ_sh

    Parameters
    ----------
    k_eq : int
        Number of equatorial operators
    m_polar : int
        Number of polar operators
    gamma_sh : float
        Shadow decay rate

    Returns
    -------
    float
        Optimal scrambling time (np.inf if u* = 0)
    """
    u_star = goldilocks_u_star(k_eq, m_polar)
    if u_star <= 0:
        return np.inf  # need full 2-design
    return float(u_to_time(u_star, gamma_sh))


# =============================================================================
# Variance Multiplier for Pauli Strings
# =============================================================================

def variance_multiplier(k_eq: int, m_polar: int, u: ArrayLike) -> ArrayLike:
    """
    Compute the variance multiplier V_{k,m}(u) for a Pauli string.

        V_{k,m}(u) = 3^{k+m} / ((1−u)^k · (1+2u)^m)

    This is proportional to the variance of the shadow estimator for a
    tensor product of k equatorial and m polar Pauli operators.

    Parameters
    ----------
    k_eq : int
        Number of equatorial (X/Y) entries
    m_polar : int
        Number of polar (Z) entries
    u : float or np.ndarray
        Dimensionless decay parameter

    Returns
    -------
    float or np.ndarray
        Variance multiplier V_{k,m}(u)
    """
    u = np.asarray(u, dtype=float)
    numerator = 3.0**(k_eq + m_polar)
    denominator = (1.0 - u)**k_eq * (1.0 + 2.0 * u)**m_polar
    return numerator / denominator


def variance_multiplier_minimum(k_eq: int, m_polar: int) -> Tuple[float, float]:
    """
    Compute the minimum variance multiplier and where it occurs.

    Parameters
    ----------
    k_eq : int
        Number of equatorial operators
    m_polar : int
        Number of polar operators

    Returns
    -------
    Tuple[float, float]
        (u_star, V_min) — optimal u and minimum variance multiplier
    """
    u_star = goldilocks_u_star(k_eq, m_polar)
    V_min = float(variance_multiplier(k_eq, m_polar, u_star))
    return u_star, V_min


# =============================================================================
# Composite Observable Variance
# =============================================================================

def composite_variance(
    pauli_decomposition: Dict[str, float],
    u: float,
) -> float:
    """
    Compute the variance for a composite observable O = Σ_α c_α P_α.

    Since cross-terms cancel for Pauli observables:

        Var[ô] = Σ_α c_α² / λ_α(u)

    Parameters
    ----------
    pauli_decomposition : dict
        Mapping from Pauli label ("X", "Y", "Z") to coefficient c_α.
    u : float
        Dimensionless decay parameter

    Returns
    -------
    float
        Variance of the shadow estimator
    """
    total = 0.0
    for label, coeff in pauli_decomposition.items():
        label_upper = label.upper()
        if label_upper == "Z":
            total += coeff**2 * float(inv_lambda_Z(u))
        elif label_upper in ("X", "Y"):
            total += coeff**2 * float(inv_lambda_X(u))
        elif label_upper == "I":
            pass  # Identity contributes nothing to variance
        else:
            raise ValueError(f"Unknown Pauli label: {label}")
    return total


def composite_variance_optimal_u(
    pauli_decomposition: Dict[str, float],
    n_points: int = 10000,
) -> Tuple[float, float]:
    """
    Find the optimal u and minimum variance by numerical sweep.

    Parameters
    ----------
    pauli_decomposition : dict
        Mapping from Pauli label to coefficient
    n_points : int
        Number of sweep points

    Returns
    -------
    Tuple[float, float]
        (u_optimal, variance_minimum)
    """
    u_sweep = np.linspace(1e-4, 1.0 - 1e-4, n_points)
    variances = np.array([
        composite_variance(pauli_decomposition, u) for u in u_sweep
    ])
    idx = np.argmin(variances)
    return float(u_sweep[idx]), float(variances[idx])


def weighted_goldilocks_u_star(
    pauli_decomposition: Dict[str, float],
) -> float:
    """
    Closed-form Goldilocks optimum u* for a single-qubit weighted Pauli sum.

    For an observable O = c_X X + c_Y Y + c_Z Z with real coefficients,
    the shadow-norm variance multiplier
    Σ_α c_α² / λ_α(u) = 3(c_X² + c_Y²)/(1-u) + 3 c_Z²/(1+2u)
    is minimised at

        u* = max(0, (r - 1) / (r + 2)),
        r  = |c_Z| · sqrt(2 / (c_X² + c_Y²)).

    See Proposition `prop:weighted_goldilocks` (§5.5 of the thesis).
    Continuous-coefficient analogue of :func:`goldilocks_u_star`, which
    handles only integer-count tensor-product Pauli strings.

    Edge cases:
      - Pure polar (c_X = c_Y = 0):       u* = 1   (no scrambling, V → c_Z²)
      - Equatorial-dominated (r ≤ 1):     u* = 0   (full Haar scrambling)
      - Mixed with r > 1:                 interior u* = (r-1)/(r+2)

    Parameters
    ----------
    pauli_decomposition : dict
        Mapping from Pauli label ("X", "Y", "Z") to real coefficient c_α.
        Identity entries are ignored (they contribute nothing to variance).

    Returns
    -------
    float
        Optimal decay parameter u* ∈ [0, 1].

    Examples
    --------
    >>> weighted_goldilocks_u_star({"X": 1.0, "Y": 1.0, "Z": 3.0})  # r = 3
    0.4
    >>> weighted_goldilocks_u_star({"X": 1.0})                      # equatorial-only
    0.0
    >>> weighted_goldilocks_u_star({"Z": 1.0})                      # pure polar
    1.0
    """
    c_X = pauli_decomposition.get("X", 0.0)
    c_Y = pauli_decomposition.get("Y", 0.0)
    c_Z = pauli_decomposition.get("Z", 0.0)
    equatorial = c_X**2 + c_Y**2
    if equatorial == 0.0:
        return 1.0  # pure polar: u* = 1
    r = abs(c_Z) * np.sqrt(2.0 / equatorial)
    if r <= 1.0:
        return 0.0  # equatorial-dominated: u* = 0
    return (r - 1.0) / (r + 2.0)


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("=== optimisation.py Module Test ===\n")

    # Test Goldilocks formula
    print("Goldilocks formula:")
    test_cases = [
        (0, 1, "Pure Z"),
        (1, 0, "Pure X"),
        (1, 1, "X⊗Z"),
        (2, 1, "X⊗X⊗Z"),
        (1, 2, "X⊗Z⊗Z"),
    ]
    for k_eq, m_pol, name in test_cases:
        u_star = goldilocks_u_star(k_eq, m_pol)
        u_star_val, V_min = variance_multiplier_minimum(k_eq, m_pol)
        print(f"  {name} (k={k_eq}, m={m_pol}): u*={u_star:.3f}, V_min={V_min:.2f}")

    # Verify X⊗Z: u*=1/4, V_min=8
    u_xz, V_xz = variance_multiplier_minimum(1, 1)
    print(f"\nX⊗Z check: u*={u_xz:.4f} (expected 0.25), V_min={V_xz:.2f} (expected 8.0)")

    # Composite variance: X+Z
    print("\nComposite variance for O = X + Z:")
    u_opt, V_opt = composite_variance_optimal_u({"X": 1.0, "Z": 1.0})
    V_haar = composite_variance({"X": 1.0, "Z": 1.0}, 0.0)
    print(f"  Optimal: u*={u_opt:.3f}, V={V_opt:.2f}")
    print(f"  Haar limit (u=0): V={V_haar:.2f}")
