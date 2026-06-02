"""
analytical.py — Analytical Eigenvalue Engine for Continuous-Time Classical Shadows

Closed-form expressions for the shadow measurement channel eigenvalues,
derived from the 3×3 invariant subspace of the two-copy Liouvillian.

For isotropic stochastic Pauli noise (κ_X = κ_Y = κ_Z ≡ κ), the full
16×16 two-copy Liouvillian decouples: the subspace {X⊗X, Y⊗Y, Z⊗Z}
forms a closed 3×3 mixing system with eigenvalues η₀ = 0 (steady state)
and η₁ = −3 (doubly degenerate). The shadow decay rate is γ_sh = 12σ²κ.

The eigenvalues of the measurement channel are then exact linear functions
of the dimensionless decay parameter u = exp(−γ_sh·t):

    λ_Z(u) = (1 + 2u) / 3
    λ_X(u) = λ_Y(u) = (1 − u) / 3

These satisfy the eigenvalue budget conservation: λ_X + λ_Y + λ_Z = 1.

Author: Pablo Bee Olmedo

References
----------
- Thesis: "Derivation of the Continuous-Time Shadow Decay Rate"
- Thesis: "March16th_base" §3 (Eigenvalue Dynamics)
"""

import numpy as np
from typing import Union, Tuple, Optional
from dataclasses import dataclass

from .core import (
    I2, SIGMA_X, SIGMA_Y, SIGMA_Z,
    PAULI_DICT, P_0, P_1,
)

ArrayLike = Union[float, np.ndarray]


# =============================================================================
# Shadow Decay Rate
# =============================================================================

def shadow_decay_rate(sigma: float = 1.0, kappa: float = 1.0) -> float:
    """
    Compute the characteristic shadow decay rate γ_sh.

    For isotropic stochastic Pauli noise with diffusion constant σ² and
    per-axis noise strength κ, the shadow measurement channel decays at:

        γ_sh = 12 σ² κ

    This is derived from the non-zero eigenvalue (η₁ = −3) of the 3×3
    mixing matrix, scaled by the prefactor 4σ²κ.

    Parameters
    ----------
    sigma : float
        Noise diffusion constant (default 1.0)
    kappa : float
        Isotropic per-axis noise strength (default 1.0)

    Returns
    -------
    float
        Shadow decay rate γ_sh
    """
    return 12.0 * sigma**2 * kappa


# =============================================================================
# Dimensionless Decay Parameter u
# =============================================================================

def time_to_u(t: ArrayLike, gamma_sh: float) -> ArrayLike:
    """
    Convert physical time to the dimensionless decay parameter.

        u(t) = exp(−γ_sh · t)

    Maps t ∈ [0, ∞) to u ∈ [1, 0]:
      - t = 0  →  u = 1  (bare Z-basis measurement)
      - t → ∞  →  u = 0  (Haar / 2-design limit)

    Parameters
    ----------
    t : float or np.ndarray
        Physical scrambling time(s)
    gamma_sh : float
        Shadow decay rate

    Returns
    -------
    float or np.ndarray
        Decay parameter u
    """
    return np.exp(-gamma_sh * np.asarray(t, dtype=float))


def u_to_time(u: ArrayLike, gamma_sh: float) -> ArrayLike:
    """
    Convert dimensionless decay parameter back to physical time.

        t = −ln(u) / γ_sh

    Parameters
    ----------
    u : float or np.ndarray
        Decay parameter (must be in (0, 1])
    gamma_sh : float
        Shadow decay rate

    Returns
    -------
    float or np.ndarray
        Physical time t
    """
    u = np.asarray(u, dtype=float)
    return -np.log(u) / gamma_sh


# =============================================================================
# Eigenvalue Trajectories (Noiseless, Isotropic)
# =============================================================================

def lambda_Z(u: ArrayLike) -> ArrayLike:
    """
    Eigenvalue of the measurement channel for the Z (polar) Pauli operator.

        λ_Z(u) = (1 + 2u) / 3

    At u = 1 (t = 0): λ_Z = 1  (Z perfectly preserved)
    At u = 0 (t → ∞): λ_Z = 1/3  (Haar limit)

    Parameters
    ----------
    u : float or np.ndarray
        Dimensionless decay parameter

    Returns
    -------
    float or np.ndarray
        Eigenvalue λ_Z
    """
    return (1.0 + 2.0 * np.asarray(u, dtype=float)) / 3.0


def lambda_X(u: ArrayLike) -> ArrayLike:
    """
    Eigenvalue of the measurement channel for the X (equatorial) Pauli operator.

        λ_X(u) = (1 − u) / 3

    At u = 1 (t = 0): λ_X = 0  (X information absent — measurement singularity)
    At u = 0 (t → ∞): λ_X = 1/3  (Haar limit)

    Parameters
    ----------
    u : float or np.ndarray
        Dimensionless decay parameter

    Returns
    -------
    float or np.ndarray
        Eigenvalue λ_X = λ_Y
    """
    return (1.0 - np.asarray(u, dtype=float)) / 3.0


# Alias: Y eigenvalue is identical to X under isotropic noise
lambda_Y = lambda_X


def eigenvalues(u: ArrayLike) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Return all three Pauli eigenvalues (λ_X, λ_Y, λ_Z) at decay parameter u.

    Parameters
    ----------
    u : float or np.ndarray
        Dimensionless decay parameter

    Returns
    -------
    Tuple
        (λ_X(u), λ_Y(u), λ_Z(u))
    """
    lx = lambda_X(u)
    ly = lambda_Y(u)
    lz = lambda_Z(u)
    return lx, ly, lz


def eigenvalue_budget(u: ArrayLike) -> ArrayLike:
    """
    Verify the eigenvalue conservation law: λ_X + λ_Y + λ_Z = 1.

    Parameters
    ----------
    u : float or np.ndarray
        Dimensionless decay parameter

    Returns
    -------
    float or np.ndarray
        Sum of eigenvalues (should be identically 1.0)
    """
    lx, ly, lz = eigenvalues(u)
    return lx + ly + lz


# =============================================================================
# Inverse Eigenvalues (Variance Multipliers)
# =============================================================================

def inv_lambda_Z(u: ArrayLike) -> ArrayLike:
    """
    Inverse eigenvalue for the Z Pauli operator.

        λ_Z⁻¹(u) = 3 / (1 + 2u)

    Parameters
    ----------
    u : float or np.ndarray
        Dimensionless decay parameter

    Returns
    -------
    float or np.ndarray
        Inverse eigenvalue
    """
    return 3.0 / (1.0 + 2.0 * np.asarray(u, dtype=float))


def inv_lambda_X(u: ArrayLike) -> ArrayLike:
    """
    Inverse eigenvalue for the X (and Y) Pauli operator.

        λ_X⁻¹(u) = 3 / (1 − u)

    Diverges as u → 1 (t → 0): the measurement singularity.

    Parameters
    ----------
    u : float or np.ndarray
        Dimensionless decay parameter (must be < 1 to be finite)

    Returns
    -------
    float or np.ndarray
        Inverse eigenvalue
    """
    return 3.0 / (1.0 - np.asarray(u, dtype=float))


inv_lambda_Y = inv_lambda_X


# =============================================================================
# Analytical M-Matrix (Pauli Basis)
# =============================================================================

def analytical_M_matrix(u: float) -> np.ndarray:
    """
    Construct the 4×4 measurement channel matrix in the Pauli basis {I, X, Y, Z}.

    For isotropic stochastic noise, M_t is diagonal in the Pauli basis:

        M(u) = diag(1, λ_X(u), λ_Y(u), λ_Z(u))

    The identity eigenvalue is always 1 (trace preservation).

    Parameters
    ----------
    u : float
        Dimensionless decay parameter

    Returns
    -------
    np.ndarray
        4×4 diagonal matrix
    """
    lx = float(lambda_X(u))
    lz = float(lambda_Z(u))
    return np.diag([1.0, lx, lx, lz]).astype(complex)


def analytical_M_inverse(u: float) -> np.ndarray:
    """
    Construct the inverse measurement channel matrix in the Pauli basis.

        M⁻¹(u) = diag(1, λ_X⁻¹(u), λ_Y⁻¹(u), λ_Z⁻¹(u))

    Parameters
    ----------
    u : float
        Dimensionless decay parameter (must be in (0, 1) for finite inverse)

    Returns
    -------
    np.ndarray
        4×4 diagonal inverse matrix

    Raises
    ------
    ValueError
        If u is too close to 1 (measurement singularity)
    """
    if u >= 1.0 - 1e-12:
        raise ValueError(
            f"Cannot invert M-matrix at u={u:.6f} (measurement singularity at u=1). "
            "The equatorial eigenvalues λ_X = λ_Y = 0 at t=0."
        )
    ilx = float(inv_lambda_X(u))
    ilz = float(inv_lambda_Z(u))
    return np.diag([1.0, ilx, ilx, ilz]).astype(complex)


# =============================================================================
# Pauli Decomposition & Inverse Channel
# =============================================================================

def decompose_pauli(rho: np.ndarray) -> np.ndarray:
    """
    Decompose a 2×2 operator into Pauli coordinates [c_I, c_X, c_Y, c_Z].

        ρ = (c_I · I + c_X · X + c_Y · Y + c_Z · Z) / 2

    where c_α = Tr(σ_α · ρ).

    Parameters
    ----------
    rho : np.ndarray
        2×2 operator

    Returns
    -------
    np.ndarray
        Pauli coefficient vector [c_I, c_X, c_Y, c_Z]
    """
    rho = np.asarray(rho, dtype=complex)
    paulis = [I2, SIGMA_X, SIGMA_Y, SIGMA_Z]
    return np.array([np.trace(P @ rho) for P in paulis], dtype=complex)


def recompose_pauli(coeffs: np.ndarray) -> np.ndarray:
    """
    Reconstruct a 2×2 operator from Pauli coordinates.

        ρ = (c_I · I + c_X · X + c_Y · Y + c_Z · Z) / 2

    Parameters
    ----------
    coeffs : np.ndarray
        Pauli coefficient vector [c_I, c_X, c_Y, c_Z]

    Returns
    -------
    np.ndarray
        2×2 operator
    """
    paulis = [I2, SIGMA_X, SIGMA_Y, SIGMA_Z]
    return sum(c * P for c, P in zip(coeffs, paulis)) / 2.0


def analytical_inverse_channel(P_lab: np.ndarray, u: float) -> np.ndarray:
    """
    Apply the analytical inverse measurement channel to a lab-frame projector.

    Given a measurement outcome projector P_lab = U†|b⟩⟨b|U, compute
    the classical shadow estimator ρ̂ = M_t⁻¹(P_lab).

    The inversion is performed in the Pauli basis:
      1. Decompose P_lab into Pauli coordinates
      2. Multiply each coordinate by the inverse eigenvalue
      3. Recompose into a 2×2 operator

    Parameters
    ----------
    P_lab : np.ndarray
        Lab-frame projector (2×2)
    u : float
        Dimensionless decay parameter at the measurement time

    Returns
    -------
    np.ndarray
        Classical shadow estimator ρ̂ (2×2)
    """
    coeffs = decompose_pauli(P_lab)
    inv_eigenvals = np.array([
        1.0,
        float(inv_lambda_X(u)),
        float(inv_lambda_X(u)),
        float(inv_lambda_Z(u)),
    ], dtype=complex)
    return recompose_pauli(coeffs * inv_eigenvals)


# =============================================================================
# The 3×3 Mixing Matrix (Pedagogical / Validation)
# =============================================================================

@dataclass
class MixingMatrixResult:
    """Container for the analytical 3×3 mixing matrix and its spectrum."""
    matrix: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    decay_rate: float


def mixing_matrix(sigma: float = 1.0, kappa: float = 1.0) -> MixingMatrixResult:
    """
    Construct the 3×3 mixing matrix governing {X⊗X, Y⊗Y, Z⊗Z} evolution.

    The coupled ODE system is:

        d/dt [c_X, c_Y, c_Z]ᵀ = 4σ²κ · A · [c_X, c_Y, c_Z]ᵀ

    where A = [[-2, 1, 1], [1, -2, 1], [1, 1, -2]].

    The eigenvalues of A are {0, −3, −3}. The physical decay rate is
    γ_sh = 4σ²κ × 3 = 12σ²κ.

    Parameters
    ----------
    sigma : float
        Noise diffusion constant
    kappa : float
        Isotropic per-axis noise strength

    Returns
    -------
    MixingMatrixResult
        The mixing matrix, its eigenvalues, eigenvectors, and decay rate
    """
    prefactor = 4.0 * sigma**2 * kappa

    A = np.array([
        [-2,  1,  1],
        [ 1, -2,  1],
        [ 1,  1, -2],
    ], dtype=float)

    M = prefactor * A

    evals, evecs = np.linalg.eigh(M)
    # Sort by magnitude (ascending): 0, −12σ²κ, −12σ²κ
    idx = np.argsort(np.abs(evals))
    evals = evals[idx]
    evecs = evecs[:, idx]

    gamma_sh = shadow_decay_rate(sigma, kappa)

    return MixingMatrixResult(
        matrix=M,
        eigenvalues=evals,
        eigenvectors=evecs,
        decay_rate=gamma_sh,
    )


# =============================================================================
# Solving the ODE: Exact Coefficients c_X(t), c_Y(t), c_Z(t)
# =============================================================================

def pauli_coefficients(
    t: ArrayLike,
    gamma_sh: float,
    c0: Tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Solve the 3×3 mixing ODE for the two-copy Pauli coefficients.

    Given initial condition Q(0) = ½(I⊗I + Z⊗Z), the Z⊗Z component has
    c_Z(0) = 1, c_X(0) = c_Y(0) = 0. The solution is:

        c_Z(t) = 1/3 + 2/3 · exp(−γ_sh·t)
        c_X(t) = c_Y(t) = 1/3 − 1/3 · exp(−γ_sh·t)

    These are exactly the eigenvalues λ_Z(t) and λ_X(t) of the measurement
    channel. For general initial conditions the solution mixes the steady-state
    (1/3, 1/3, 1/3) with the transient eigenmodes.

    Parameters
    ----------
    t : float or np.ndarray
        Time point(s)
    gamma_sh : float
        Shadow decay rate
    c0 : Tuple[float, float, float]
        Initial condition (c_X(0), c_Y(0), c_Z(0)). Default is (0, 0, 1).

    Returns
    -------
    Tuple
        (c_X(t), c_Y(t), c_Z(t))
    """
    t = np.asarray(t, dtype=float)
    u = np.exp(-gamma_sh * t)

    # Steady-state component (eigenvalue η₀ = 0)
    c_bar = sum(c0) / 3.0

    # Transient deviations from steady state
    dx0 = c0[0] - c_bar
    dy0 = c0[1] - c_bar
    dz0 = c0[2] - c_bar

    c_X = c_bar + dx0 * u
    c_Y = c_bar + dy0 * u
    c_Z = c_bar + dz0 * u

    return c_X, c_Y, c_Z


# =============================================================================
# Shadow Norm (Analytical)
# =============================================================================

def shadow_norm_pauli(observable_label: str, u: float) -> float:
    """
    Compute the shadow norm for a single Pauli observable.

    The shadow norm (worst-case second moment) for a single-qubit Pauli P_α is:

        ||P_α||²_shadow = 3   (at the Haar limit, u = 0)

    At finite time, the shadow norm scales as λ_α⁻¹(u).

    Parameters
    ----------
    observable_label : str
        "X", "Y", or "Z"
    u : float
        Dimensionless decay parameter

    Returns
    -------
    float
        Shadow norm squared
    """
    label = observable_label.upper()
    if label == "Z":
        return float(inv_lambda_Z(u))
    elif label in ("X", "Y"):
        return float(inv_lambda_X(u))
    else:
        raise ValueError(f"Unknown Pauli label: {observable_label}")


def shadow_norm_composite(
    coefficients: dict,
    u: float,
) -> float:
    """
    Compute the shadow norm for a composite observable O = Σ_α c_α P_α.

    By Pauli anticommutation, cross-terms vanish:

        ||O||²_shadow = Σ_α c_α² / λ_α(u)

    Parameters
    ----------
    coefficients : dict
        Mapping from Pauli label to coefficient, e.g. {"X": 1.0, "Z": 1.0}
    u : float
        Dimensionless decay parameter

    Returns
    -------
    float
        Shadow norm squared
    """
    total = 0.0
    for label, c in coefficients.items():
        total += c**2 * shadow_norm_pauli(label, u)
    return total


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("=== analytical.py Module Test ===\n")

    sigma, kappa = 1.0, 1.0
    gamma_sh = shadow_decay_rate(sigma, kappa)
    print(f"Shadow decay rate γ_sh = {gamma_sh:.1f}  (expected 12.0)")

    # Test eigenvalue trajectories at key points
    print("\nEigenvalue trajectories:")
    for u_val in [1.0, 0.5, 0.25, 0.0]:
        lx = lambda_X(u_val)
        lz = lambda_Z(u_val)
        budget = float(eigenvalue_budget(u_val))
        print(f"  u={u_val:.2f}: λ_X={lx:.4f}, λ_Z={lz:.4f}, budget={budget:.6f}")

    # Test eigenvalue budget conservation
    u_test = np.linspace(0, 1, 1000)
    budget_err = np.max(np.abs(eigenvalue_budget(u_test) - 1.0))
    print(f"\nBudget conservation max error: {budget_err:.2e}")

    # Test M-matrix at Haar limit
    print("\nM-matrix at u=0 (Haar limit):")
    M0 = analytical_M_matrix(0.0)
    print(f"  diag = {np.diag(M0).real}")

    # Test inverse channel at Haar limit (should give 3P - I)
    P_test = P_0  # |0⟩⟨0|
    rho_hat = analytical_inverse_channel(P_test, 0.0)
    rho_hat_expected = 3.0 * P_test - I2
    print(f"\nInverse channel at u=0: matches 3P−I = {np.allclose(rho_hat, rho_hat_expected)}")

    # Test 3×3 mixing matrix
    mix = mixing_matrix(sigma, kappa)
    print(f"\nMixing matrix eigenvalues: {mix.eigenvalues}")
    print(f"Shadow decay rate from mixing: {mix.decay_rate:.1f}")

    # Test Pauli coefficients (ODE solution)
    t_test = np.array([0.0, 0.1, 1.0])
    cx, cy, cz = pauli_coefficients(t_test, gamma_sh)
    print(f"\nPauli coefficients at t=0: c_X={cx[0]:.4f}, c_Z={cz[0]:.4f}")
    print(f"Pauli coefficients at t=∞: c_X={float(lambda_X(0.0)):.4f}, c_Z={float(lambda_Z(0.0)):.4f}")

    # Test shadow norm
    print(f"\nShadow norm ||Z||² at u=0: {shadow_norm_pauli('Z', 0.0):.1f}  (expected 3.0)")
    print(f"Shadow norm ||X+Z||² at u=0: {shadow_norm_composite({'X': 1, 'Z': 1}, 0.0):.1f}  (expected 6.0)")

    # Verify against the Goldilocks minimum for X⊗Z
    u_fine = np.linspace(0.001, 0.999, 10000)
    V_XZ = inv_lambda_X(u_fine) * inv_lambda_Z(u_fine)
    u_min = u_fine[np.argmin(V_XZ)]
    V_min = np.min(V_XZ)
    print(f"\nGoldilocks for X⊗Z: u* ≈ {u_min:.3f} (expected 0.250), V_min ≈ {V_min:.3f} (expected 8.0)")
