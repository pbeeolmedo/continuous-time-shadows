"""
lindblad.py — Open Systems & Lindbladian Dynamics Module

This module provides tools for analysing the Lindbladian dynamics of classical
shadow tomography in open quantum systems. It includes:

- Liouvillian superoperator construction
- M-matrix computation for continuous-time shadows
- Spectral gap analysis
- Two-copy (ρ⊗ρ) dynamics for variance analysis

Author: Pablo Bee Olmedo
"""

import numpy as np
from scipy.linalg import expm, eigvals
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

from .core import (
    I2, I4, 
    SIGMA_X, SIGMA_Y, SIGMA_Z,
    KET_0, KET_1,
    P_0, P_1,
    vectorise, unvectorise,
)
from .analytical import (
    shadow_decay_rate,
    time_to_u,
    analytical_M_matrix,
    analytical_M_inverse,
    eigenvalues as analytical_eigenvalues,
)


# =============================================================================
# Superoperator Construction
# =============================================================================

def commutator_superoperator(H: np.ndarray) -> np.ndarray:
    """
    Construct the commutator superoperator K(H).
    
    The commutator superoperator satisfies:
        vec([H, ρ]) = K(H) @ vec(ρ)
    
    Explicitly:
        K(H) = I ⊗ H - H^T ⊗ I
    
    Parameters
    ----------
    H : np.ndarray
        Hermitian operator (typically a Hamiltonian)
    
    Returns
    -------
    np.ndarray
        Superoperator K(H) with shape (d², d²)
    
    Notes
    -----
    For a d×d operator H, the superoperator has dimension d²×d².
    Uses column-major (Fortran) vectorisation convention.
    """
    d = H.shape[0]
    I_d = np.eye(d, dtype=complex)
    K = np.kron(I_d, H) - np.kron(H.T, I_d)
    return K


def dissipator_superoperator(L: np.ndarray) -> np.ndarray:
    """
    Construct the dissipator superoperator D(L) for jump operator L.
    
    The Lindblad dissipator is:
        D[L](ρ) = L ρ L† - ½{L†L, ρ}
    
    In superoperator form:
        D(L) = L* ⊗ L - ½(I ⊗ L†L + (L†L)^T ⊗ I)
    
    Parameters
    ----------
    L : np.ndarray
        Jump (Lindblad) operator
    
    Returns
    -------
    np.ndarray
        Dissipator superoperator D(L)
    """
    d = L.shape[0]
    I_d = np.eye(d, dtype=complex)
    L_dag_L = L.conj().T @ L
    
    # L ρ L† term: (L*)⊗L in column-major convention
    sandwich_term = np.kron(L.conj(), L)
    
    # -½{L†L, ρ} term
    anticomm_term = -0.5 * (np.kron(I_d, L_dag_L) + np.kron(L_dag_L.T, I_d))
    
    return sandwich_term + anticomm_term


def generalised_commutator_superoperator(K: np.ndarray) -> np.ndarray:
    """
    Construct the generalised commutator superoperator for two-copy dynamics.
    
    For two-copy evolution on ρ⊗ρ:
        C(K) = I ⊗ K - K^T ⊗ I
    
    where K is already a superoperator (d²×d²), giving C(K) with shape (d⁴×d⁴).
    
    Parameters
    ----------
    K : np.ndarray
        Superoperator (d²×d²)
    
    Returns
    -------
    np.ndarray
        Two-copy superoperator C(K) with shape (d⁴, d⁴)
    """
    dim = K.shape[0]
    I_dim = np.eye(dim, dtype=complex)
    C = np.kron(I_dim, K) - np.kron(K.T, I_dim)
    return C


# =============================================================================
# Liouvillian Construction
# =============================================================================

def liouvillian_hamiltonian(
    H_list: List[np.ndarray],
    sigma: float = 1.0
) -> np.ndarray:
    """
    Construct the Liouvillian for stochastic Hamiltonian evolution.
    
    For the stochastic Schrödinger equation with Hamiltonians {Hₖ}:
        𝓛 = -σ²/2 Σₖ [Hₖ, [Hₖ, ·]]
    
    In superoperator form:
        L = -σ²/2 Σₖ K(Hₖ)²
    
    Parameters
    ----------
    H_list : List[np.ndarray]
        List of Hamiltonian operators {H₁, H₂, ...}
    sigma : float
        Noise strength parameter
    
    Returns
    -------
    np.ndarray
        Liouvillian superoperator L
    
    Notes
    -----
    This generates the averaged dynamics for the stochastic evolution.
    """
    d = H_list[0].shape[0]
    n = d * d
    L = np.zeros((n, n), dtype=complex)
    
    for H in H_list:
        K = commutator_superoperator(H)
        L += K @ K
    
    L *= -sigma**2 / 2
    return L


def liouvillian_two_copy(
    H_list: List[np.ndarray],
    sigma: float = 1.0
) -> np.ndarray:
    """
    Construct the two-copy Liouvillian for variance analysis.
    
    For analysing Var[Tr(O ρ̂)] we need the dynamics on ρ⊗ρ:
        𝓛² = -σ²/2 Σₖ C(Kₖ)²
    
    Parameters
    ----------
    H_list : List[np.ndarray]
        List of Hamiltonian operators
    sigma : float
        Noise strength
    
    Returns
    -------
    np.ndarray
        Two-copy Liouvillian with shape (d⁴, d⁴)
    """
    d = H_list[0].shape[0]
    n = d * d
    n2 = n * n  # d⁴ for two copies
    
    L2 = np.zeros((n2, n2), dtype=complex)
    
    for H in H_list:
        K = commutator_superoperator(H)
        C = generalised_commutator_superoperator(K)
        L2 += C @ C
    
    L2 *= -sigma**2 / 2
    return L2


# =============================================================================
# Time Evolution
# =============================================================================

def evolution_propagator(L: np.ndarray, t: float = 1.0) -> np.ndarray:
    """
    Compute the evolution propagator e^{tL}.
    
    Parameters
    ----------
    L : np.ndarray
        Liouvillian superoperator
    t : float
        Evolution time
    
    Returns
    -------
    np.ndarray
        Propagator e^{tL}
    """
    return expm(t * L)


def evolve_state(
    rho_0: np.ndarray,
    L: np.ndarray,
    t: float
) -> np.ndarray:
    """
    Evolve a density matrix under Lindbladian dynamics.
    
    Parameters
    ----------
    rho_0 : np.ndarray
        Initial state (d×d)
    L : np.ndarray
        Liouvillian superoperator (d²×d²)
    t : float
        Evolution time
    
    Returns
    -------
    np.ndarray
        Final state ρ(t) = e^{tL}[ρ₀]
    """
    d = rho_0.shape[0]
    vec_rho_0 = vectorise(rho_0)
    vec_rho_t = expm(t * L) @ vec_rho_0
    return unvectorise(vec_rho_t, dim=d)


def evolve_trajectory(
    rho_0: np.ndarray,
    L: np.ndarray,
    times: np.ndarray
) -> List[np.ndarray]:
    """
    Evolve a density matrix and return trajectory at specified times.
    
    Parameters
    ----------
    rho_0 : np.ndarray
        Initial state
    L : np.ndarray
        Liouvillian
    times : np.ndarray
        Array of time points
    
    Returns
    -------
    List[np.ndarray]
        List of density matrices at each time point
    """
    return [evolve_state(rho_0, L, t) for t in times]


# =============================================================================
# M-Matrix Construction
# =============================================================================

@dataclass
class MMatrixResult:
    """Container for M-matrix computation results."""
    M: np.ndarray
    M_inv: Optional[np.ndarray]
    t: float
    is_invertible: bool
    condition_number: float


def build_basis_operators() -> dict:
    """
    Build the basis operators B_α = |i⟩⟨j| for α ∈ {0,1,2,3}.
    
    Returns
    -------
    dict
        Dictionary mapping index to basis operator
    """
    return {
        0: np.outer(KET_0, KET_0.conj()),  # |0⟩⟨0|
        1: np.outer(KET_0, KET_1.conj()),  # |0⟩⟨1|
        2: np.outer(KET_1, KET_0.conj()),  # |1⟩⟨0|
        3: np.outer(KET_1, KET_1.conj()),  # |1⟩⟨1|
    }


def compute_Q_t(
    t: float,
    L2: np.ndarray,
    Q_0: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute Q(t) for the M-matrix.
    
    Q(t) = e^{L²t}[Q₀] where Q₀ = |0⟩⟨0|⊗|0⟩⟨0| + |1⟩⟨1|⊗|1⟩⟨1|
    
    Parameters
    ----------
    t : float
        Evolution time
    L2 : np.ndarray
        Two-copy Liouvillian
    Q_0 : np.ndarray, optional
        Initial Q operator. If None, uses standard form.
    
    Returns
    -------
    np.ndarray
        Q(t) as a 4×4 matrix
    """
    if Q_0 is None:
        Q_0 = np.kron(P_0, P_0) + np.kron(P_1, P_1)
    
    vec_Q_0 = vectorise(Q_0)
    vec_Q_t = expm(L2 * t) @ vec_Q_0
    Q_t = unvectorise(vec_Q_t, dim=4)
    
    return Q_t


def compute_M_element(
    alpha: int,
    beta: int,
    Q_t: np.ndarray,
    B: dict
) -> complex:
    """
    Compute a single element of the M-matrix.
    
    M_αβ = Tr[(B_α ⊗ I) Q(t) (I ⊗ B_β)]
    
    Parameters
    ----------
    alpha, beta : int
        Matrix indices (0-3)
    Q_t : np.ndarray
        Q(t) matrix (4×4)
    B : dict
        Basis operators
    
    Returns
    -------
    complex
        Matrix element M_αβ
    """
    B_alpha_tensor_I = np.kron(B[alpha], I2)
    I_tensor_B_beta = np.kron(I2, B[beta])
    return np.trace(B_alpha_tensor_I @ Q_t @ I_tensor_B_beta)


def compute_M_matrix(
    t: float,
    L2: np.ndarray,
    compute_inverse: bool = True
) -> MMatrixResult:
    """
    Compute the full 4×4 M-matrix at time t.
    
    The M-matrix encodes the measurement channel for classical shadows
    and is used to construct the inverse channel.
    
    Parameters
    ----------
    t : float
        Evolution time
    L2 : np.ndarray
        Two-copy Liouvillian (16×16)
    compute_inverse : bool
        Whether to compute M⁻¹
    
    Returns
    -------
    MMatrixResult
        Result containing M, M_inv, and metadata
    """
    B = build_basis_operators()
    Q_t = compute_Q_t(t, L2)
    
    M = np.zeros((4, 4), dtype=complex)
    for alpha in range(4):
        for beta in range(4):
            M[alpha, beta] = compute_M_element(alpha, beta, Q_t, B)
    
    # Compute condition number and inverse
    cond = np.linalg.cond(M)
    is_invertible = cond < 1e10
    
    M_inv = None
    if compute_inverse and is_invertible:
        M_inv = np.linalg.inv(M)
    
    return MMatrixResult(
        M=M,
        M_inv=M_inv,
        t=t,
        is_invertible=is_invertible,
        condition_number=cond
    )


def compute_M_matrix_analytical(
    t: float,
    sigma: float = 1.0,
    kappa: float = 1.0,
) -> MMatrixResult:
    """
    Compute the M-matrix using closed-form analytical eigenvalue formulas.

    This is the fast path: O(1) instead of the O(d⁴) numerical expm.
    For isotropic stochastic Pauli noise, the M-matrix is diagonal in
    the Pauli basis.

    Parameters
    ----------
    t : float
        Evolution time
    sigma : float
        Noise diffusion constant
    kappa : float
        Isotropic noise strength

    Returns
    -------
    MMatrixResult
        M-matrix and its inverse (always invertible for t > 0)
    """
    gamma_sh = shadow_decay_rate(sigma, kappa)
    u = float(time_to_u(t, gamma_sh))

    M = analytical_M_matrix(u)

    M_inv = None
    is_invertible = u < 1.0 - 1e-12
    if is_invertible:
        M_inv = analytical_M_inverse(u)

    cond = np.linalg.cond(M)

    return MMatrixResult(
        M=M,
        M_inv=M_inv,
        t=t,
        is_invertible=is_invertible,
        condition_number=cond,
    )


def _computational_to_pauli_M(M_comp: np.ndarray) -> np.ndarray:
    """
    Convert a 4x4 M-matrix from the computational basis to the Pauli basis.

    T converts vec_comp to vec_pauli (r_alpha = Tr(P_alpha rho)), so
    M_pauli = T @ M_comp @ T^{-1}.

    Parameters
    ----------
    M_comp : np.ndarray
        4x4 M-matrix in the computational basis.

    Returns
    -------
    np.ndarray
        4x4 M-matrix in the Pauli basis {I, X, Y, Z}.
    """
    T = np.array([
        [1,  0,   0,  1],
        [0,  1,   1,  0],
        [0, -1j, 1j,  0],
        [1,  0,   0, -1],
    ], dtype=complex)
    return T @ M_comp @ np.linalg.inv(T)


def validate_analytical_eigenvalues(
    sigma: float = 1.0,
    kappa: float = 1.0,
    times: Optional[np.ndarray] = None,
) -> dict:
    """
    Compare analytical eigenvalue formulas against the 16×16 numerical
    M-matrix computation.

    This serves as ground-truth validation that the 3×3 mixing matrix
    reduction and closed-form eigenvalue formulas are correct.

    Parameters
    ----------
    sigma : float
        Noise diffusion constant
    kappa : float
        Isotropic noise strength
    times : np.ndarray, optional
        Time points to compare (default: 100 points from 0.01 to 3.0)

    Returns
    -------
    dict
        Contains 'times', 'analytical_eigenvalues', 'numerical_eigenvalues',
        'max_residual', 'all_close'
    """
    if times is None:
        times = np.linspace(0.01, 3.0, 100)

    paulis = [SIGMA_X, SIGMA_Y, SIGMA_Z]
    L2 = liouvillian_two_copy(paulis, sigma)

    gamma_sh = shadow_decay_rate(sigma, kappa)

    anal_eigs = []
    num_eigs = []

    for t in times:
        # Analytical
        u = float(time_to_u(t, gamma_sh))
        lx, ly, lz = analytical_eigenvalues(u)
        anal_eigs.append([float(lx), float(ly), float(lz)])

        # Numerical
        result = compute_M_matrix(t, L2)
        M = result.M
        # Extract eigenvalues from numerical M-matrix (Pauli basis)
        # For isotropic noise, the numerical M should be approximately diagonal
        # in the Pauli basis. Extract via the change-of-basis.
        M_pauli = _computational_to_pauli_M(M)
        num_eigs.append([
            float(np.real(M_pauli[1, 1])),  # λ_X
            float(np.real(M_pauli[2, 2])),  # λ_Y
            float(np.real(M_pauli[3, 3])),  # λ_Z
        ])

    anal_eigs = np.array(anal_eigs)
    num_eigs = np.array(num_eigs)

    residuals = np.abs(anal_eigs - num_eigs)
    max_residual = float(np.max(residuals))

    return {
        'times': times,
        'analytical_eigenvalues': anal_eigs,
        'numerical_eigenvalues': num_eigs,
        'residuals': residuals,
        'max_residual': max_residual,
        'all_close': max_residual < 1e-8,
    }


# =============================================================================
# Spectral Analysis
# =============================================================================

@dataclass
class SpectralGapResult:
    """Container for spectral gap analysis results."""
    eigenvalues: np.ndarray
    spectral_gap: float
    relaxation_time: float
    steady_state: Optional[np.ndarray]


def compute_spectral_gap(L: np.ndarray) -> SpectralGapResult:
    """
    Compute the spectral gap of a Liouvillian.
    
    The spectral gap Δ = -max{Re(λ) : λ ≠ 0} determines the convergence
    rate to the steady state.
    
    Parameters
    ----------
    L : np.ndarray
        Liouvillian superoperator
    
    Returns
    -------
    SpectralGapResult
        Eigenvalues, spectral gap, relaxation time, and steady state
    """
    eigs = eigvals(L)
    
    # Sort by real part (descending)
    idx = np.argsort(-np.real(eigs))
    eigs_sorted = eigs[idx]
    
    # Spectral gap: negative of the largest non-zero eigenvalue's real part
    # The largest eigenvalue should be 0 (trace preservation)
    nonzero_mask = np.abs(eigs_sorted) > 1e-10
    if np.any(nonzero_mask):
        largest_nonzero = eigs_sorted[nonzero_mask][0]
        gap = -np.real(largest_nonzero)
    else:
        gap = 0.0
    
    # Relaxation time τ = 1/Δ
    relaxation_time = 1.0 / gap if gap > 0 else np.inf
    
    # Compute steady state (null space of L)
    # For trace-preserving L, the steady state is the eigenvector with λ=0
    w, v = np.linalg.eig(L)
    zero_idx = np.argmin(np.abs(w))
    steady_vec = v[:, zero_idx]
    
    # Reshape and normalise
    d = int(np.sqrt(L.shape[0]))
    steady_state = unvectorise(steady_vec, dim=d)
    steady_state = steady_state / np.trace(steady_state)
    
    return SpectralGapResult(
        eigenvalues=eigs_sorted,
        spectral_gap=gap,
        relaxation_time=relaxation_time,
        steady_state=np.real_if_close(steady_state)
    )



# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("=== lindblad.py Module Test ===\n")
    
    # Build standard Pauli Liouvillian
    paulis = [SIGMA_X, SIGMA_Y, SIGMA_Z]
    
    print("Single-copy Liouvillian:")
    L1 = liouvillian_hamiltonian(paulis, sigma=1.0)
    print(f"  Shape: {L1.shape}")
    
    # Spectral analysis
    spec = compute_spectral_gap(L1)
    print(f"  Spectral gap: {spec.spectral_gap:.4f}")
    print(f"  Relaxation time: {spec.relaxation_time:.4f}")
    
    print("\nTwo-copy Liouvillian:")
    L2 = liouvillian_two_copy(paulis, sigma=1.0)
    print(f"  Shape: {L2.shape}")
    
    # M-matrix at t=0
    print("\nM-matrix at t=0:")
    M_0 = compute_M_matrix(0.0, L2)
    print(f"  M(0) diag = {np.diag(M_0.M).real}")
    print(f"  Condition number: {M_0.condition_number:.2e}")

    # M-matrix at large t (steady state)
    print("\nM-matrix as t → ∞:")
    M_inf = compute_M_matrix(1e6, L2)
    print(f"  M(∞) diag = {np.diag(M_inf.M).real}")
    print(f"  Invertible: {M_inf.is_invertible}")

    if M_inf.M_inv is not None:
        identity_check = M_inf.M @ M_inf.M_inv
        error = np.max(np.abs(identity_check - np.eye(4)))
        print(f"  Max error in M @ M⁻¹ = I: {error:.2e}")
