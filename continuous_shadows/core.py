"""
core.py — Foundation Module for Continuous-Time Classical Shadows

This module consolidates qubit system definitions, standard bases (Pauli matrices,
computational basis states), and centralised vectorisation logic.

Author: Pablo Bee Olmedo
"""

from __future__ import annotations

import numpy as np
from typing import Union, Optional

# qutip is only needed by the QubitSystem convenience class below. It is imported
# lazily inside that class so the core shadow pipeline (and every other module)
# works without qutip installed.

# =============================================================================
# Standard Bases & Constants
# =============================================================================

# Identity matrices
I2 = np.eye(2, dtype=complex)
I4 = np.eye(4, dtype=complex)

# Single-qubit gates
H2 = (1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=complex)  # Hadamard
S2 = np.array([[1, 0], [0, 1j]], dtype=complex)  # S gate (phase gate)

# Pauli matrices
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Pauli dictionary for easy lookup
PAULI_DICT = {
    "I": I2,
    "X": SIGMA_X,
    "Y": SIGMA_Y,
    "Z": SIGMA_Z,
}

# Computational basis kets
KET_0 = np.array([1.0, 0.0], dtype=complex)
KET_1 = np.array([0.0, 1.0], dtype=complex)

# Eigenstates of Pauli X
KET_PLUS = (1.0 / np.sqrt(2.0)) * np.array([1.0, 1.0], dtype=complex)
KET_MINUS = (1.0 / np.sqrt(2.0)) * np.array([1.0, -1.0], dtype=complex)

# Eigenstates of Pauli Y
KET_PLUS_I = (1.0 / np.sqrt(2.0)) * np.array([1.0, 1j], dtype=complex)
KET_MINUS_I = (1.0 / np.sqrt(2.0)) * np.array([1.0, -1j], dtype=complex)

# Projectors onto computational basis
P_0 = np.outer(KET_0, KET_0.conj())
P_1 = np.outer(KET_1, KET_1.conj())

# Projectors onto X eigenstates
P_PLUS = np.outer(KET_PLUS, KET_PLUS.conj())
P_MINUS = np.outer(KET_MINUS, KET_MINUS.conj())

# Projectors onto Y eigenstates
P_PLUS_I = np.outer(KET_PLUS_I, KET_PLUS_I.conj())
P_MINUS_I = np.outer(KET_MINUS_I, KET_MINUS_I.conj())


# =============================================================================
# Vectorisation Utilities (Column-major / Fortran order)
# =============================================================================

def vectorise(rho: Union[np.ndarray, "QubitSystem"]) -> np.ndarray:
    """
    Vectorise a density matrix using column-major (Fortran) ordering.
    
    This implements the vec(·) operation: |ρ⟩⟩ = vec(ρ)
    
    Parameters
    ----------
    rho : np.ndarray or QubitSystem
        Density matrix to vectorise. If QubitSystem, extracts the numpy array.
    
    Returns
    -------
    np.ndarray
        Vectorised density matrix with shape (d², )
    
    Notes
    -----
    Uses order='F' (Fortran/column-major) consistently throughout the codebase.
    This means columns are stacked vertically: vec(ρ) = [ρ[:,0], ρ[:,1], ...]
    """
    if isinstance(rho, np.ndarray):
        return rho.flatten(order='F')
    elif hasattr(rho, 'rho'):  # QubitSystem object
        return rho.rho.full().flatten(order='F')
    else:
        return np.asarray(rho).flatten(order='F')


def unvectorise(vec: np.ndarray, dim: int = 2) -> np.ndarray:
    """
    Unvectorise a vector back to a density matrix.
    
    This implements the inverse of vec(·): ρ = unvec(|ρ⟩⟩)
    
    Parameters
    ----------
    vec : np.ndarray
        Vectorised density matrix with shape (d², )
    dim : int
        Dimension of the Hilbert space (default 2 for single qubit)
    
    Returns
    -------
    np.ndarray
        Density matrix with shape (dim, dim)
    
    Notes
    -----
    Uses order='F' (Fortran/column-major) to be consistent with vectorise().
    """
    return vec.reshape((dim, dim), order='F')



# =============================================================================
# Bloch Vector Utilities
# =============================================================================

def density_matrix_to_bloch_vector(rho: np.ndarray) -> np.ndarray:
    """
    Convert a 2×2 density matrix to its corresponding Bloch vector.
    
    For a single-qubit state ρ = (I + r·σ)/2, returns r = (⟨X⟩, ⟨Y⟩, ⟨Z⟩).
    
    Parameters
    ----------
    rho : np.ndarray
        2×2 density matrix
    
    Returns
    -------
    np.ndarray
        Bloch vector (x, y, z) with shape (3,)
    
    Raises
    ------
    ValueError
        If input is not a 2×2 matrix
    """
    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (2, 2):
        raise ValueError(f"Input density matrix must be 2×2, got shape {rho.shape}")
    
    x = float(np.real(np.trace(rho @ SIGMA_X)))
    y = float(np.real(np.trace(rho @ SIGMA_Y)))
    z = float(np.real(np.trace(rho @ SIGMA_Z)))
    
    return np.array([x, y, z])


def bloch_vector_to_density_matrix(r: np.ndarray) -> np.ndarray:
    """
    Convert a Bloch vector to its corresponding 2×2 density matrix.
    
    Parameters
    ----------
    r : np.ndarray
        Bloch vector (x, y, z) with shape (3,)
    
    Returns
    -------
    np.ndarray
        2×2 density matrix: ρ = (I + r·σ)/2
    """
    r = np.asarray(r, dtype=float)
    if r.shape != (3,):
        raise ValueError(f"Bloch vector must have shape (3,), got {r.shape}")
    
    return 0.5 * (I2 + r[0] * SIGMA_X + r[1] * SIGMA_Y + r[2] * SIGMA_Z)


# =============================================================================
# QubitSystem Class
# =============================================================================

class QubitSystem:
    """
    Container for an n-qubit density matrix using QuTiP.
    
    Provides convenient methods for validation, manipulation, and conversion
    of quantum states.
    
    Examples
    --------
    >>> QubitSystem(3)                 # random mixed ρ on 3 qubits
    >>> QubitSystem(3, mixed=False)    # random pure state on 3 qubits
    >>> QubitSystem(3, rho=rho_1)      # use rho_1 (validated)
    >>> QubitSystem(1).to_bloch_vector()  # get Bloch vector for 1 qubit
    """

    def __init__(
        self,
        n: int,
        rho: Optional[Union[np.ndarray, qt.Qobj]] = None,
        mixed: bool = True,
        normalise: bool = False,
        atol: float = 1e-8
    ):
        """
        Initialise a qubit system.
        
        Parameters
        ----------
        n : int
            Number of qubits
        rho : np.ndarray or qt.Qobj, optional
            Density matrix (2^n × 2^n). If None, generates random state.
        mixed : bool
            If rho is None, whether to generate mixed (True) or pure (False) random state
        normalise : bool
            If True, normalise rho to have trace 1
        atol : float
            Numerical tolerance for trace/Hermiticity checks
        """
        import qutip as qt  # local import: QubitSystem is the only qutip user

        self.n = n
        self.dim = 2 ** n
        self.atol = atol

        if rho is None:
            # Generate random state
            if mixed:
                obj = qt.rand_dm(self.dim)
            else:
                psi = qt.rand_ket(self.dim)
                obj = psi * psi.dag()
        else:
            # Coerce to Qobj
            if isinstance(rho, qt.Qobj):
                obj = rho
            else:
                obj = qt.Qobj(rho)

        # Validate shape
        if obj.shape != (self.dim, self.dim):
            raise ValueError(
                f"rho must have shape {(self.dim, self.dim)} for n={n}, got {obj.shape}"
            )

        # Set subsystem structure: n qubits of dimension 2
        obj.dims = [[2] * n, [2] * n]

        # Validate Hermiticity
        if not qt.isherm(obj):
            raise ValueError("rho is not Hermitian")

        # Normalise if requested
        if normalise:
            obj = obj / obj.tr()

        # Validate trace
        if not np.allclose(obj.tr(), 1.0, atol=atol):
            raise ValueError(f"Tr(rho) ≈ {obj.tr()}, not 1")

        # Validate positivity
        eigvals = np.linalg.eigvalsh(obj.full())
        if np.min(eigvals) < -atol:
            raise ValueError(
                f"rho is not positive semidefinite, min eigenvalue = {np.min(eigvals)}"
            )

        self.rho = obj

    # ---------- Properties ----------

    def is_hermitian(self) -> bool:
        """Check if the density matrix is Hermitian."""
        import qutip as qt
        return qt.isherm(self.rho)

    def trace(self) -> complex:
        """Return the trace of the density matrix."""
        return self.rho.tr()

    def purity(self) -> float:
        """Return the purity Tr(ρ²)."""
        purity = (self.rho * self.rho).tr()
        return float(np.real_if_close(purity))

    def is_pure(self) -> bool:
        """Return True if ρ is a pure state (Tr(ρ²) ≈ 1)."""
        return bool(np.isclose(self.purity(), 1.0, atol=self.atol))

    # ---------- Conversion Methods ----------

    def to_numpy(self, vectorise: bool = False) -> np.ndarray:
        """
        Return ρ as a dense NumPy array.
        
        Parameters
        ----------
        vectorise : bool
            If True, return vectorised form (column-major)
        
        Returns
        -------
        np.ndarray
            Density matrix or vectorised form
        """
        if vectorise:
            return self.rho.full().flatten(order='F')
        return self.rho.full()

    def to_bloch_vector(self) -> np.ndarray:
        """
        Convert a 2×2 density matrix to its corresponding Bloch vector.
        
        Returns
        -------
        np.ndarray
            Bloch vector (x, y, z) with shape (3,)
        
        Raises
        ------
        ValueError
            If this is not a single-qubit system
        """
        if self.n != 1:
            raise ValueError(
                f"Bloch vector only defined for single qubit, got n={self.n}"
            )
        return density_matrix_to_bloch_vector(self.rho.full())

    # ---------- Subsystem Operations ----------

    def get_reduced_qubit(self, k: int) -> qt.Qobj:
        """
        Return the 2×2 reduced density matrix of qubit k.
        
        Parameters
        ----------
        k : int
            Qubit index (0, 1, ..., n-1)
        
        Returns
        -------
        qt.Qobj
            Reduced density matrix (2×2)
        
        Raises
        ------
        ValueError
            If k is out of range
        """
        if not (0 <= k < self.n):
            raise ValueError(f"k must be between 0 and {self.n - 1}, got {k}")
        return self.rho.ptrace(k)

    # ---------- Representation ----------

    def __repr__(self) -> str:
        purity_str = f"{self.purity():.4f}"
        return f"QubitSystem(n={self.n}, dim={self.dim}, purity={purity_str})"


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("=== core.py Module Test ===\n")
    
    # Test Pauli matrices
    print("Pauli matrices defined:")
    print(f"  SIGMA_X:\n{SIGMA_X}\n")
    
    # Test vectorisation
    rho = P_0  # |0><0|
    vec_rho = vectorise(rho)
    rho_back = unvectorise(vec_rho, dim=2)
    print(f"Vectorisation test (|0><0|):")
    print(f"  vec(ρ) = {vec_rho}")
    print(f"  unvec(vec(ρ)) == ρ: {np.allclose(rho, rho_back)}\n")
    
    # Test Bloch vector conversion
    bloch = density_matrix_to_bloch_vector(P_0)
    print(f"Bloch vector of |0><0|: {bloch}")
    
    rho_reconstructed = bloch_vector_to_density_matrix(bloch)
    print(f"Reconstructed ρ matches: {np.allclose(P_0, rho_reconstructed)}\n")
    
    # Test QubitSystem
    qs = QubitSystem(1, rho=P_0)
    print(f"QubitSystem: {qs}")
    print(f"  Bloch vector: {qs.to_bloch_vector()}")
    print(f"  Is pure: {qs.is_pure()}")
