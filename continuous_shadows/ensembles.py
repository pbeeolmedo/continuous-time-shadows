"""
ensembles.py — Unitary Ensemble Module for Classical Shadows

This module implements the abstract base class for unitary ensembles and provides
concrete implementations for Pauli (discrete) and Stochastic Hamiltonian (continuous)
ensembles.

Author: Pablo Bee Olmedo
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Literal
from dataclasses import dataclass

from .core import I2, H2, S2, SIGMA_X, SIGMA_Y, SIGMA_Z
from .analytical import (
    shadow_decay_rate,
    time_to_u,
    analytical_inverse_channel,
    eigenvalues as analytical_eigenvalues,
)


# =============================================================================
# Abstract Base Class
# =============================================================================

class UnitaryEnsemble(ABC):
    """
    Abstract base class for unitary ensembles used in classical shadow tomography.
    
    Subclasses must implement the `sample_unitary()` method which returns
    a unitary matrix drawn from the ensemble's distribution.
    
    Properties
    ----------
    name : str
        Human-readable name of the ensemble
    dimension : int
        Hilbert space dimension (2 for single qubit)
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this ensemble."""
        pass
    
    @property
    def dimension(self) -> int:
        """Return the Hilbert space dimension."""
        return 2  # Single qubit by default
    
    @abstractmethod
    def sample_unitary(self) -> np.ndarray:
        """
        Draw a single unitary from the ensemble.
        
        Returns
        -------
        np.ndarray
            Unitary matrix with shape (dimension, dimension)
        """
        pass
    
    def sample_unitaries(self, n: int) -> List[np.ndarray]:
        """
        Draw n independent unitaries from the ensemble.
        
        Parameters
        ----------
        n : int
            Number of unitaries to sample
        
        Returns
        -------
        List[np.ndarray]
            List of n unitary matrices
        """
        return [self.sample_unitary() for _ in range(n)]
    
    @abstractmethod
    def inverse_channel(self, P_lab: np.ndarray) -> np.ndarray:
        """
        Apply the inverse measurement channel for shadow estimation.

        Subclasses may extend the signature with an optional time argument
        ``t`` (e.g. ``StochasticHamiltonianEnsemble.inverse_channel(P_lab, t=None)``),
        which callers in :mod:`shadows` detect via ``isinstance`` checks.

        Parameters
        ----------
        P_lab : np.ndarray
            Projector in the lab frame (U† P_comp U)

        Returns
        -------
        np.ndarray
            Classical shadow estimator ρ̂
        """
        pass


# =============================================================================
# Pauli Ensemble (Discrete)
# =============================================================================

@dataclass
class PauliSample:
    """Container for a Pauli ensemble sample."""
    unitary: np.ndarray
    basis_label: str  # "X", "Y", or "Z"


class PauliEnsemble(UnitaryEnsemble):
    """
    Standard Pauli (Clifford) ensemble for classical shadow tomography.
    
    Randomly selects one of the three Pauli bases {X, Y, Z} and returns
    the unitary that rotates that basis to the computational (Z) basis.
    
    The inverse channel for this ensemble is: ρ̂ = 3P - I
    
    References
    ----------
    Huang, Kueng, Preskill. "Predicting many properties of a quantum system 
    from very few measurements" (2020).
    """
    
    @property
    def name(self) -> str:
        return "Pauli"
    
    def sample_unitary(self) -> np.ndarray:
        """
        Draw a single-qubit unitary from the Pauli ensemble.
        
        Returns
        -------
        np.ndarray
            Unitary that maps the chosen Pauli eigenbasis to Z-basis
        """
        _, sample = self.sample_unitary_with_label()
        return sample.unitary
    
    def sample_unitary_with_label(self) -> Tuple[str, PauliSample]:
        """
        Draw a unitary and return it with its basis label.
        
        Returns
        -------
        Tuple[str, PauliSample]
            Basis label ("X", "Y", "Z") and PauliSample dataclass
        """
        basis_idx = np.random.randint(3)  # 0 -> Z, 1 -> X, 2 -> Y
        
        if basis_idx == 0:
            U = I2.copy()
            label = "Z"
        elif basis_idx == 1:
            U = H2.copy()  # Hadamard: X basis -> Z basis
            label = "X"
        else:
            # HS† maps Y eigenbasis to Z basis (H @ S†, i.e. apply S† first, then H)
            U = H2 @ S2.conj().T
            label = "Y"
        
        return label, PauliSample(unitary=U, basis_label=label)
    
    def inverse_channel(self, P_lab: np.ndarray) -> np.ndarray:
        """
        Apply the Pauli ensemble inverse channel.
        
        For the Pauli 3-design: ρ̂ = 3P_lab - I
        
        Parameters
        ----------
        P_lab : np.ndarray
            Projector in lab frame
        
        Returns
        -------
        np.ndarray
            Classical shadow estimator
        """
        return 3.0 * P_lab - I2


# =============================================================================
# Stochastic Hamiltonian Ensemble (Continuous)
# =============================================================================

SamplingType = Literal["Rademacher", "Normal", "ThreePoint"] # Literal means it can only be one of these strings, not any string.


@dataclass
class StochasticEnsembleParams:
    """Parameters for the stochastic Hamiltonian ensemble."""
    T_max: float = 1.0
    n_steps: int = 10
    kappa: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    sampling_type: SamplingType = "Rademacher"


class StochasticHamiltonianEnsemble(UnitaryEnsemble):
    """
    Stochastic Hamiltonian ensemble for continuous-time classical shadows.
    
    Generates unitaries via a discrete-time approximation to a stochastic
    Schrödinger equation:
    
        U ← exp(-i vₙ·σ) ... exp(-i v₂·σ) exp(-i v₁·σ)
    
    where each vₖ is a random rotation vector built from Wiener-like increments.
    
    Parameters
    ----------
    T_max : float
        Total evolution time
    n_steps : int
        Number of discretisation steps
    kappa : tuple of 3 floats
        Noise strengths (κₓ, κᵧ, κᵤ) along each Pauli direction
    sampling_type : str
        Method for sampling increments: "Rademacher", "Normal", or "ThreePoint"
    
    Attributes
    ----------
    params : StochasticEnsembleParams
        Encapsulated parameters
    dt : float
        Time step size (T_max / n_steps)
    
    References
    ----------
    [Add your thesis reference here]
    """
    
    def __init__(
        self,
        T_max: float = 1.0,
        n_steps: int = 10,
        kappa: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        sampling_type: SamplingType = "Rademacher"
    ):
        self.params = StochasticEnsembleParams(
            T_max=T_max,
            n_steps=n_steps,
            kappa=kappa,
            sampling_type=sampling_type
        )
        self._kappa_array = np.asarray(kappa, dtype=float)
        self.dt = T_max / n_steps
    
    @property
    def name(self) -> str:
        return f"Stochastic(T={self.params.T_max}, n={self.params.n_steps})"
    
    @property
    def T_max(self) -> float:
        return self.params.T_max
    
    @property
    def n_steps(self) -> int:
        return self.params.n_steps
    
    @property
    def kappa(self) -> Tuple[float, float, float]:
        return self.params.kappa
    
    @property
    def sampling_type(self) -> str:
        return self.params.sampling_type
    
    def _sample_delta_W(self) -> np.ndarray:
        """
        Sample a single Brownian increment vector ΔW = (ΔWₓ, ΔWᵧ, ΔWᵤ).
        
        Returns
        -------
        np.ndarray
            Increment vector with shape (3,)
        """
        dt = self.dt
        sampling = self.params.sampling_type
        
        if sampling == "Rademacher":
            # Binary: ±√dt with equal probability
            return np.array([
                np.random.choice([-np.sqrt(dt), np.sqrt(dt)]) 
                for _ in range(3)
            ], dtype=float)
        
        elif sampling == "Normal":
            # Gaussian: N(0, dt)
            return np.random.randn(3) * np.sqrt(dt)
        
        elif sampling == "ThreePoint":
            # Three-point distribution for higher-order convergence
            choices = [-np.sqrt(3 * dt), 0.0, np.sqrt(3 * dt)]
            probs = [1/6, 2/3, 1/6]
            return np.array([
                np.random.choice(choices, p=probs) 
                for _ in range(3)
            ], dtype=float)
        
        else:
            raise ValueError(f"Unknown sampling_type: {sampling}")
    
    def _exp_minus_i_v_sigma(self, v: np.ndarray) -> np.ndarray:
        """
        Compute exp(-i v·σ) using the SU(2) exponential formula.
        
        exp(-i v·σ) = cos|v| I - i sin|v| (v·σ / |v|)
        
        Parameters
        ----------
        v : np.ndarray
            Rotation vector (vₓ, vᵧ, vᵤ)
        
        Returns
        -------
        np.ndarray
            2×2 unitary matrix
        """
        v_norm = np.linalg.norm(v)
        
        if v_norm == 0:
            return I2.copy()
        
        v_dot_sigma = v[0] * SIGMA_X + v[1] * SIGMA_Y + v[2] * SIGMA_Z
        
        return (np.cos(v_norm)*I2 - 1j*np.sin(v_norm)*(v_dot_sigma/v_norm))
    
    def sample_unitary(self) -> np.ndarray:
        """
        Generate a single-qubit unitary from stochastic SU(2) evolution.
        
        Returns
        -------
        np.ndarray
            2×2 unitary matrix
        """
        U = I2.copy()
        
        for _ in range(self.params.n_steps):
            dW = self._sample_delta_W()
            v = np.sqrt(self._kappa_array) * dW
            exp_term = self._exp_minus_i_v_sigma(v)
            U = exp_term @ U
        
        return U
    
    def sample_unitary_trajectory(
        self
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Generate a full trajectory of unitaries at each time step.

        Returns
        -------
        Tuple[np.ndarray, List[np.ndarray]]
            times : array of time points [0, dt, 2dt, ..., T_max]
            unitaries : list of unitaries at each time point (including t=0)
        """
        U = I2.copy()

        times = [0.0]
        unitaries = [U.copy()]

        for step in range(self.params.n_steps):
            dW = self._sample_delta_W()
            v = np.sqrt(self._kappa_array) * dW
            exp_term = self._exp_minus_i_v_sigma(v)
            U = exp_term @ U

            times.append(times[-1] + self.dt)
            unitaries.append(U.copy())

        return np.array(times), unitaries

    def inverse_channel(self, P_lab: np.ndarray, t: float = None) -> np.ndarray:
        """
        Apply the analytical inverse channel for stochastic ensemble.
        
        Uses the closed-form eigenvalue formulas from analytical.py:
            M⁻¹(P_lab) at the specified time or at T_max.
        
        Parameters
        ----------
        P_lab : np.ndarray
            Projector in lab frame
        t : float, optional
            Measurement time. If None, uses T_max (final time).
        
        Returns
        -------
        np.ndarray
            Classical shadow estimator ρ̂
        """
        if t is None:
            t = self.params.T_max
        
        # Compute the dimensionless decay parameter at time t
        kappa_iso = np.mean(self._kappa_array)
        gamma_sh = shadow_decay_rate(sigma=1.0, kappa=kappa_iso)
        u = float(time_to_u(t, gamma_sh))
        
        return analytical_inverse_channel(P_lab, u)
    
    def get_eigenvalues(self, t: float = None):
        """
        Return the analytical eigenvalues (λ_X, λ_Y, λ_Z) at time t.
        
        Parameters
        ----------
        t : float, optional
            Time. If None, uses T_max.
        
        Returns
        -------
        Tuple[float, float, float]
            (λ_X, λ_Y, λ_Z)
        """
        if t is None:
            t = self.params.T_max
        
        kappa_iso = np.mean(self._kappa_array)
        gamma_sh = shadow_decay_rate(sigma=1.0, kappa=kappa_iso)
        u = float(time_to_u(t, gamma_sh))
        
        return analytical_eigenvalues(u)


# =============================================================================
# Factory Function
# =============================================================================

def create_ensemble(
    ensemble_type: str,
    **kwargs
) -> UnitaryEnsemble:
    """
    Factory function to create an ensemble by name.
    
    Parameters
    ----------
    ensemble_type : str
        "pauli" or "stochastic"
    **kwargs
        Additional parameters for stochastic ensemble
    
    Returns
    -------
    UnitaryEnsemble
        Configured ensemble instance
    
    Examples
    --------
    >>> ens = create_ensemble("pauli")
    >>> ens = create_ensemble("stochastic", T_max=2.0, n_steps=20)
    """
    ensemble_type = ensemble_type.lower()
    
    if ensemble_type == "pauli":
        return PauliEnsemble()
    
    elif ensemble_type == "stochastic":
        return StochasticHamiltonianEnsemble(**kwargs)
    
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("=== ensembles.py Module Test ===\n")
    
    # Test Pauli ensemble
    print("Pauli Ensemble:")
    pauli = PauliEnsemble()
    print(f"  Name: {pauli.name}")
    
    for i in range(5):
        label, sample = pauli.sample_unitary_with_label()
        print(f"  Sample {i+1}: basis={label}")
    
    # Test Stochastic ensemble
    print("\nStochastic Hamiltonian Ensemble:")
    stoch = StochasticHamiltonianEnsemble(
        T_max=1.0, 
        n_steps=10, 
        kappa=(1.0, 1.0, 1.0)
    )
    print(f"  Name: {stoch.name}")
    print(f"  dt = {stoch.dt}")
    
    U = stoch.sample_unitary()
    print(f"  Sample unitary shape: {U.shape}")
    err = np.max(np.abs(U @ U.conj().T - I2))
    print(f"  Unitarity error ‖U†U − I‖_max = {err:.2e}  (float64 machine eps ~ 2e-16)")
    
    # Test trajectory
    times, unitaries = stoch.sample_unitary_trajectory()
    print(f"  Trajectory length: {len(unitaries)} (including t=0)")
    print(f"  Time points: {times[:3]}...{times[-1]}")
    
    # Test factory
    print("\nFactory function:")
    ens1 = create_ensemble("pauli")
    ens2 = create_ensemble("stochastic", T_max=2.0)
    print(f"  Created: {ens1.name}, {ens2.name}")
