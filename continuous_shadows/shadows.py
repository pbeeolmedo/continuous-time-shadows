"""
shadows.py — Classical Shadow Estimation Module

This module provides the simulation engine for classical shadow tomography,
including snapshot generation and median-of-means estimation.

Key Classes:
- ShadowExperiment: Manages data generation (runs simulations, produces DataFrames)
- ShadowEstimator: Implements median-of-means estimation logic

Author: Pablo Bee Olmedo
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Callable, Tuple, List
from dataclasses import dataclass

from .core import (
    I2, P_0, P_1, KET_0, KET_1,
    density_matrix_to_bloch_vector,
    SIGMA_X, SIGMA_Y, SIGMA_Z
)
from .ensembles import (
    UnitaryEnsemble,
    PauliEnsemble,
    StochasticHamiltonianEnsemble,
    create_ensemble
)
from .analytical import (
    shadow_decay_rate,
    time_to_u,
    analytical_inverse_channel,
    inv_lambda_X,
    inv_lambda_Z,
    shadow_norm_composite,
)


# =============================================================================
# Shadow Experiment: Data Generation
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a shadow experiment."""
    num_runs: int
    ensemble: UnitaryEnsemble
    rng_seed: Optional[int] = None


class ShadowExperiment:
    """
    Shadow Experiment manager for classical shadow tomography.
    
    Generates measurement data for a given initial state and unitary ensemble.
    Supports both Pauli (discrete) and Stochastic (continuous-time) ensembles.
    
    The experiment produces a pandas DataFrame with columns:
        - run: run index
        - method: ensemble name
        - U_t: unitary applied
        - rho_t: rotated state U ρ₀ U†
        - rho_t_coords: Bloch vector of rotated state
        - (stochastic only) step, dT, cum_dT: time information
        - (after add_snapshots) rho_hat_snapshot: classical shadow estimator
    
    Parameters
    ----------
    rho_0 : np.ndarray
        Initial 2×2 density matrix
    num_runs : int
        Number of independent experimental runs
    ensemble : UnitaryEnsemble
        Ensemble object to sample unitaries from
    
    Examples
    --------
    >>> from continuous_shadows.core import P_0
    >>> from continuous_shadows.ensembles import PauliEnsemble
    >>> exp = ShadowExperiment(P_0, num_runs=1000, ensemble=PauliEnsemble())
    >>> df = exp.run()
    >>> df = exp.add_snapshots()
    """
    
    def __init__(
        self,
        rho_0: np.ndarray,
        num_runs: int,
        ensemble: UnitaryEnsemble,
        store_trajectory: bool = True,
    ):
        self.rho_0 = np.asarray(rho_0, dtype=complex)
        if self.rho_0.shape != (2, 2):
            raise ValueError(f"rho_0 must be 2×2, got {self.rho_0.shape}")

        self.num_runs = int(num_runs)
        self.ensemble = ensemble
        self.store_trajectory = store_trajectory
        self.df: Optional[pd.DataFrame] = None
    
    def run(self) -> pd.DataFrame:
        """
        Execute the experiment and generate the measurement DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Experiment data with columns depending on ensemble type
        """
        if isinstance(self.ensemble, PauliEnsemble):
            self.df = self._run_pauli()
        elif isinstance(self.ensemble, StochasticHamiltonianEnsemble):
            self.df = self._run_stochastic()
        else:
            # Generic ensemble: treat like Pauli (one sample per run)
            self.df = self._run_generic()
        
        return self.df
    
    def _run_pauli(self) -> pd.DataFrame:
        """Generate data for Pauli ensemble (no time axis)."""
        rows = []
        
        for run in range(self.num_runs):
            label, sample = self.ensemble.sample_unitary_with_label()
            U = sample.unitary
            rho_t = U @ self.rho_0 @ U.conj().T
            
            rows.append({
                "run": run,
                "method": "Pauli",
                "label": label,
                "U_t": U,
                "rho_t": rho_t,
                "rho_t_coords": density_matrix_to_bloch_vector(rho_t),
            })
        
        return pd.DataFrame(rows)
    
    def _run_stochastic(self) -> pd.DataFrame:
        """Generate data for stochastic ensemble.

        If store_trajectory is True (default), stores every intermediate
        time step.  If False, stores only the final unitary per run,
        which is ~n_steps times faster and uses far less memory.
        """
        rows = []
        dt = self.ensemble.dt

        if self.store_trajectory:
            for run in range(self.num_runs):
                times, unitaries = self.ensemble.sample_unitary_trajectory()
                rho_states = [U @ self.rho_0 @ U.conj().T for U in unitaries]

                for step, (t, U_t, rho_t) in enumerate(
                    zip(times, unitaries, rho_states)
                ):
                    rows.append({
                        "run": run,
                        "method": "Stochastic",
                        "sampling_type": self.ensemble.sampling_type,
                        "step": step,
                        "dT": dt,
                        "cum_dT": t,
                        "U_t": U_t,
                        "rho_t": rho_t,
                        "rho_t_coords": density_matrix_to_bloch_vector(rho_t),
                    })
        else:
            # Final-only mode: one row per run at t = T_max
            T_max = self.ensemble.T_max
            for run in range(self.num_runs):
                U = self.ensemble.sample_unitary()
                rho_t = U @ self.rho_0 @ U.conj().T

                rows.append({
                    "run": run,
                    "method": "Stochastic",
                    "sampling_type": self.ensemble.sampling_type,
                    "step": self.ensemble.n_steps,
                    "dT": dt,
                    "cum_dT": T_max,
                    "U_t": U,
                    "rho_t": rho_t,
                    "rho_t_coords": density_matrix_to_bloch_vector(rho_t),
                })

        return pd.DataFrame(rows)
    
    def _run_generic(self) -> pd.DataFrame:
        """Generate data for a generic ensemble."""
        rows = []
        
        for run in range(self.num_runs):
            U = self.ensemble.sample_unitary()
            rho_t = U @ self.rho_0 @ U.conj().T
            
            rows.append({
                "run": run,
                "method": self.ensemble.name,
                "U_t": U,
                "rho_t": rho_t,
                "rho_t_coords": density_matrix_to_bloch_vector(rho_t),
            })
        
        return pd.DataFrame(rows)
    
    def add_snapshots(
        self,
        inverse_channel: Optional[Callable] = None,
        rng_seed: int = 0,
        time_aware: bool = True,
    ) -> pd.DataFrame:
        """
        Add classical shadow snapshots to the DataFrame.
        
        For each row:
          1. Take U_t and compute rotated state ρ_rot = U_t ρ₀ U_t†
          2. Sample measurement outcome b ∈ {0, 1} with P(b=0) = ⟨0|ρ_rot|0⟩
          3. Form projector in lab frame: P_lab = U_t† |b⟩⟨b| U_t
          4. Apply inverse channel: ρ̂ = M⁻¹(P_lab)
        
        Parameters
        ----------
        inverse_channel : callable, optional
            Function P_lab -> ρ̂ (or (P_lab, t) -> ρ̂ for time-aware).
            If None, uses ensemble's default.
        rng_seed : int
            Random seed for reproducible measurement sampling
        time_aware : bool
            If True and using stochastic ensemble, each snapshot uses
            the inverse channel at its specific measurement time (cum_dT).
            This is the correct finite-time inversion. Default True.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with added 'rho_hat_snapshot' column
        """
        if self.df is None:
            raise RuntimeError("No data yet. Call .run() first.")
        
        if inverse_channel is None:
            inverse_channel = self.ensemble.inverse_channel

        rng = np.random.default_rng(rng_seed)
        
        # Vectorised snapshot computation
        snapshots = self._compute_snapshots_vectorised(
            inverse_channel, rng, time_aware=time_aware
        )
        
        self.df = self.df.copy()
        self.df["rho_hat_snapshot"] = snapshots
        
        return self.df
    
    def _compute_snapshots_vectorised(
        self,
        inverse_channel: Callable,
        rng: np.random.Generator,
        time_aware: bool = True,
    ) -> List[np.ndarray]:
        """
        Compute snapshots using vectorised operations where possible.
        
        For stochastic ensembles with time_aware=True, each snapshot
        uses the inverse channel at the correct measurement time.
        """
        # Extract arrays for vectorised computation
        n_rows = len(self.df)
        U_array = np.stack(self.df["U_t"].values)  # (n_rows, 2, 2)

        # Recompute rho_rot = U ρ_0 U† for all rows at once
        # Using einsum: 'nij,jk,nlk->nil' means U @ rho_0 @ U†
        rho_rot_array = np.einsum(
            'nij,jk,nlk->nil',
            U_array,
            self.rho_0,
            U_array.conj()
        )

        # Sample measurement outcomes
        p0_array = np.real(rho_rot_array[:, 0, 0])
        outcomes = (rng.random(n_rows) < p0_array).astype(int)  # 1 if |0⟩, 0 if |1⟩
        
        # Get times if available (for time-aware inversion)
        has_time = "cum_dT" in self.df.columns
        times = self.df["cum_dT"].values if has_time else None
        
        # Check if inverse_channel accepts a time argument
        # StochasticHamiltonianEnsemble.inverse_channel has (P_lab, t=None)
        use_time = time_aware and has_time and isinstance(
            self.ensemble, StochasticHamiltonianEnsemble
        )
        
        # Compute snapshots
        snapshots = []
        for i in range(n_rows):
            U = U_array[i]
            P_comp = P_0 if outcomes[i] == 1 else P_1
            P_lab = U.conj().T @ P_comp @ U
            
            if use_time and times[i] > 1e-12:
                rho_hat = inverse_channel(P_lab, t=times[i])
            else:
                rho_hat = inverse_channel(P_lab)
            snapshots.append(rho_hat)
        
        return snapshots

    def matrices_to_columns(self, inplace: bool = True) -> pd.DataFrame:
        """
        Flatten object-dtype matrix columns into plain scalar float columns.

        Delegates to the module-level :func:`flatten_matrix_columns`.  See that
        function for the full column mapping.

        Parameters
        ----------
        inplace : bool
            If True (default), update ``self.df`` in addition to returning the
            flattened DataFrame.

        Returns
        -------
        pd.DataFrame
            Flattened DataFrame with only scalar-dtype columns.

        Examples
        --------
        >>> exp_ket0.add_snapshots()
        >>> df_ket0 = exp_ket0.matrices_to_columns()
        >>> df_ket0.to_parquet("ket0.parquet")
        """
        if self.df is None:
            raise RuntimeError("No data yet. Call .run() first.")
        df = flatten_matrix_columns(self.df)
        if inplace:
            self.df = df
        return df

    @staticmethod
    def matrices_from_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Reconstruct object-dtype matrix columns from their scalar expansions.

        Inverse of :meth:`matrices_to_columns`.  Restores ``U_t``, ``rho_t``,
        ``rho_hat_snapshot`` (if present), and ``rho_t_coords`` as numpy-array
        columns, and drops the individual scalar columns.

        Parameters
        ----------
        df : pd.DataFrame
            Flattened DataFrame as produced by ``matrices_to_columns()``.

        Returns
        -------
        pd.DataFrame
            DataFrame with matrix object columns restored.
        """
        df = df.copy()
        _idx = [("00", 0, 0), ("01", 0, 1), ("10", 1, 0), ("11", 1, 1)]

        def _rebuild_matrix(df: pd.DataFrame, prefix: str) -> Optional[np.ndarray]:
            re_cols = [f"{prefix}_re_{t}" for t, *_ in _idx]
            im_cols = [f"{prefix}_im_{t}" for t, *_ in _idx]
            if not all(c in df.columns for c in re_cols):
                return None
            re = df[re_cols].values.reshape(-1, 2, 2)
            im = df[im_cols].values.reshape(-1, 2, 2)
            return re + 1j * im

        for col, prefix in [("U_t", "U_t"), ("rho_t", "rho_t"),
                             ("rho_hat_snapshot", "snap")]:
            arr = _rebuild_matrix(df, prefix)
            if arr is not None:
                df[col] = list(arr)
                drop = ([f"{prefix}_re_{t}" for t, *_ in _idx] +
                        [f"{prefix}_im_{t}" for t, *_ in _idx])
                df = df.drop(columns=drop)

        bloch_cols = ["bloch_x", "bloch_y", "bloch_z"]
        if all(c in df.columns for c in bloch_cols):
            coords = df[bloch_cols].values           # (N, 3)
            df["rho_t_coords"] = list(coords)
            df = df.drop(columns=bloch_cols)

        return df


# =============================================================================
# Standalone serialisation helpers
# =============================================================================

_MATRIX_COLS = {
    "U_t":              "U_t",
    "rho_t":            "rho_t",
    "rho_hat_snapshot": "snap",
}
_MAT_IDX = [("00", 0, 0), ("01", 0, 1), ("10", 1, 0), ("11", 1, 1)]


def flatten_matrix_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten object-dtype matrix / vector columns into scalar float64 columns.

    Works on any plain DataFrame, not just those attached to a
    :class:`ShadowExperiment`.  This makes it suitable for DataFrames built by
    concatenating many sub-experiments.

    Columns handled
    ---------------
    * ``U_t``              (2×2 complex) → 8 float cols, interleaved re/im:
                           ``U_t_re_00``, ``U_t_im_00``, ..., ``U_t_re_11``, ``U_t_im_11``
    * ``rho_t``            (2×2 complex) → same pattern, prefix ``rho_t_``
    * ``rho_hat_snapshot`` (2×2 complex, if present) → prefix ``snap_``
    * ``rho_t_coords``     (3,) float   → ``bloch_x``, ``bloch_y``, ``bloch_z``

    The original object columns are dropped.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with object-dtype matrix columns.

    Returns
    -------
    pd.DataFrame
        New DataFrame with only scalar-dtype columns (safe for Parquet / HDF5).

    Examples
    --------
    >>> from continuous_shadows.shadows import flatten_matrix_columns
    >>> df_var_flat = flatten_matrix_columns(df_var)
    >>> df_var_flat.to_parquet("var.parquet")
    """
    df = df.copy()

    for col, prefix in _MATRIX_COLS.items():
        if col not in df.columns:
            continue
        arr = np.stack(df[col].values)              # (N, 2, 2) complex
        for tag, i, j in _MAT_IDX:
            df[f"{prefix}_re_{tag}"] = arr[:, i, j].real
            df[f"{prefix}_im_{tag}"] = arr[:, i, j].imag
        df = df.drop(columns=[col])

    if "rho_t_coords" in df.columns:
        coords = np.stack(df["rho_t_coords"].values)    # (N, 3)
        df["bloch_x"] = coords[:, 0]
        df["bloch_y"] = coords[:, 1]
        df["bloch_z"] = coords[:, 2]
        df = df.drop(columns=["rho_t_coords"])

    return df


# =============================================================================
# Shadow Estimator: Median-of-Means
# =============================================================================

class ShadowEstimator:
    """
    Classical shadow observable estimator using median-of-means.
    
    Implements the median-of-means protocol for robust observable estimation
    with provable error bounds.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'run' and 'rho_hat_snapshot' columns
    rho_0 : np.ndarray
        True state (for computing errors)
    delta : float
        Failure probability (default 0.05)
    M : int
        Number of observables to estimate simultaneously (affects K)
    N_tot : int, optional
        Total number of samples (defaults to number of unique runs)
    
    Attributes
    ----------
    K : int
        Number of groups for median-of-means
    N : int
        Samples per group
    
    References
    ----------
    Huang, Kueng, Preskill. "Predicting many properties of a quantum system 
    from very few measurements" (2020).
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        rho_0: np.ndarray,
        delta: float = 0.05,
        M: int = 1,
        N_tot: Optional[int] = None
    ):
        self.df = df.copy()
        self.rho_0 = np.asarray(rho_0, dtype=complex)
        self.delta = float(delta)
        self.M = int(M)
        
        # Total number of samples
        if N_tot is None:
            N_tot = self.df["run"].nunique()
        self.N_tot = int(N_tot)
        
        # Compute K and N from the median-of-means formula
        # K = 2 log₂(2M/δ) ensures failure probability ≤ δ
        K = int(np.floor(2 * np.log2(2 * self.M / self.delta)))
        self.K = max(1, K)
        self.N = max(1, self.N_tot // self.K)
    
    def _compute_observable_estimates(
        self,
        df_subset: pd.DataFrame,
        observable: np.ndarray
    ) -> pd.Series:
        """
        Compute Tr(O ρ̂) for each snapshot in the DataFrame.
        
        Uses vectorised operations for efficiency.
        """
        # Vectorised trace computation
        rho_hats = np.stack(df_subset["rho_hat_snapshot"].values)
        # Tr(O @ ρ̂) = sum_ij O_ij * ρ̂_ji = einsum('ij,nji->n', O, rho_hats)
        estimates = np.real(np.einsum('ij,nji->n', observable, rho_hats))
        return pd.Series(estimates, index=df_subset.index)
    
    def _median_of_means(
        self,
        df_subset: pd.DataFrame,
        observable: np.ndarray,
        observable_str: str = "obs"
    ) -> Tuple[float, float, float]:
        """
        Compute median-of-means estimate for an observable.
        
        Returns
        -------
        Tuple[float, float, float]
            (mom_estimate, true_value, error)
        """
        observable = np.asarray(observable, dtype=complex)
        true_value = float(np.real(np.trace(observable @ self.rho_0)))
        
        df_local = df_subset.copy()
        df_local["group"] = (df_local["run"] // self.N).astype(int)
        
        # Compute observable estimates
        colname = f"observable_{observable_str}_estimate"
        df_local[colname] = self._compute_observable_estimates(df_local, observable)
        
        # Median of group means
        group_means = df_local.groupby("group")[colname].mean()
        mom_estimate = float(group_means.median())
        
        error = float(abs(mom_estimate - true_value))
        
        return mom_estimate, true_value, error
    
    def estimate(
        self,
        observable: np.ndarray,
        observable_str: str = "obs"
    ) -> Tuple[float, float, float]:
        """
        Compute median-of-means estimate using all data.
        
        Parameters
        ----------
        observable : np.ndarray
            2×2 observable matrix
        observable_str : str
            Label for the observable (for column naming)
        
        Returns
        -------
        Tuple[float, float, float]
            (mom_estimate, true_value, error)
        """
        return self._median_of_means(self.df, observable, observable_str)
    
    def estimate_time_series(
        self,
        observable: np.ndarray,
        observable_str: str = "obs",
        time_col: str = "cum_dT"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Compute median-of-means estimates as a function of time.
        
        Parameters
        ----------
        observable : np.ndarray
            2×2 observable matrix
        observable_str : str
            Label for the observable
        time_col : str
            Name of the time column (e.g., 'cum_dT', 'step')
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, float]
            (time_values, mom_estimates, errors, true_value)
        """
        if time_col not in self.df.columns:
            raise ValueError(f"Time column '{time_col}' not found in DataFrame")
        
        observable = np.asarray(observable, dtype=complex)
        true_value = float(np.real(np.trace(observable @ self.rho_0)))
        
        time_values = np.sort(self.df[time_col].unique())
        mom_estimates = []
        
        for t in time_values:
            df_t = self.df[self.df[time_col] == t]
            estimate, _, _ = self._median_of_means(df_t, observable, observable_str)
            mom_estimates.append(estimate)
        
        mom_estimates = np.array(mom_estimates)
        errors = np.abs(mom_estimates - true_value)
        
        return time_values, mom_estimates, errors, true_value

    @staticmethod
    def shadow_norm(
        pauli_decomposition: dict,
        u: float,
    ) -> float:
        """
        Compute the shadow norm for a composite observable.

        Uses the additive formula:
            ||O||²_shadow = Σ_α c_α² / λ_α(u)

        Parameters
        ----------
        pauli_decomposition : dict
            Mapping from Pauli label ("X", "Y", "Z") to coefficient
        u : float
            Dimensionless decay parameter

        Returns
        -------
        float
            Shadow norm squared
        """
        return shadow_norm_composite(pauli_decomposition, u)


# =============================================================================
# Convenience Functions
# =============================================================================

def run_shadow_experiment(
    rho_0: np.ndarray,
    num_runs: int,
    ensemble_type: str = "pauli",
    add_snapshots: bool = True,
    rng_seed: int = 0,
    **ensemble_kwargs
) -> pd.DataFrame:
    """
    Convenience function to run a complete shadow experiment.
    
    Parameters
    ----------
    rho_0 : np.ndarray
        Initial state (2×2)
    num_runs : int
        Number of experimental runs
    ensemble_type : str
        "pauli" or "stochastic"
    add_snapshots : bool
        Whether to add classical shadow snapshots
    rng_seed : int
        Random seed for snapshot generation
    **ensemble_kwargs
        Additional arguments for stochastic ensemble (T_max, n_steps, etc.)
    
    Returns
    -------
    pd.DataFrame
        Experiment data
    
    Examples
    --------
    >>> from continuous_shadows.core import P_0
    >>> df = run_shadow_experiment(P_0, num_runs=1000, ensemble_type="pauli")
    >>> df = run_shadow_experiment(
    ...     P_0, num_runs=100, ensemble_type="stochastic",
    ...     T_max=1.0, n_steps=50
    ... )
    """
    ensemble = create_ensemble(ensemble_type, **ensemble_kwargs)
    exp = ShadowExperiment(rho_0, num_runs, ensemble)
    df = exp.run()
    
    if add_snapshots:
        df = exp.add_snapshots(rng_seed=rng_seed)
    
    return df


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("=== shadows.py Module Test ===\n")
    
    # Test state
    rho_0 = P_0  # |0><0|
    
    # Test Pauli experiment
    print("Pauli Ensemble Experiment:")
    exp_pauli = ShadowExperiment(
        rho_0=rho_0,
        num_runs=500,
        ensemble=PauliEnsemble()
    )
    df_pauli = exp_pauli.run()
    df_pauli = exp_pauli.add_snapshots(rng_seed=42)
    print(f"  Shape: {df_pauli.shape}")
    print(f"  Columns: {df_pauli.columns.tolist()}")
    
    # Test estimator
    print("\nShadow Estimator (Pauli):")
    estimator = ShadowEstimator(df_pauli, rho_0)
    print(f"  K={estimator.K}, N={estimator.N}")
    
    mom_X, true_X, err_X = estimator.estimate(SIGMA_X, "X")
    mom_Z, true_Z, err_Z = estimator.estimate(SIGMA_Z, "Z")
    print(f"  ⟨X⟩: estimate={mom_X:.4f}, true={true_X:.4f}, error={err_X:.4f}")
    print(f"  ⟨Z⟩: estimate={mom_Z:.4f}, true={true_Z:.4f}, error={err_Z:.4f}")
    
    # Test Stochastic experiment
    print("\nStochastic Ensemble Experiment:")
    exp_stoch = ShadowExperiment(
        rho_0=rho_0,
        num_runs=50,
        ensemble=StochasticHamiltonianEnsemble(T_max=1.0, n_steps=20)
    )
    df_stoch = exp_stoch.run()
    df_stoch = exp_stoch.add_snapshots(rng_seed=42)
    print(f"  Shape: {df_stoch.shape}")
    print(f"  Columns: {df_stoch.columns.tolist()}")
    
    # Test time series estimation
    print("\nTime Series Estimation:")
    estimator_stoch = ShadowEstimator(df_stoch, rho_0)
    times, estimates, errors, true_val = estimator_stoch.estimate_time_series(SIGMA_Z)
    print(f"  Time points: {len(times)}")
    print(f"  Final estimate: {estimates[-1]:.4f}, true={true_val:.4f}")
    
    # Test convenience function
    print("\nConvenience Function:")
    df_quick = run_shadow_experiment(rho_0, num_runs=100, ensemble_type="pauli")
    print(f"  Shape: {df_quick.shape}")
