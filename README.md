# continuous-shadows

A small Python library for simulating **continuous-time classical shadow
tomography** on a single qubit. It accompanies the MSc thesis *Continuous-Time
Classical Shadows* (University of Copenhagen, 2026).

**Version** 2.1.0. **Python** 3.10 or newer.

---

## The idea

Classical shadow tomography estimates many properties of a quantum state from a
small number of randomised measurements. In the standard Huang–Kueng–Preskill
(HKP) protocol the randomisation comes from discrete Clifford or Pauli gates.
Here the randomisation comes from a continuous stochastic Hamiltonian instead.

You evolve a qubit for a random time under a stochastic SU(2) drive, measure in
the computational basis, then invert the measurement channel to recover a shadow
estimator. The evolution time becomes a tunable knob. Stopping early, late, or
somewhere in between changes the variance of the estimate, and for many
observables an intermediate stopping time beats the HKP limit. That trade-off is
the Goldilocks effect at the centre of the thesis.

The whole channel is known in closed form, so most quantities are one-line
formulas with no numerical integration. A separate numerical superoperator
engine reproduces those formulas independently as a cross-check.

---

## Install

```bash
git clone https://github.com/pbeeolmedo/continuous-time-shadows
cd continuous-time-shadows

pip install -e .                    # core library: numpy, scipy, pandas, pyarrow
pip install -e ".[qubitsystem]"     # also qutip, used only by the QubitSystem helper
```

`qutip` is optional and backs only the convenience `QubitSystem` class in
`core.py`; everything else runs on the core stack.

---

## Quick start

Run a shadow experiment and estimate an observable.

```python
from continuous_shadows import P_0, SIGMA_Z
from continuous_shadows import StochasticHamiltonianEnsemble, ShadowExperiment, ShadowEstimator

# 1. continuous-time ensemble: evolve to T_max in 100 stochastic steps
ens = StochasticHamiltonianEnsemble(T_max=0.40, n_steps=100, kappa=(1, 1, 1))

# 2. run 10,000 shots on the |0> state
exp = ShadowExperiment(rho_0=P_0, num_runs=10_000, ensemble=ens)
df = exp.run()
df = exp.add_snapshots()          # time-aware inverse channel applied per shot

# 3. median-of-means estimate of <Z>
#    delta = failure probability, M = number of observables; K is derived as
#    K = floor(2 log2(2M / delta)). You do not set K directly.
est = ShadowEstimator(df, rho_0=P_0, delta=0.05, M=1)
estimate, true_val, error = est.estimate(SIGMA_Z)
print(f"<Z> estimate={estimate:.4f}  true={true_val:.4f}  error={error:.4f}")
```

To reproduce the discrete HKP baseline instead, swap in the Pauli ensemble:

```python
from continuous_shadows import create_ensemble
ens = create_ensemble("pauli")     # random Pauli-basis rotation (reproduces HKP)
```

---

## How the maths maps to code

Every formula in the thesis is implemented directly, each in one self-contained
function.

```python
from continuous_shadows import shadow_decay_rate, time_to_u, lambda_X, lambda_Z

gamma2 = shadow_decay_rate(sigma=1.0, kappa=1.0)   # gamma^(2) = 12 sigma^2 kappa = 12.0
u = time_to_u(t=0.3, gamma_sh=gamma2)              # u = exp(-gamma^(2) t)
lambda_Z(u)   # (1 + 2u)/3      polar eigenvalue
lambda_X(u)   # (1 - u)/3       equatorial eigenvalue (= lambda_Y)
```

**Shadow norm.** For a single Pauli the shadow norm is `1/lambda_alpha(u)`, and
for a sum of Paulis it is additive.

```python
from continuous_shadows import shadow_norm_pauli, shadow_norm_composite
shadow_norm_pauli("Z", u=0.0)                       # 1/lambda_Z = 3 at the Haar limit
shadow_norm_composite({"X": 1.0, "Z": 1.0}, u=0.0)  # sum c^2 / lambda = 6
```

**Goldilocks optimum.** For a tensor-product Pauli string, `k_eq` counts the
equatorial (X or Y) factors and `m_polar` counts the polar (Z) factors.

```python
from continuous_shadows import goldilocks_u_star, variance_multiplier, weighted_goldilocks_u_star

goldilocks_u_star(k_eq=1, m_polar=1)             # X(x)Z  ->  u* = 0.25
variance_multiplier(k_eq=1, m_polar=1, u=0.25)   # V(u*)  =  8.0

# weighted single-qubit observable O = X + Y + 3Z  (polarity ratio mu = 3)
weighted_goldilocks_u_star({"X": 1.0, "Y": 1.0, "Z": 3.0})   # u* = 0.4
```

**Numerical cross-check.** The 16x16 two-copy Liouvillian reproduces the
closed-form eigenvalues to about 1e-10.

```python
from continuous_shadows import validate_analytical_eigenvalues
res = validate_analytical_eigenvalues(sigma=1.0, kappa=1.0)
print(res["max_residual"])   # < 1e-8
```

> **Note on the thesis code listings.** The appendix listings use the in-folder
> flat form (`from core import P_0`, `from ensembles import …`). After installing
> this package the equivalent import is `from continuous_shadows import P_0`
> (or, fully qualified, `from continuous_shadows.core import P_0`). The function
> names and signatures are identical.

---

## Repository layout

```
continuous-time-shadows/
├── continuous_shadows/        # the importable package (six modules)
│   ├── core.py                # Pauli matrices, kets, projectors, vectorisation, QubitSystem
│   ├── analytical.py          # closed-form eigenvalues, inverse channel, shadow norm
│   ├── ensembles.py           # PauliEnsemble (HKP) and StochasticHamiltonianEnsemble
│   ├── shadows.py             # ShadowExperiment (DataFrame) and ShadowEstimator (MoM)
│   ├── lindblad.py            # superoperator algebra and numerical M-matrix (validation)
│   └── optimisation.py        # Goldilocks formula and variance-multiplier utilities
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

| Module | Role |
|---|---|
| `core` | Pauli matrices, the six cardinal kets and projectors, column-major vectorisation, and the optional `QubitSystem` class. |
| `analytical` | Closed-form eigenvalues, the inverse channel, and the shadow norm. The symbolic core of the library. |
| `ensembles` | `PauliEnsemble` reproduces the HKP protocol. `StochasticHamiltonianEnsemble` integrates the SU(2) SDE exactly per step. |
| `shadows` | `ShadowExperiment` produces a pandas DataFrame. `ShadowEstimator` runs median-of-means. |
| `lindblad` | Two-copy Liouvillian and numerical M-matrix. Used to validate `analytical`; not needed to run experiments. |
| `optimisation` | Goldilocks `u*`, the variance multiplier, and the weighted single-qubit optimum. |

---

## Design notes

- **Single qubit.** All analytical results are derived for one qubit. Multi-qubit
  observables factorise across qubits (thesis Section 5).
- **Isotropic drive.** The eigenvalue formulas assume equal scrambling strength on
  all three axes (`kappa_X = kappa_Y = kappa_Z`). This is what decouples the
  two-copy Liouvillian and makes the formulas exact.
- **Vectorisation.** Column-major (Fortran order), so superoperators act from the
  left: `vec(A rho B) = (B^T (x) A) vec(rho)`. Defined in `core.vectorise`.
- **Imports.** Internal modules use relative imports, so install the package
  (`pip install -e .`) or run from the repository root.
- **qutip is optional.** It is imported lazily inside `QubitSystem` only.

---

## License

Released under the MIT License. See the [LICENSE](LICENSE) file.

## Author

Pablo Bee Olmedo. MSc Quantum Information Science, University of Copenhagen, 2026.
Advisor: Daniel Stilck França.
