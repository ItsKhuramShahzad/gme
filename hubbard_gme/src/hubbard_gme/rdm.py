"""
rdm.py
------
Computation of n-body Reduced Density Matrices (n-RDMs) for arbitrary
site clusters of the Hubbard model.

Core formula
------------
For subsystem S = {i_1, ..., i_n} and wavefunction |ψ⟩:

    ρ[a, b] = ⟨ψ| c†_{p_1} ··· c†_{p_n}  c_{q_n} ··· c_{q_1} |ψ⟩

where
    p_k = 2·i_k + σ_a[k]   (creation  spin-orbital, from bra index a)
    q_k = 2·i_k + σ_b[k]   (annihilation spin-orbital, from ket index b)
    σ[k] = (index >> k) & 1   (0 = ↑, 1 = ↓)

The result is a Hermitian 2ⁿ × 2ⁿ matrix, normalised so Tr(ρ) = 1.
"""


import numpy as np
from itertools import combinations
from collections import defaultdict
from openfermion import FermionOperator
from openfermion.linalg import get_number_preserving_sparse_operator
import os
import sys
import pickle
import hashlib
import platform
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple, List




from .metrics import rdm_metrics


# ── Site-cluster generators ──────────────────────────────────────────────────

def generate_site_clusters(L: int, n: int) -> list[tuple[int, ...]]:
    """
    Return all contiguous n-site clusters on a ring of size L.

    Example: L=4, n=2 → [(0,1), (1,2), (2,3), (3,0)]
    """
    return [tuple((start + i) % L for i in range(n)) for start in range(L)]

# ── Thermal probabilities ───────────────────────────────────────────────────
def thermal_probabilities(energies: np.ndarray, T: float)-> np.ndarray:
    """
    Compute Boltzmann probabilities for a set of energies at temperature T.

    Parameters
    ----------
    energies : 1D array of energy levels.
    T        : Temperature (in same units as energies).

    Returns
    -------
    probs : 1D array of probabilities, summing to 1.
    """
    if T <= 0:
        raise ValueError("Temperature T must be positive.")

    beta = 1.0/T
    energies_shifted = energies- energies[0]
    weight = np.exp(-beta * energies_shifted)
    probs = weight / np.sum(weight)
   
    return probs
# ── RDM normalization ─────────────────────────────────────────────────────
def normalize_density_matrix(rho: np.ndarray, tol : float = 1e-14)-> np.ndarray: 
    """
    Normalize an RDM to have unit trace.

    Parameters
    ----------
    rho : 2D array (RDM)

    Returns
    -------
    rho_normalized : 2D array, normalized RDM.
    """
    tr = np.trace(rho).real
    if abs(tr)< tol:
        raise ValueError("Trace of RDm is to small to normalise.")
    return rho/tr

def compute_thermal_nRDM(orbital_indices: int, energies: np.ndarray, vecs: np.ndarray, T: float, n_spin_orbitals: int, n_electrons: int, spin_preserving: bool = True, prob_cutoff: float = 1e-14, degeneracy_tol: float = 1e-10)-> np.ndarray: 
    """
    Compute thermal extracted n-spin density matrix Γ^(n)(T).

    Γ^(n)(T) = Σ_m p_m(T) Γ^(n)[ψ_m]

    Important:
    - compute_general_nRDM returns an unnormalised Γ^(n).
    - We sum unnormalised Γ_m with Boltzmann weights.
    - Then we normalise only once at the end.
    """
    n = len(orbital_indices)
    Gamma_T = np.zeros((2**n, 2**n), dtype= complex)
    # T= 0 case: only ground state contributes
    if T< 1e-14:
        E0= energies [0 ]
        ground_idx= np.where(abs(energies-E0) < degeneracy_tol)[0]
        for idx in ground_idx: 
            Gamma_m = compute_general_nRDM(orbital_indices, vecs[:, idx], n_spin_orbitals, n_electrons, spin_preserving=spin_preserving)
            Gamma_T += Gamma_m/len(ground_idx)
        
        return normalize_density_matrix(Gamma_T)
    
    # T>0 case: sum over all states with Boltzmann weights
    probs = thermal_probabilities(energies, T)
    
    for idx, p_m in enumerate(probs): 
         if p_m < prob_cutoff:
             continue
         
         Gamma_m = compute_general_nRDM(orbital_indices, vecs[:, idx], n_spin_orbitals, n_electrons, spin_preserving=spin_preserving  )      
         Gamma_T += p_m*Gamma_m
    return normalize_density_matrix(Gamma_T)   
            
    
# ── Core RDM computation ─────────────────────────────────────────────────────


def compute_general_nRDM(
    orbital_indices: list[int],
    psi: np.ndarray,
    n_spin_orbitals: int,
    n_electrons: int,
    spin_preserving: bool = True,
) -> np.ndarray:
    """
    Compute the n-body RDM for a given set of site indices and wavefunction.

    Parameters
    ----------
    orbital_indices : List of n spatial-site indices.
    psi             : Wavefunction vector in the number-preserving basis.
    n_spin_orbitals : 2 * L.
    n_electrons     : Total electron count (for basis construction).
    spin_preserving : If True, assumes fixed N↑ and N↓.

    Returns
    -------
    np.ndarray  shape (2ⁿ, 2ⁿ), complex, NOT yet normalised.
    """
    n = len(orbital_indices)
    if n == 0:
        return np.array([[1.0 + 0j]])

    dim = 2 ** n
    rho_n = np.zeros((dim, dim), dtype=complex)

    for bra_idx in range(dim):
        bra_spins = [(bra_idx >> k) & 1 for k in range(n)]

        for ket_idx in range(dim):
            ket_spins = [(ket_idx >> k) & 1 for k in range(n)]

            # Build operator string: c†_{p_0} c†_{p_1} … c_{q_{n-1}} … c_{q_0}
            op_terms: list[tuple[int, int]] = []
            for body in range(n):
                site = orbital_indices[body]
                p = 2 * site + bra_spins[body]   # creation spin-orbital
                q = 2 * site + ket_spins[body]   # annihilation spin-orbital
                op_terms.append((p, 1))           # c†_p
                op_terms.append((q, 0))           # c_q

            op = FermionOperator(tuple(op_terms))
            op_sparse = get_number_preserving_sparse_operator(
                op, n_spin_orbitals, n_electrons, spin_preserving=spin_preserving
            )
            rho_n[bra_idx, ket_idx] = np.vdot(psi, op_sparse @ psi)

    return rho_n


# ── Pure/Excited RDM pair ────────────────────────────────────────────────────

def compute_pure_and_excited_RDMs(
    sites: list[int],
    psi0: np.ndarray,
    psi1: np.ndarray,
    n_spin_orbitals: int,
    n_electrons: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute and normalise the ground-state RDM (ρ₀) and first-excited-state
    RDM (ρ₁) for the given subsystem.

    Returns
    -------
    rho0, rho1  (both normalised to unit trace)
    """
    def _normalised_rdm(psi):
        # Compute directly here. PKL save/reuse is handled at the pipeline level.
        # Do NOT call load_or_compute_rdm_pkl with positional arguments: its
        # current API is keyword-only and requires L, U, t, mode, state_label, etc.
        rho = compute_general_nRDM(sites, psi, n_spin_orbitals, n_electrons)
        tr = np.trace(rho).real
        return rho / tr if abs(tr) > 1e-14 else rho

    return _normalised_rdm(psi0), _normalised_rdm(psi1)


# ── Full multi-body RDM sweep ────────────────────────────────────────────────

def compute_all_pure_rdms(
    L: int,
    psi0: np.ndarray,
    psi1: np.ndarray,
    n_spin_orbitals: int,
    n_electrons: int,
    max_n: int = 4,
    min_n: int = 1,
) -> tuple[dict, dict, dict]:
    """
    Compute n-body RDMs for n = min_n, …, max_n across all contiguous
    site clusters on the ring.

    Parameters
    ----------
    L               : Number of lattice sites.
    psi0, psi1      : Ground and first-excited wavefunctions.
    n_spin_orbitals : 2 * L.
    n_electrons     : Total electron count.
    max_n           : Maximum cluster size to compute.
    min_n           : Minimum cluster size to compute (default 1).
                      Set min_n = max_n to compute ONLY that body order.
                      Useful for n = N runs where you only want the full RDM.

    Returns
    -------
    rdm_gs : dict  {sites_tuple → ρ₀ (ground state)}
    rdm_ex   : dict  {sites_tuple → ρ₁ (excited state)}
    metrics  : dict  {sites_tuple → rdm_metrics(ρ₀)}
    """
    rdm_gs: dict = {}
    rdm_ex:   dict = {}
    metrics:  dict = {}

    for n in range(min_n, max_n + 1):
        clusters = generate_site_clusters(L, n)
        print(f"  Computing {n}-body RDMs ({len(clusters)} clusters) …", flush=True)

        for sites in clusters:
            rho0, rho1 = compute_pure_and_excited_RDMs(
                list(sites), psi0, psi1, n_spin_orbitals, n_electrons
            )
            rdm_gs[sites] = rho0
            rdm_ex[sites]   = rho1
            metrics[sites]  = rdm_metrics(rho0)

    return rdm_gs, rdm_ex, metrics


def compare_all_subsystem_rdms(
    rdm_dict,
    atol=1e-10,
    rtol=1e-10,
    show_equal=True,
    show_not_equal=True,
):
    equal_pairs = defaultdict(list)
    not_equal_pairs = defaultdict(list)

    keys = sorted(rdm_dict.keys(), key=lambda x: (len(x), x))

    total_compared = 0

    for k1, k2 in combinations(keys, 2):
        rho1 = rdm_dict[k1]
        rho2 = rdm_dict[k2]

        if rho1.shape != rho2.shape:
            continue

        size = len(k1)
        total_compared += 1

        max_diff = float(np.max(np.abs(rho1 - rho2)))
        same = np.allclose(rho1, rho2, atol=atol, rtol=rtol)

        row = (k1, k2, rho1.shape, max_diff)

        if same:
            equal_pairs[size].append(row)
        else:
            not_equal_pairs[size].append(row)

    total_equal = sum(len(v) for v in equal_pairs.values())
    total_not_equal = sum(len(v) for v in not_equal_pairs.values())

    print("\n" + "=" * 90)
    print("RDM COMPARISON SUMMARY")
    print("=" * 90)
    print(f"Total comparable pairs : {total_compared}")
    print(f"Equal pairs            : {total_equal}")
    print(f"Not equal pairs        : {total_not_equal}")
    print("=" * 90)

    def print_table(title, data, symbol):
        print("\n" + title)
        print("=" * 90)

        if not data:
            print("None")
            return

        for size in sorted(data.keys()):
            print(f"\nSubsystem size = {size}")
            print("-" * 90)
            print(f"{'Status':<8} {'RDM 1':<22} {'RDM 2':<22} {'Shape':<15} {'Max diff'}")
            print("-" * 90)

            for k1, k2, shape, max_diff in sorted(data[size], key=lambda x: (x[0], x[1])):
                print(
                    f"{symbol:<8} "
                    f"{str(k1):<22} "
                    f"{str(k2):<22} "
                    f"{str(shape):<15} "
                    f"{max_diff:.3e}"
                )

    if show_equal:
        print_table("EQUAL RDM PAIRS", equal_pairs, "EQUAL")

    if show_not_equal:
        print_table("NOT EQUAL RDM PAIRS", not_equal_pairs, "DIFF")

    return {
        "equal": dict(equal_pairs),
        "not_equal": dict(not_equal_pairs),
        "summary": {
            "total_compared": total_compared,
            "total_equal": total_equal,
            "total_not_equal": total_not_equal,
        },
    }
from collections import defaultdict

def print_rdm_results(results, show_equal=True, show_not_equal=True):
    """
    Print RDM comparison results from compare_all_subsystem_rdms.
    
    Accepts a dict with keys 'equal', 'not_equal', 'summary' where
    'equal' and 'not_equal' are dicts mapping subsystem_size -> list of rows.
    """
    equal = results.get("equal", {})
    not_equal = results.get("not_equal", {})
    summary = results.get("summary", {})

    total = summary.get("total_compared", 0)
    total_equal = summary.get("total_equal", 0)
    total_not_equal = summary.get("total_not_equal", 0)

    print("\n" + "=" * 90)
    print("RDM COMPARISON RESULTS")
    print("=" * 90)
    print(f"Total comparable pairs : {total}")
    print(f"Equal pairs            : {total_equal}")
    print(f"Not equal pairs        : {total_not_equal}")
    print("=" * 90)

    def print_section(title, data, status):
        print("\n" + title)
        print("=" * 90)
        if not data:
            print("None")
            return
        for size in sorted(data.keys()):
            rows = sorted(data[size], key=lambda x: (x[0], x[1]))
            print(f"\nSubsystem size = {size}")
            print("-" * 90)
            print(f"{'Status':<10} {'RDM 1':<22} {'RDM 2':<22} {'Shape':<12} {'Max diff'}")
            print("-" * 90)
            for k1, k2, shape, max_diff in rows:
                print(
                    f"{status:<10} "
                    f"{str(k1):<22} "
                    f"{str(k2):<22} "
                    f"{str(shape):<12} "
                    f"{max_diff:.3e}"
                )

    if show_equal:
        print_section("EQUAL SUBSYSTEM RDMs", equal, "EQUAL")
    if show_not_equal:
        print_section("NOT-EQUAL SUBSYSTEM RDMs", not_equal, "DIFF")

# results = compare_all_subsystem_rdms_not_equal(rdm_pure)
"""
RDM and system/state cache helpers for hubbard_gme.

Recommended location:
    src/hubbard_gme/rdm.py

These functions are designed to sit in the same module as your existing
compute_general_nRDM(...) function.

Central cache policy
--------------------
All RDM/state/convergence PKL files are stored in one shared project folder:

    <project_root>/results/rdm_pkl_results/

where <project_root> is automatically detected from this file location:

    <project_root>/src/hubbard_gme/rdm.py

So every module can reuse the same PKL files. No module should save its own
separate RDM cache inside tests/, notebooks/, or local outputs/.

Folder structure
----------------
    results/
    └── rdm_pkl_results/
        ├── state_cache/
        ├── rdm_cache/
        └── convergence_runs/

What this saves
---------------
1. System/state cache:
   - L, Ne, U, t, mode
   - n_so, n_electrons
   - E0, psi0
   - E1, psi1 if provided
   - optional Hamiltonian H
   - basis/operator/sigma metadata
   - hashes for state vectors and Hamiltonian

2. RDM cache:
   - rho
   - full metadata: L, Ne, U, t, mode, sites, n_so, state_label
   - parent state path/key/hash
   - metrics from metrics.py: trace, purity, entropy, eigvals
   - additional metrics: numerical rank, lambda_min, lambda_max
"""



try:
    # Use your existing metric functions.
    from .metrics import rdm_metrics
except ImportError:
    from hubbard_gme.metrics import rdm_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Central project-wide cache folders
# ─────────────────────────────────────────────────────────────────────────────

# rdm.py location:
#   <project_root>/src/hubbard_gme/rdm.py
#
# Path(__file__).resolve().parents[2] gives:
#   <project_root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RDM_PKL_RESULTS_DIR = PROJECT_ROOT / "results" / "rdm_pkl_results"
STATE_CACHE_DIR = RDM_PKL_RESULTS_DIR / "state_cache"
RDM_CACHE_DIR = RDM_PKL_RESULTS_DIR / "rdm_cache"
CONVERGENCE_PKL_DIR = RDM_PKL_RESULTS_DIR / "convergence_runs"

# Create the shared cache folders automatically.
RDM_PKL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
STATE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
RDM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CONVERGENCE_PKL_DIR.mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """Return the hubbard_gme project root."""
    return PROJECT_ROOT


def get_rdm_pkl_results_dir() -> Path:
    """Return central PKL result root: results/rdm_pkl_results."""
    return RDM_PKL_RESULTS_DIR


def get_state_cache_dir() -> Path:
    """Return central system/state cache directory."""
    return STATE_CACHE_DIR


def get_rdm_cache_dir() -> Path:
    """Return central RDM cache directory."""
    return RDM_CACHE_DIR


def get_convergence_pkl_dir() -> Path:
    """Return central convergence-run PKL directory."""
    return CONVERGENCE_PKL_DIR


def print_rdm_cache_dirs() -> None:
    """Print central cache paths for debugging."""
    print(f"PROJECT_ROOT          = {PROJECT_ROOT}")
    print(f"RDM_PKL_RESULTS_DIR   = {RDM_PKL_RESULTS_DIR}")
    print(f"STATE_CACHE_DIR       = {STATE_CACHE_DIR}")
    print(f"RDM_CACHE_DIR         = {RDM_CACHE_DIR}")
    print(f"CONVERGENCE_PKL_DIR   = {CONVERGENCE_PKL_DIR}")


def _resolve_state_cache_dir(cache_dir: Optional[str | os.PathLike] = None) -> Path:
    """
    If cache_dir is None, use central project state cache.
    If cache_dir is provided, use that explicit path.
    """
    if cache_dir is None:
        STATE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return STATE_CACHE_DIR

    path = Path(cache_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_rdm_cache_dir(cache_dir: Optional[str | os.PathLike] = None) -> Path:
    """
    If cache_dir is None, use central project RDM cache.
    If cache_dir is provided, use that explicit path.
    """
    if cache_dir is None:
        RDM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return RDM_CACHE_DIR

    path = Path(cache_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_convergence_pkl_dir(cache_dir: Optional[str | os.PathLike] = None) -> Path:
    """
    If cache_dir is None, use central project convergence PKL cache.
    If cache_dir is provided, use that explicit path.
    """
    if cache_dir is None:
        CONVERGENCE_PKL_DIR.mkdir(parents=True, exist_ok=True)
        return CONVERGENCE_PKL_DIR

    path = Path(cache_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# General helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float_tag(x: float) -> str:
    """
    Convert float to filename-safe text.

    Examples
    --------
    4.0  -> 4
    4.50 -> 4p5
    -1.5 -> m1p5
    """
    s = f"{float(x):.12g}"
    return s.replace("-", "m").replace(".", "p").replace("+", "")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _array_sha256(arr: Optional[np.ndarray]) -> Optional[str]:
    """
    Stable hash for NumPy arrays.
    Useful for checking whether psi0/psi1/H changed.
    """
    if arr is None:
        return None

    arr_c = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(arr_c.shape).encode("utf-8"))
    h.update(str(arr_c.dtype).encode("utf-8"))
    h.update(arr_c.view(np.uint8))
    return h.hexdigest()


def _basic_env_meta() -> Dict[str, Any]:
    # Included as meta["environment"] in EVERY pkl save path across the package
    # (thermal, u/t/v-scan, tT-surface, standalone single-point, molecular),
    # so recording the invoking command here stores it in each pkl at exactly
    # one place. "command" is the reproducible CLI string; "argv" keeps the raw
    # tokens; "cwd" pins where it was run from.
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
    }


def _ensure_tuple_int(x) -> Tuple[int, ...]:
    return tuple(int(v) for v in x)


# ─────────────────────────────────────────────────────────────────────────────
# Cache keys / filenames
# ─────────────────────────────────────────────────────────────────────────────

def make_system_cache_key(
    *,
    L: int,
    Ne: int,
    U: float,
    t: float,
    mode: str,
    cache_version: str = "v1",
) -> str:
    """
    Unique filename for the Hamiltonian/eigenstate cache.
    """
    return (
        f"system_{cache_version}_"
        f"L{L}_Ne{Ne}_U{_safe_float_tag(U)}_t{_safe_float_tag(t)}_"
        f"mode-{mode}.pkl"
    )


def make_rdm_cache_key(
    *,
    L: int,
    Ne: int,
    U: float,
    t: float,
    mode: str,
    n: int,
    sites: Tuple[int, ...],
    n_so: int,
    state_label: str = "gs",
    cache_version: str = "v1",
) -> str:
    """
    Unique filename for one RDM.

    Important:
    RDM depends on L, Ne, U, t, mode, sites, n_so, and state_label.
    """
    sites = _ensure_tuple_int(sites)
    sites_tag = "-".join(str(s) for s in sites)

    return (
        f"rdm_{cache_version}_"
        f"L{L}_Ne{Ne}_U{_safe_float_tag(U)}_t{_safe_float_tag(t)}_"
        f"mode-{mode}_state-{state_label}_"
        f"n{n}_sites-{sites_tag}_nso{n_so}.pkl"
    )


# ─────────────────────────────────────────────────────────────────────────────
# RDM metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_rdm_cache_metrics(rho: np.ndarray) -> Dict[str, Any]:
    """
    Compute RDM metrics using metrics.py.

    Returns
    -------
    dict with:
        trace, purity, entropy, eigvals,
        rank_numerical, lambda_min, lambda_max
    """
    info = rdm_metrics(rho)
    eigvals = np.asarray(info["eigvals"])

    info["rank_numerical"] = int(np.sum(eigvals > 1e-10))
    info["lambda_min"] = float(np.min(eigvals)) if eigvals.size else None
    info["lambda_max"] = float(np.max(eigvals)) if eigvals.size else None

    return info


# ─────────────────────────────────────────────────────────────────────────────
# System/state cache
# ─────────────────────────────────────────────────────────────────────────────

def save_system_state_pkl(
    *,
    L: int,
    Ne: int,
    U: float,
    t: float,
    mode: str,
    n_so: int,
    n_electrons: int,
    E0: float,
    psi0: np.ndarray,
    cache_dir: Optional[str | os.PathLike] = None,
    E1: Optional[float] = None,
    psi1: Optional[np.ndarray] = None,
    H: Optional[np.ndarray] = None,
    basis_meta: Optional[Dict[str, Any]] = None,
    operator_meta: Optional[Dict[str, Any]] = None,
    sigma_meta: Optional[Dict[str, Any]] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
    cache_version: str = "v1",
    overwrite: bool = True,
) -> str:
    """
    Save system-level data once in central state_cache by default.

    Returns
    -------
    cache_path : str
        Full path to saved system cache .pkl.
    """
    cache_dir = _resolve_state_cache_dir(cache_dir)

    filename = make_system_cache_key(
        L=L, Ne=Ne, U=U, t=t, mode=mode, cache_version=cache_version
    )
    cache_path = cache_dir / filename

    if cache_path.exists() and not overwrite:
        raise FileExistsError(f"System cache already exists: {cache_path}")

    states: Dict[str, Dict[str, Any]] = {
        "gs": {
            "label": "ground_state",
            "state_index": 0,
            "energy": float(E0),
            "psi": psi0,
            "psi_shape": tuple(psi0.shape),
            "psi_dtype": str(psi0.dtype),
            "psi_hash": _array_sha256(psi0),
        }
    }

    if E1 is not None and psi1 is not None:
        states["excited_1"] = {
            "label": "first_excited_state",
            "state_index": 1,
            "energy": float(E1),
            "psi": psi1,
            "psi_shape": tuple(psi1.shape),
            "psi_dtype": str(psi1.dtype),
            "psi_hash": _array_sha256(psi1),
        }

    meta: Dict[str, Any] = {
        "created_at": _now_iso(),
        "cache_version": cache_version,
        "object_type": "hubbard_system_state_cache",
        "cache_dir": str(cache_dir),
        "L": int(L),
        "Ne": int(Ne),
        "U": float(U),
        "t": float(t),
        "mode": str(mode),
        "n_so": int(n_so),
        "n_electrons": int(n_electrons),
        "states_available": list(states.keys()),
        "H_saved": H is not None,
        "H_shape": None if H is None else tuple(H.shape),
        "H_dtype": None if H is None else str(H.dtype),
        "H_hash": _array_sha256(H) if H is not None else None,
        "basis_meta": basis_meta or {},
        "operator_meta": operator_meta or {},
        "sigma_meta": sigma_meta or {},
        "environment": _basic_env_meta(),
    }

    if extra_meta:
        meta["extra_meta"] = dict(extra_meta)

    data = {
        "meta": meta,
        "states": states,
        "H": H,
    }

    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, cache_path)

    return str(cache_path)


def load_system_state_pkl(
    *,
    L: int,
    Ne: int,
    U: float,
    t: float,
    mode: str,
    cache_dir: Optional[str | os.PathLike] = None,
    cache_version: str = "v1",
    validate: bool = True,
) -> Dict[str, Any]:
    """
    Load system/state cache for a specific Hubbard setup from central state_cache by default.

    Returns
    -------
    data : dict with keys:
        meta, states, H, cache_path
    """
    cache_dir = _resolve_state_cache_dir(cache_dir)

    filename = make_system_cache_key(
        L=L, Ne=Ne, U=U, t=t, mode=mode, cache_version=cache_version
    )
    cache_path = cache_dir / filename

    if not cache_path.exists():
        raise FileNotFoundError(f"No system cache found at: {cache_path}")

    with open(cache_path, "rb") as f:
        data = pickle.load(f)

    if "meta" not in data or "states" not in data:
        raise ValueError(f"Invalid system cache file: {cache_path}")

    if validate:
        meta = data["meta"]
        expected = {
            "cache_version": cache_version,
            "L": int(L),
            "Ne": int(Ne),
            "U": float(U),
            "t": float(t),
            "mode": str(mode),
        }

        mismatches = []
        for key, expected_value in expected.items():
            cached_value = meta.get(key)
            if cached_value != expected_value:
                mismatches.append((key, cached_value, expected_value))

        if mismatches:
            msg = [f"System cache metadata mismatch for {cache_path}:"]
            for key, cached_value, expected_value in mismatches:
                msg.append(f"  {key}: cached={cached_value!r}, expected={expected_value!r}")
            raise ValueError("\n".join(msg))

    data["cache_path"] = str(cache_path)
    return data


def load_or_compute_system_state_pkl(
    *,
    compute_fn: Callable[..., Tuple[float, np.ndarray, float, np.ndarray, int]],
    compute_kwargs: Dict[str, Any],
    L: int,
    Ne: int,
    U: float,
    t: float,
    mode: str,
    cache_dir: Optional[str | os.PathLike] = None,
    cache_version: str = "v1",
    force_recompute: bool = False,
    H: Optional[np.ndarray] = None,
    basis_meta: Optional[Dict[str, Any]] = None,
    operator_meta: Optional[Dict[str, Any]] = None,
    sigma_meta: Optional[Dict[str, Any]] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    General wrapper to load or compute the system/eigenstate cache.

    compute_fn should return:
        E0, psi0, E1, psi1, n_so

    Returns
    -------
    result : dict
        {
            "E0": E0,
            "psi0": psi0,
            "E1": E1,
            "psi1": psi1,
            "n_so": n_so,
            "cache_path": path,
            "loaded_from_cache": bool,
            "raw": full_cache_dict,
        }
    """
    cache_dir = _resolve_state_cache_dir(cache_dir)

    if not force_recompute:
        try:
            data = load_system_state_pkl(
                cache_dir=cache_dir,
                L=L,
                Ne=Ne,
                U=U,
                t=t,
                mode=mode,
                cache_version=cache_version,
                validate=True,
            )

            gs = data["states"]["gs"]
            ex1 = data["states"].get("excited_1", None)

            return {
                "E0": gs["energy"],
                "psi0": gs["psi"],
                "E1": None if ex1 is None else ex1["energy"],
                "psi1": None if ex1 is None else ex1["psi"],
                "n_so": data["meta"]["n_so"],
                "cache_path": data["cache_path"],
                "loaded_from_cache": True,
                "raw": data,
            }
        except FileNotFoundError:
            pass

    E0, psi0, E1, psi1, n_so = compute_fn(**compute_kwargs)

    cache_path = save_system_state_pkl(
        cache_dir=cache_dir,
        L=L,
        Ne=Ne,
        U=U,
        t=t,
        mode=mode,
        n_so=n_so,
        n_electrons=Ne,
        E0=E0,
        psi0=psi0,
        E1=E1,
        psi1=psi1,
        H=H,
        basis_meta=basis_meta,
        operator_meta=operator_meta,
        sigma_meta=sigma_meta,
        extra_meta=extra_meta,
        cache_version=cache_version,
        overwrite=True,
    )

    data = load_system_state_pkl(
        cache_dir=cache_dir,
        L=L,
        Ne=Ne,
        U=U,
        t=t,
        mode=mode,
        cache_version=cache_version,
        validate=True,
    )

    return {
        "E0": E0,
        "psi0": psi0,
        "E1": E1,
        "psi1": psi1,
        "n_so": n_so,
        "cache_path": cache_path,
        "loaded_from_cache": False,
        "raw": data,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RDM cache
# ─────────────────────────────────────────────────────────────────────────────

def save_rdm_pkl(
    *,
    rho: np.ndarray,
    L: int,
    Ne: int,
    U: float,
    t: float,
    mode: str,
    sites: Tuple[int, ...],
    n_so: int,
    n_electrons: int,
    cache_dir: Optional[str | os.PathLike] = None,
    state_label: str = "gs",
    state_energy: Optional[float] = None,
    parent_state_cache_path: Optional[str] = None,
    parent_state_cache_key: Optional[str] = None,
    parent_state_hash: Optional[str] = None,
    basis_meta: Optional[Dict[str, Any]] = None,
    operator_meta: Optional[Dict[str, Any]] = None,
    sigma_meta: Optional[Dict[str, Any]] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
    cache_version: str = "v1",
    overwrite: bool = True,
) -> str:
    """
    Save one RDM with metadata, metrics, and reference to parent state cache.

    By default saves to:
        <project_root>/results/rdm_pkl_results/rdm_cache/

    Returns
    -------
    cache_path : str
    """
    cache_dir = _resolve_rdm_cache_dir(cache_dir)

    sites = _ensure_tuple_int(sites)
    n = len(sites)

    filename = make_rdm_cache_key(
        L=L,
        Ne=Ne,
        U=U,
        t=t,
        mode=mode,
        n=n,
        sites=sites,
        n_so=n_so,
        state_label=state_label,
        cache_version=cache_version,
    )

    cache_path = cache_dir / filename

    if cache_path.exists() and not overwrite:
        raise FileExistsError(f"RDM cache already exists: {cache_path}")

    metrics = compute_rdm_cache_metrics(rho)

    meta: Dict[str, Any] = {
        "created_at": _now_iso(),
        "cache_version": cache_version,
        "object_type": "reduced_density_matrix",
        "cache_dir": str(cache_dir),
        "L": int(L),
        "Ne": int(Ne),
        "U": float(U),
        "t": float(t),
        "mode": str(mode),
        "n": int(n),
        "sites": sites,
        "n_so": int(n_so),
        "n_electrons": int(n_electrons),
        "state_label": str(state_label),
        "state_energy": None if state_energy is None else float(state_energy),
        "rho_shape": tuple(rho.shape),
        "rho_dtype": str(rho.dtype),

        # Parent system/state reference
        "parent_state_cache_path": parent_state_cache_path,
        "parent_state_cache_key": parent_state_cache_key,
        "parent_state_hash": parent_state_hash,

        # RDM construction convention
        "rdm_function": "compute_general_nRDM",
        "normalization": "rho / Tr(rho).real",
        "basis_meta": basis_meta or {},
        "operator_meta": operator_meta or {},
        "sigma_meta": sigma_meta or {},
        "environment": _basic_env_meta(),

        # Duplicate most-used metrics in meta for quick inspection.
        "trace": float(metrics["trace"]),
        "purity": float(metrics["purity"]),
        "entropy": float(metrics["entropy"]),
        "rank_numerical": int(metrics["rank_numerical"]),
        "lambda_min": metrics["lambda_min"],
        "lambda_max": metrics["lambda_max"],
    }

    if extra_meta:
        meta["extra_meta"] = dict(extra_meta)

    data = {
        "meta": meta,
        "rho": rho,
        "metrics": metrics,
    }

    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, cache_path)

    return str(cache_path)


def load_rdm_pkl(
    *,
    L: int,
    Ne: int,
    U: float,
    t: float,
    mode: str,
    sites: Tuple[int, ...],
    n_so: int,
    cache_dir: Optional[str | os.PathLike] = None,
    state_label: str = "gs",
    cache_version: str = "v1",
    validate: bool = True,
    update_missing_metrics: bool = True,
) -> Dict[str, Any]:
    """
    Load a specific RDM pickle file.

    By default loads from:
        <project_root>/results/rdm_pkl_results/rdm_cache/

    Returns
    -------
    data : dict with:
        rho, meta, metrics, cache_path
    """
    cache_dir = _resolve_rdm_cache_dir(cache_dir)

    sites = _ensure_tuple_int(sites)
    n = len(sites)

    filename = make_rdm_cache_key(
        L=L,
        Ne=Ne,
        U=U,
        t=t,
        mode=mode,
        n=n,
        sites=sites,
        n_so=n_so,
        state_label=state_label,
        cache_version=cache_version,
    )

    cache_path = cache_dir / filename

    if not cache_path.exists():
        raise FileNotFoundError(f"No cached RDM found at: {cache_path}")

    with open(cache_path, "rb") as f:
        data = pickle.load(f)

    if "rho" not in data or "meta" not in data:
        raise ValueError(f"Invalid RDM cache file: {cache_path}")

    rho = data["rho"]
    meta = data["meta"]

    if validate:
        expected = {
            "cache_version": cache_version,
            "L": int(L),
            "Ne": int(Ne),
            "U": float(U),
            "t": float(t),
            "mode": str(mode),
            "n": int(n),
            "sites": sites,
            "n_so": int(n_so),
            "state_label": str(state_label),
        }

        mismatches = []
        for key, expected_value in expected.items():
            cached_value = meta.get(key)
            if cached_value != expected_value:
                mismatches.append((key, cached_value, expected_value))

        if mismatches:
            msg = [f"RDM cache metadata mismatch for {cache_path}:"]
            for key, cached_value, expected_value in mismatches:
                msg.append(f"  {key}: cached={cached_value!r}, expected={expected_value!r}")
            raise ValueError("\n".join(msg))

    metrics_were_missing = "metrics" not in data
    if metrics_were_missing:
        data["metrics"] = compute_rdm_cache_metrics(rho)

    # Repair old metadata if metrics were missing in old files.
    for key in ["trace", "purity", "entropy", "rank_numerical", "lambda_min", "lambda_max"]:
        if key not in meta:
            meta[key] = data["metrics"][key]

    data["cache_path"] = str(cache_path)

    if metrics_were_missing and update_missing_metrics:
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(
                {"meta": meta, "rho": rho, "metrics": data["metrics"]},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        os.replace(tmp_path, cache_path)

    return data


def load_or_compute_rdm_pkl(
    *,
    psi: np.ndarray,
    L: int,
    Ne: int,
    U: float,
    t: float,
    mode: str,
    sites: Tuple[int, ...],
    n_so: int,
    n_electrons: int,
    cache_dir: Optional[str | os.PathLike] = None,
    state_label: str = "gs",
    state_energy: Optional[float] = None,
    parent_state_cache_path: Optional[str] = None,
    parent_state_cache_key: Optional[str] = None,
    parent_state_hash: Optional[str] = None,
    basis_meta: Optional[Dict[str, Any]] = None,
    operator_meta: Optional[Dict[str, Any]] = None,
    sigma_meta: Optional[Dict[str, Any]] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
    cache_version: str = "v1",
    force_recompute: bool = False,
    normalize_trace: bool = True,
) -> Dict[str, Any]:
    """
    General wrapper:
    1. Try to load requested RDM from central cache.
    2. If missing or force_recompute=True, compute using compute_general_nRDM.
    3. Save rho + metadata + metrics.
    4. Return rho, metadata, metrics, cache path, and loaded_from_cache flag.

    Returns
    -------
    result : dict
        {
            "rho": rho,
            "meta": meta,
            "metrics": metrics,
            "cache_path": path,
            "loaded_from_cache": bool,
        }
    """
    cache_dir = _resolve_rdm_cache_dir(cache_dir)
    sites = _ensure_tuple_int(sites)

    if not force_recompute:
        try:
            data = load_rdm_pkl(
                cache_dir=cache_dir,
                L=L,
                Ne=Ne,
                U=U,
                t=t,
                mode=mode,
                sites=sites,
                n_so=n_so,
                state_label=state_label,
                cache_version=cache_version,
                validate=True,
                update_missing_metrics=True,
            )
            return {
                "rho": data["rho"],
                "meta": data["meta"],
                "metrics": data["metrics"],
                "cache_path": data["cache_path"],
                "loaded_from_cache": True,
            }
        except FileNotFoundError:
            pass

    # compute_general_nRDM must already exist in your rdm.py.
    rho = compute_general_nRDM(list(sites), psi, n_so, n_electrons)

    if normalize_trace:
        tr = np.trace(rho)
        if abs(tr) < 1e-14:
            raise ValueError(f"RDM trace is nearly zero for sites={sites}. Cannot normalize.")
        rho = rho / tr.real

    cache_path = save_rdm_pkl(
        rho=rho,
        cache_dir=cache_dir,
        L=L,
        Ne=Ne,
        U=U,
        t=t,
        mode=mode,
        sites=sites,
        n_so=n_so,
        n_electrons=n_electrons,
        state_label=state_label,
        state_energy=state_energy,
        parent_state_cache_path=parent_state_cache_path,
        parent_state_cache_key=parent_state_cache_key,
        parent_state_hash=parent_state_hash,
        basis_meta=basis_meta,
        operator_meta=operator_meta,
        sigma_meta=sigma_meta,
        extra_meta=extra_meta,
        cache_version=cache_version,
        overwrite=True,
    )

    data = load_rdm_pkl(
        cache_dir=cache_dir,
        L=L,
        Ne=Ne,
        U=U,
        t=t,
        mode=mode,
        sites=sites,
        n_so=n_so,
        state_label=state_label,
        cache_version=cache_version,
        validate=True,
        update_missing_metrics=False,
    )

    return {
        "rho": data["rho"],
        "meta": data["meta"],
        "metrics": data["metrics"],
        "cache_path": cache_path,
        "loaded_from_cache": False,
    }
