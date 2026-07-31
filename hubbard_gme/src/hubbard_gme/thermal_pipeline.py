"""
thermal_pipeline.py
-------------------
Temperature-dependent and parameter-dependent GME analysis
for the 1D Hubbard model.

Convex-roof solver
------------------
Mixed thermal nBRDMs are handled by convex_roof_gcg (GCG = Generalized
Conjugate Gradient on U(k), Rothlisberger–Lehmann–Loss 2009).
For pure states (T=0, n=L) the exact C_GME formula is used directly.
Pass solver='supervisor' for a fast analytical lower bound (useful for
quick tests; not a true convex-roof optimization).

Three public entry points
--------------------------
run_thermal_gme_analysis(L, t, U, T_values, ...)
    Compute C_GME(n, T) for n = n_min..L as a function of temperature.
    At T → 0, C_GCG(n=L) converges to the exact ground-state value from
    the PRB paper.  For n = 2 the same convex-roof procedure is used for
    mixed thermal states; the direct bipartite pure-state formula can be
    plotted only as an optional diagnostic.

find_first_excited_singlet(L, t, U, mode, n_electrons)
    Classify eigenstates by spin and return the index, energy, and
    wavefunction of the first S=0 excited state (above the triplet manifold).

run_excited_singlet_gme(L, t, U, ...)
    Compute n-body GME for the first singlet excited state (pure-state
    analysis; analogous to the ground-state analysis but for state idx > 0).

run_t_dependence_gme(L, t_values, U, ...)
    Compute C_GME(n) at T = 0 for each t value (fixing U).
    Plots C_GME vs t/U for n = n_min..L.
    Expected physics: no t-dependence for n = L; decreasing for n < L.
"""
from __future__ import annotations

import os
import pickle
import hashlib
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Optional, Sequence

from openfermion import FermionOperator
from openfermion.linalg import get_number_preserving_sparse_operator

from .hamiltonian import (
    build_hubbard_hamiltonian,
    compute_all_eigenstates,
    compute_k_eigenstates,
    classify_eigenstates_spin,
    print_spin_classification,
)
from .rdm import (
    generate_site_clusters,
    normalize_density_matrix,
    get_rdm_pkl_results_dir,
    _safe_float_tag,
    _basic_env_meta,
)
from .gme import supervisor_gme_lower_bound, compute_C_GME_pure
from .convex_roof import convex_roof_gcg
from .metrics import compute_purity, compute_entropy


# ── Color palette for n=2..6 ─────────────────────────────────────────────────

# 8 distinct colours/markers: a run covers body orders n = 2 … L, so with L = 8
# a 5-entry palette wrapped around and gave n=2/n=7 and n=3/n=8 identical
# colour+marker pairs (ambiguous legend). Keep >= 8 entries.
_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
           '#9467bd', '#8c564b', '#e377c2', '#17becf']
_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']


# ── Central PKL cache for thermal runs ───────────────────────────────────────

def _thermal_runs_dir() -> Path:
    """Return central thermal-run cache directory."""
    d = get_rdm_pkl_results_dir() / "thermal_runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _thermal_job_cache_dir() -> Path:
    """Return central per-job thermal cache directory."""
    d = _thermal_runs_dir() / "job_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _rdm_bank_dir() -> Path:
    """
    Return the central, task-INDEPENDENT RDM-bank store.

    The bank Γ^(n)[ψ_m] depends only on the Hamiltonian (L, t, U, mode, Ne,
    sector) — not on temperature, solver, or which analysis consumes it — so it
    lives in one generic folder shared by every task (thermal, surface,
    excited-singlet), keyed by those Hamiltonian parameters + sector.
    """
    d = get_rdm_pkl_results_dir() / "rdm_bank"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _array_digest(values: Optional[np.ndarray]) -> str:
    """Short stable hash for a 1-D parameter grid used in cache filenames."""
    if values is None:
        return "none"
    arr = np.ascontiguousarray(np.asarray(values, dtype=float))
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.view(np.uint8))
    return h.hexdigest()[:12]


def _array_sha256(arr: np.ndarray) -> str:
    """Stable SHA256 digest for a NumPy array."""
    arr_c = np.ascontiguousarray(np.asarray(arr))
    h = hashlib.sha256()
    h.update(str(arr_c.shape).encode("utf-8"))
    h.update(str(arr_c.dtype).encode("utf-8"))
    h.update(arr_c.view(np.uint8))
    return h.hexdigest()


def _optional_float_tag(value: Optional[float]) -> str:
    return "all" if value is None else _safe_float_tag(float(value))


def _iter_tag(max_iter: Optional[int]) -> str:
    return "default" if max_iter is None else str(int(max_iter))


def _run_timestamp() -> str:
    """Return YYYYMMDD_HHMMSS string for unique output filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _grid_tag(values: np.ndarray, name: str) -> str:
    """Compact filename-safe range tag, e.g. 'T0to5_N30' or 't0.05to2_N20'."""
    mn, mx, n = float(values[0]), float(values[-1]), len(values)
    return f"{name}{mn:g}to{mx:g}_N{n}"


def _sector_tag(spin_preserving: bool) -> str:
    """Filename tag for the Hilbert-space sector: 'sz0' or 'full'."""
    return "sz0" if spin_preserving else "full"


def _sector_label(spin_preserving: bool) -> str:
    """Human-readable Hilbert-space sector for plot captions."""
    return "Sz=0 sector" if spin_preserving else "full Hamiltonian (all Sz)"


def _thermal_cache_path(
    *,
    analysis_type: str,
    L: int,
    t: float,
    U: float,
    t_prime: Optional[float],
    t2: Optional[float],
    t_perp: Optional[float],
    mode: str,
    n_electrons: int,
    n_min: int,
    solver: str,
    n_restarts: int,
    max_iter: Optional[int],
    tol: Optional[float],
    k_max_extra: int,
    grid_name: str,
    grid_values: Optional[np.ndarray],
    cache_version: str,
    thermal_energy_cutoff: Optional[float] = None,
    average_site_clusters: bool = True,
    adaptive_cutoff: bool = False,
    cutoff_increment: float = 5.0,
    k: Optional[int] = None,
    spin_preserving: bool = True,
) -> Path:
    """Deterministic PKL path for one thermal-pipeline run."""
    k_tag = "none" if k is None else str(int(k))
    tol_tag = "default" if tol is None else _safe_float_tag(float(tol))
    if grid_values is not None:
        grid_tag = f"{_grid_tag(np.asarray(grid_values), grid_name)}-{_array_digest(grid_values)}"
    else:
        grid_tag = f"{grid_name}-none"
    filename = (
        f"{analysis_type}_{cache_version}_"
        f"L{L}_Ne{n_electrons}_U{_safe_float_tag(U)}_t{_safe_float_tag(t)}_"
        f"mode-{mode}_tp{_optional_float_tag(t_prime)}_"
        f"t2{_optional_float_tag(t2)}_tperp{_optional_float_tag(t_perp)}_"
        f"nmin{n_min}_solver-{solver}_{_sector_tag(spin_preserving)}_"
        f"restarts{int(n_restarts)}_iter{_iter_tag(max_iter)}_"
        f"kmax{int(k_max_extra)}_tol{tol_tag}_"
        f"cut{_optional_float_tag(thermal_energy_cutoff)}_"
        f"adaptive{int(bool(adaptive_cutoff))}_step{_safe_float_tag(cutoff_increment)}_"
        f"clusteravg{int(bool(average_site_clusters))}_"
        f"{grid_tag}_"
        f"k{k_tag}_diag0.pkl"
    )
    return _thermal_runs_dir() / filename


def _save_thermal_result_pkl(
    *,
    cache_path: Path,
    metadata: dict,
    result: dict,
) -> str:
    """Atomically save a complete thermal-pipeline result."""
    payload = {
        "meta": {
            **metadata,
            "object_type": "hubbard_thermal_pipeline_result",
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "environment": _basic_env_meta(),
        },
        "result": result,
    }
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, cache_path)
    return str(cache_path)


def _load_thermal_result_pkl(cache_path: Path) -> dict:
    """Load a complete thermal-pipeline result."""
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "result" not in payload:
        raise ValueError(f"Invalid thermal cache file: {cache_path}")
    return payload


def _thermal_gme_job_cache_path(
    *,
    run_key: str,
    i: int,
    T: float,
    n: int,
    gamma_hash: str,
) -> Path:
    """Deterministic path for one thermal GME job result."""
    filename = (
        f"tgme_{run_key}_i{int(i)}_T{_safe_float_tag(float(T))}_"
        f"n{int(n)}_rho{gamma_hash[:16]}.pkl"
    )
    return _thermal_job_cache_dir() / filename


def _save_thermal_gme_job_pkl(
    *,
    cache_path: Path,
    payload: dict,
) -> str:
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, cache_path)
    return str(cache_path)


def _load_thermal_gme_job_pkl(cache_path: Path) -> Optional[dict]:
    if not cache_path.exists():
        return None
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "result" not in payload:
        return None
    return payload


def _rdm_bank_cache_path(
    *,
    L: int,
    t: float,
    U: float,
    t_prime: Optional[float],
    t2: Optional[float],
    t_perp: Optional[float],
    mode: str,
    n_electrons: int,
    n_min: int,
    average_site_clusters: bool,
    cache_version: str,
    spin_preserving: bool = True,
    n_max: Optional[int] = None,
    V: float = 0.0,
) -> Path:
    """
    Deterministic path for the generic RDM bank (energies + averaged Γ^(n)).

    Keyed only by the Hamiltonian (L, Ne, U, t, V, mode, hoppings), n_min, the
    Sz sector, and the cluster-averaging convention — deliberately NOT by
    temperature, solver, or analysis type, so every task shares one file.
    A capped bank (n_max < L) gets an ``_nmax{N}`` tag so it never collides
    with (or is mistaken for) the full n_min..L bank; the uncapped filename is
    unchanged for backward compatibility. The nearest-neighbour V is tagged
    only when non-zero, so existing V=0 bank filenames stay valid.
    """
    nmax_tag = "" if (n_max is None or n_max >= L) else f"_nmax{int(n_max)}"
    v_tag = ("_V" + _safe_float_tag(float(V))) if abs(V) > 1e-12 else ""
    filename = (
        f"rdmbank_{cache_version}_"
        f"L{L}_Ne{n_electrons}_U{_safe_float_tag(U)}_t{_safe_float_tag(t)}_"
        f"mode-{mode}_tp{_optional_float_tag(t_prime)}_"
        f"t2{_optional_float_tag(t2)}_tperp{_optional_float_tag(t_perp)}{v_tag}_"
        f"nmin{n_min}{nmax_tag}_{_sector_tag(spin_preserving)}_"
        f"clusteravg{int(bool(average_site_clusters))}.pkl"
    )
    return _rdm_bank_dir() / filename


def _save_thermal_rdm_bank_pkl(
    *,
    cache_path: Path,
    payload: dict,
) -> str:
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, cache_path)
    return str(cache_path)


def _load_thermal_rdm_bank_pkl(cache_path: Path) -> Optional[dict]:
    if not cache_path.exists():
        return None
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        return None
    required = {"energies", "n_so", "bank", "n_values"}
    if not required.issubset(payload):
        return None
    return payload


# ── Precomputed operator tables ───────────────────────────────────────────────

def _precompute_ops(sites: tuple, n_so: int, n_electrons: int,
                    spin_preserving: bool = True) -> dict:
    """
    Build the (2^n)^2 sparse operator table for compute_general_nRDM
    for a fixed site cluster.  Operators depend only on sites, not on
    any wavefunction, so they can be reused across all eigenstates.

    Returns dict {(bra_idx, ket_idx): sparse_matrix}.
    """
    n = len(sites)
    dim = 2 ** n
    ops = {}
    for bra in range(dim):
        bra_spins = [(bra >> k) & 1 for k in range(n)]
        for ket in range(dim):
            ket_spins = [(ket >> k) & 1 for k in range(n)]
            op_terms = []
            for body in range(n):
                site = sites[body]
                p = 2 * site + bra_spins[body]
                q = 2 * site + ket_spins[body]
                op_terms += [(p, 1), (q, 0)]
            op = FermionOperator(tuple(op_terms))
            ops[(bra, ket)] = get_number_preserving_sparse_operator(
                op, n_so, n_electrons, spin_preserving=spin_preserving
            )
    return ops


def _rdm_from_ops(ops: dict, psi: np.ndarray, n: int) -> np.ndarray:
    """Evaluate ρ[bra,ket] = ⟨ψ|O_{bra,ket}|ψ⟩ using precomputed operators."""
    dim = 2 ** n
    rho = np.zeros((dim, dim), dtype=complex)
    for (bra, ket), op in ops.items():
        rho[bra, ket] = np.vdot(psi, op @ psi)
    return rho


_THERMAL_WORKER_EIGVECS: Optional[np.ndarray] = None
_THERMAL_WORKER_N_SO: Optional[int] = None
_THERMAL_WORKER_N_ELECTRONS: Optional[int] = None
_THERMAL_WORKER_SPIN_PRESERVING: bool = True


def _init_thermal_rdm_worker(
    eigvecs: np.ndarray,
    n_so: int,
    n_electrons: int,
    spin_preserving: bool = True,
) -> None:
    """Initialise multiprocessing worker state for RDM-bank jobs."""
    global _THERMAL_WORKER_EIGVECS
    global _THERMAL_WORKER_N_SO
    global _THERMAL_WORKER_N_ELECTRONS
    global _THERMAL_WORKER_SPIN_PRESERVING
    _THERMAL_WORKER_EIGVECS = eigvecs
    _THERMAL_WORKER_N_SO = int(n_so)
    _THERMAL_WORKER_N_ELECTRONS = int(n_electrons)
    _THERMAL_WORKER_SPIN_PRESERVING = bool(spin_preserving)


def _reset_thermal_rdm_worker() -> None:
    """Drop the parent's module-level reference to the shared eigenvectors."""
    global _THERMAL_WORKER_EIGVECS
    _THERMAL_WORKER_EIGVECS = None


def _compute_cluster_rdm_bank(
    n: int,
    sites: tuple,
    eigvecs: np.ndarray,
    n_so: int,
    n_electrons: int,
    spin_preserving: bool = True,
) -> tuple[int, tuple, np.ndarray, float]:
    """Compute Γ^(n)[ψ_m] for one site cluster and all eigenstates."""
    t0 = time.time()
    ops = _precompute_ops(sites, n_so, n_electrons, spin_preserving=spin_preserving)
    n_states = eigvecs.shape[1]
    dim = 2 ** n
    rdms = np.zeros((n_states, dim, dim), dtype=complex)
    for m in range(n_states):
        rdms[m] = _rdm_from_ops(ops, eigvecs[:, m], n)
    return n, sites, rdms, float(time.time() - t0)


def _thermal_rdm_cluster_worker(args):
    """Multiprocessing worker for one thermal RDM cluster."""
    n, sites = args
    if (
        _THERMAL_WORKER_EIGVECS is None
        or _THERMAL_WORKER_N_SO is None
        or _THERMAL_WORKER_N_ELECTRONS is None
    ):
        raise RuntimeError("Thermal RDM worker was not initialized.")
    return _compute_cluster_rdm_bank(
        int(n),
        tuple(sites),
        _THERMAL_WORKER_EIGVECS,
        _THERMAL_WORKER_N_SO,
        _THERMAL_WORKER_N_ELECTRONS,
        spin_preserving=_THERMAL_WORKER_SPIN_PRESERVING,
    )


def _build_rdm_bank(
    n_values: Sequence[int],
    eigvecs: np.ndarray,
    n_so: int,
    n_electrons: int,
    L: Optional[int] = None,
    average_site_clusters: bool = True,
    n_workers: int = 1,
    verbose: bool = True,
    spin_preserving: bool = True,
) -> dict[int, np.ndarray]:
    """
    Precompute Γ^(n)[ψ_m] for ALL eigenstates m and ALL body orders n.

    By default, averages over all contiguous site clusters for n < L.
    For L=6 this gives 6 independent cluster jobs for each n=2..5.
    The n=L case uses a single full-system ordering to avoid averaging
    density matrices written in different qubit orders.

    Returns
    -------
    bank : dict {n -> ndarray shape (n_states, 2^n, 2^n), dtype=complex}
    """
    if L is None:
        L = max(n_values)

    jobs: list[tuple[int, tuple]] = []
    clusters_by_n: dict[int, list[tuple]] = {}
    for n in n_values:
        if average_site_clusters and n < L:
            clusters = [tuple(s) for s in generate_site_clusters(L, n)]
        else:
            clusters = [tuple(range(n))]
        clusters_by_n[n] = clusters
        jobs.extend((n, sites) for sites in clusters)

    if verbose:
        print(
            f"  RDM cluster jobs: {len(jobs)} "
            f"(workers={max(1, int(n_workers or 1))}, "
            f"average_site_clusters={average_site_clusters})",
            flush=True,
        )
        for n in n_values:
            print(f"    n={n}: {len(clusters_by_n[n])} cluster(s)", flush=True)

    cluster_results: dict[int, list[tuple[tuple, np.ndarray]]] = {
        n: [] for n in n_values
    }

    n_workers = max(1, int(n_workers or 1))
    if n_workers > 1 and len(jobs) > 1:
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(it, **kw):  # noqa: E306
                return it

        import multiprocessing as mp
        # ── Shared eigenvectors via fork copy-on-write ────────────────────────
        # Instead of pickling `eigvecs` to every worker (initargs → one full
        # copy per process → N × size in RAM), publish it to a module global in
        # THIS (parent) process and fork the pool. On Linux the children inherit
        # the parent's pages copy-on-write; since the workers only READ the
        # eigenvectors, the large data buffer is never copied — all workers share
        # a single physical copy. RAM stops growing with --workers.
        _init_thermal_rdm_worker(eigvecs, n_so, n_electrons, spin_preserving)
        try:
            ctx = mp.get_context("fork")
        except ValueError:                      # non-fork platform (not Linux)
            ctx = mp.get_context()
        with ctx.Pool(processes=n_workers) as pool:
            iterator = pool.imap_unordered(_thermal_rdm_cluster_worker, jobs)
            for n, sites, rdms, elapsed in tqdm(
                iterator,
                total=len(jobs),
                desc="  Thermal RDM clusters",
                unit="cluster",
                ncols=90,
            ):
                cluster_results[n].append((sites, rdms))
                if verbose:
                    print(f"    done n={n}, sites={sites}, {elapsed:.2f}s", flush=True)
        _reset_thermal_rdm_worker()             # drop parent's extra reference
    else:
        for n, sites in jobs:
            if verbose:
                print(f"  Precomputing operators for n={n}, sites={sites} ...", flush=True)
            _, _, rdms, elapsed = _compute_cluster_rdm_bank(
                n, sites, eigvecs, n_so, n_electrons,
                spin_preserving=spin_preserving,
            )
            cluster_results[n].append((sites, rdms))
            if verbose:
                print(f"    done n={n}, sites={sites}, {elapsed:.2f}s", flush=True)

    bank: dict[int, np.ndarray] = {}
    for n in n_values:
        entries = cluster_results[n]
        if not entries:
            raise RuntimeError(f"No RDM clusters were computed for n={n}.")
        if len(entries) == 1:
            rdms = entries[0][1]
        else:
            rdms = np.mean(np.stack([entry[1] for entry in entries], axis=0), axis=0)
        bank[n] = rdms
        if verbose:
            print(
                f"    -> bank[{n}].shape = {rdms.shape} "
                f"(averaged {len(entries)} cluster(s))",
                flush=True,
            )
    return bank


def build_or_load_rdm_bank(
    *,
    L: int,
    t: float,
    U: float,
    V: float = 0.0,
    mode: str = "uniform",
    n_electrons: Optional[int] = None,
    n_min: int = 2,
    t_prime: Optional[float] = None,
    t2: Optional[float] = None,
    t_perp: Optional[float] = None,
    average_site_clusters: bool = True,
    n_workers: int = 1,
    spin_preserving: bool = True,
    use_pkl_cache: bool = False,
    cache_version: str = "v1",
    force_recompute: bool = False,
    save_pkl: bool = True,
    verbose: bool = True,
    n_max: Optional[int] = None,
) -> dict:
    """
    Build — or load from the generic cache — the RDM bank for one Hamiltonian.

    The bank Γ^(n)[ψ_m] is task-independent: thermal, surface, excited-singlet
    and T=0 scans all consume slices of the *same* object. It is therefore
    cached once per (L, t, U, mode, Ne, n_min, sector, cluster-averaging) in the
    generic store results/rdm_pkl_results/rdm_bank/ and reused by every task.
    On a cache hit the (expensive) full diagonalization is skipped entirely.

    Returns
    -------
    dict: energies, bank, n_so, n_values, cache_path, loaded_from_cache
    """
    if n_electrons is None:
        n_electrons = L
    # Cap cluster size at min(L, Ne): an n-site RDM with n > Ne has zero trace.
    n_hi = min(L, n_electrons) if n_max is None else min(int(n_max), L, n_electrons)
    n_values = list(range(n_min, n_hi + 1))

    cache_path = _rdm_bank_cache_path(
        L=L, t=t, U=U, V=V, t_prime=t_prime, t2=t2, t_perp=t_perp,
        mode=mode, n_electrons=n_electrons, n_min=n_min,
        average_site_clusters=average_site_clusters,
        cache_version=cache_version, spin_preserving=spin_preserving,
        n_max=n_max,
    )

    if use_pkl_cache and not force_recompute:
        payload = _load_thermal_rdm_bank_pkl(cache_path)
        if payload is not None:
            if verbose:
                print(f"    [cache] RDM bank → {cache_path}", flush=True)
            return {
                "energies": payload["energies"],
                "bank": payload["bank"],
                "n_so": int(payload["n_so"]),
                "n_values": n_values,
                "cache_path": cache_path,
                "loaded_from_cache": True,
            }

    energies, eigvecs, n_so = compute_all_eigenstates(
        L, t, U, mode, n_electrons, t_prime=t_prime, t2=t2, t_perp=t_perp,
        spin_preserving=spin_preserving, V=V,
        classify_spin=verbose, classify_k=16,   # thermal / tT-surface table
    )
    bank = _build_rdm_bank(
        n_values, eigvecs, n_so, n_electrons,
        L=L, average_site_clusters=average_site_clusters,
        n_workers=n_workers, verbose=verbose, spin_preserving=spin_preserving,
    )

    if use_pkl_cache and save_pkl:
        payload = {
            "object_type": "hubbard_rdm_bank",
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "cache_version": cache_version,
            "L": int(L), "t": float(t), "U": float(U), "V": float(V), "mode": str(mode),
            "t_prime": None if t_prime is None else float(t_prime),
            "t2": None if t2 is None else float(t2),
            "t_perp": None if t_perp is None else float(t_perp),
            "Ne": int(n_electrons), "n_min": int(n_min),
            "n_values": list(n_values), "n_so": int(n_so),
            "average_site_clusters": bool(average_site_clusters),
            "spin_preserving": bool(spin_preserving),
            "sector": _sector_label(spin_preserving),
            "energies": energies, "bank": bank,
            "environment": _basic_env_meta(),
        }
        _save_thermal_rdm_bank_pkl(cache_path=cache_path, payload=payload)
        if verbose:
            print(f"    RDM bank → {cache_path}", flush=True)

    return {
        "energies": energies, "bank": bank, "n_so": int(n_so),
        "n_values": n_values, "cache_path": cache_path,
        "loaded_from_cache": False,
    }


# ── GME bound dispatch ────────────────────────────────────────────────────────

def _gme_lower_bound(
    Gamma_T: np.ndarray,
    n: int,
    solver: str = "gcg",
    n_restarts: int = 20,
    max_iter: int = 300,
    tol: float = 1e-9,
    k_max_extra: int = 4,
) -> tuple[float, float, dict]:
    """
    Return (C_value, purity, detail) for a (possibly mixed) thermal nBRDM.

    `detail` is the full solver record (the same rich dict the standalone GME
    pipeline stores): for GCG it includes per-k `k_records` — each with the
    per-restart `restart_log`, the decomposition, timings and effective
    settings — so the thermal caches carry the complete convergence history.

    Dispatch:
    - purity ≈ 1 (pure state): exact C_GME via compute_C_GME_pure
    - solver='gcg': convex_roof_gcg — true convex-roof optimization via
      Generalized Conjugate Gradient on U(k)
    - solver='supervisor': supervisor_gme_lower_bound — fast analytical
      lower bound (not a convex-roof optimization; useful for quick tests)
    """
    pur = float(np.real(np.trace(Gamma_T @ Gamma_T)))
    if abs(pur - 1.0) < 1e-6:
        C = float(compute_C_GME_pure(Gamma_T, n))
        return C, pur, {"solver": "pure", "C_gcg": C, "pure": True,
                        "k_records": None}

    if solver == "supervisor":
        res = supervisor_gme_lower_bound(Gamma_T, n)
        detail = {"solver": "supervisor", "k_records": None}
        detail.update(res)
        return float(res["C_LB"]), pur, detail

    # GCG convex-roof optimization — returns the full k_records/restart_log dict.
    res = convex_roof_gcg(
        Gamma_T, n,
        n_restarts=n_restarts,
        max_iter=max_iter,
        tol=tol,
        k_max_extra=k_max_extra,
        verbose=False,
    )
    res.setdefault("solver", "gcg")
    return float(res["C_gcg"]), pur, res




def _gme_job_detail_payload(
    detail: Optional[dict],
    *,
    solver: str,
    n: int,
    purity: float,
) -> tuple[dict, dict]:
    """
    Convert the solver `detail` dict into the same (result_full, decompositions)
    structure the standalone GME pipeline stores, so thermal job caches carry the
    full solver record (bounds, effective settings, per-k / per-restart logs).

    Returns (result_full, decompositions). `decompositions = {solver: k_records}`
    when a convex-roof decomposition exists (GCG); empty for pure/supervisor.
    """
    if not isinstance(detail, dict):
        return {}, {}
    solver_name = detail.get("solver", solver)
    k_records = detail.get("k_records")
    result_full = {
        "n": int(n),
        "global_purity": float(purity),
        "C_gcg": detail.get("C_gcg"),
        "C_GME": detail.get("C_gcg"),
        "rank": detail.get("rank"),
        "k_converged": detail.get("k_converged"),
        "converged": detail.get("converged"),
        "completed_successfully": detail.get("completed_successfully"),
        "measure": detail.get("measure"),
        "n_restarts_requested": detail.get("n_restarts_requested"),
        "n_restarts_effective": detail.get("n_restarts_effective"),
        "k_max_extra": detail.get("k_max_extra"),
        "tol": detail.get("tol"),
        "tol_k": detail.get("tol_k"),
        "max_iter": detail.get("max_iter"),
        "max_iter_by_k": detail.get("max_iter_by_k"),
        "elapsed_by_k": detail.get("elapsed_by_k"),
        "elapsed_seconds": detail.get("elapsed_seconds"),
        "success_by_k": detail.get("success_by_k"),
        "optimizer_success_by_k": detail.get("optimizer_success_by_k"),
        "pure": bool(detail.get("pure", False)),
    }
    decompositions = {solver_name: k_records} if k_records else {}
    return result_full, decompositions


def _wootters_concurrence(rho: np.ndarray) -> float:
    """
    Wootters concurrence for a 2-qubit (4x4) density matrix.

    C = max(0, lambda_1 - lambda_2 - lambda_3 - lambda_4)
    where lambda_i are the square roots of eigenvalues of
    R = rho * (sigma_y x sigma_y) * rho* * (sigma_y x sigma_y)
    in decreasing order.
    """
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sysy = np.kron(sy, sy)
    rho_tilde = sysy @ rho.conj() @ sysy
    R = rho @ rho_tilde
    # Eigenvalues of R are real and non-negative in exact arithmetic
    eigs = np.real(np.linalg.eigvals(R))
    eigs = np.sqrt(np.maximum(eigs, 0.0))
    eigs = np.sort(eigs)[::-1]
    return float(max(0.0, eigs[0] - eigs[1] - eigs[2] - eigs[3]))


def _cgme_analytical_n2(rho: np.ndarray) -> float:
    """
    Analytical C_GME for a 2-qubit Werner state RDM using the supervisor's formula:

        Step 1: extract Werner state parameter p from largest eigenvalue
                lambda_max = (1 + 3p) / 4  =>  p = (4*lambda_max - 1) / 3
        Step 2: compute concurrence C = max(0, (3p - 1) / 2)
        Step 3: C_GME = sqrt(1 - sqrt(1 - C^2))
    """
    eigs = np.sort(np.real(np.linalg.eigvalsh(rho)))[::-1]
    lambda_max = float(eigs[0])
    p = (4.0 * lambda_max - 1.0) / 3.0
    C = max(0.0, (3.0 * p - 1.0) / 2.0)
    val = 1.0 - np.sqrt(max(0.0, 1.0 - C ** 2))
    return float(np.sqrt(max(0.0, val)))


def _thermal_gme_worker(args):
    """Worker for one independent thermal GME evaluation at fixed (T index, n)."""
    (
        i,
        T,
        n,
        Gamma_T,
        solver,
        n_restarts,
        max_iter,
        tol,
        k_max_extra,
        seed,
        job_cache_path,
        gamma_hash,
    ) = args
    np.random.seed(int(seed))
    t0 = time.time()
    c_val, pur, detail = _gme_lower_bound(
        Gamma_T,
        n,
        solver=solver,
        n_restarts=n_restarts,
        max_iter=max_iter,
        tol=tol,
        k_max_extra=k_max_extra,
    )
    return (
        int(i),
        float(T),
        int(n),
        c_val,
        pur,
        float(time.time() - t0),
        str(job_cache_path),
        str(gamma_hash),
        detail,
    )

def _thermal_active_probabilities(
    energies: np.ndarray,
    T: float,
    reference_energy: float,
    thermal_energy_cutoff: Optional[float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return active eigenstate indices and renormalized Boltzmann probabilities.

    The cutoff implements the convergence criterion
        (E_k - E_ref) / (k_B T) < thermal_energy_cutoff
    with k_B = 1 in the units used by this code.  Set the cutoff to None to
    include all states.
    """
    if T <= 1e-14:
        raise ValueError("Temperature T must be positive for thermal weights.")
    if thermal_energy_cutoff is not None and thermal_energy_cutoff <= 0:
        raise ValueError("thermal_energy_cutoff must be positive or None.")

    if thermal_energy_cutoff is None:
        active = np.arange(len(energies))
    else:
        ratios = (energies - reference_energy) / T
        active = np.where(ratios < thermal_energy_cutoff)[0]

    if len(active) == 0:
        active = np.array([int(np.argmin(np.abs(energies - reference_energy)))])

    shifted = energies[active] - reference_energy
    exponents = -shifted / T
    exponents -= np.max(exponents)
    weights = np.exp(exponents)
    probs = weights / np.sum(weights)
    return active, probs


def _converged_thermal_cutoff(
    energies: np.ndarray,
    T: float,
    E_ref: float,
    bank: dict,
    n_values: list,
    initial_cutoff: float = 10.0,
    increment: float = 5.0,
    max_cutoff: float = 60.0,
    rdm_tol: float = 1e-6,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Adaptively find the thermal energy cutoff at which Γ^(n)(T) is converged.

    Supervisor's criterion: include eigenstates ψ_k with
        (E_k - E_0) / (k_B T) < cutoff
    Start at initial_cutoff=10, increase by `increment` until the maximum
    Frobenius-norm change in any thermal nBRDM across the step is below
    `rdm_tol`, or no new states enter the window.

    Returns
    -------
    (converged_cutoff, active_indices, active_probs)
    """
    cutoff = initial_cutoff
    active, probs = _thermal_active_probabilities(energies, T, E_ref, cutoff)
    rdms_prev = {
        n: np.einsum("m,mij->ij", probs, bank[n][active]) for n in n_values
    }

    while cutoff < max_cutoff:
        next_cutoff = min(cutoff + increment, max_cutoff)
        active_next, probs_next = _thermal_active_probabilities(
            energies, T, E_ref, next_cutoff
        )

        if len(active_next) == len(active):
            break  # no new states entered — converged

        rdms_next = {
            n: np.einsum("m,mij->ij", probs_next, bank[n][active_next])
            for n in n_values
        }
        max_diff = max(
            np.linalg.norm(rdms_next[n] - rdms_prev[n], "fro")
            for n in n_values
        )

        cutoff = next_cutoff
        active = active_next
        probs = probs_next
        rdms_prev = rdms_next

        if max_diff < rdm_tol:
            break  # Frobenius change negligible — converged

    return float(cutoff), active, probs


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _make_figure_T(
    T_values: np.ndarray,
    C_GME: dict,
    purity_dict: dict,
    n_values: list,
    system_info: dict,
    solver: str = "gcg",
    spin_preserving: bool = True,
    T_view_max: Optional[float] = None, # clip the x-axis to this value if provided
) -> plt.Figure:
    """
    Two-panel figure: C_GME vs T (top), purity vs T (bottom).
    X-axis is clipped to T_display_max = max T where any C_GME > 0.01,
    so flat-zero tails don't dominate the plot.
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(9, 6))

    L, t, U = system_info["L"], system_info["t"], system_info["U"]
    
    Ne = int(system_info.get("Ne", L) or L)
    n_values = [n for n in n_values if n <= Ne]      # never draw n > Ne
    
    gap = system_info.get("gap", None)
    C_exact = system_info.get("C_exact_gs", None)
    mode = system_info.get("mode", "uniform")
    label_suffix = system_info.get("label_suffix", "")
    ylabel = r"$C_{GME}$" if solver == "gcg" else r"$C_{GME}$ lower bound"

    # Compute display x-limit: last T where any n has C_GME > 1% of max.
    T_arr = np.asarray(T_values)
    all_c = np.array([np.asarray(C_GME[n]) for n in n_values])
    c_max = float(all_c.max()) if all_c.size > 0 else 1.0
    threshold = max(0.01 * c_max, 1e-4)
    active_mask = np.any(all_c > threshold, axis=0)
    if active_mask.any():
        last_active = int(np.where(active_mask)[0][-1])
        T_display_max = T_arr[min(last_active + max(3, len(T_arr) // 10), len(T_arr) - 1)]
    else:
        T_display_max = float(T_arr[-1])

    # Per-n colour + marker match the t/U/V scan plots (plot_scan_finalized)
    # so a curve for a given n looks identical across every figure type.
    for i, n in enumerate(n_values):
        lbl = f"n={n}"
        if n == 2:
            lbl += " (bipartite)"
        elif n == L:
            lbl += " (whole ring)"
        ax1.plot(T_arr, C_GME[n],
                 color=_COLORS[i % len(_COLORS)],
                 marker=_MARKERS[i % len(_MARKERS)], ms=5,
                 label=lbl, lw=1.6)

    # Drop the pure-formula reference if the T=0 largest-cluster RDM is MIXED
    # (lowest-T stored purity < 1). Corrects old thermal PKLs at plot time too.
    _p0 = purity_dict.get(n_values[-1]) if isinstance(purity_dict, dict) else None
    _gs_pure = (_p0 is None) or (np.asarray(_p0, float).ravel()[0] > 1.0 - 1e-6)
    if C_exact is not None and np.isfinite(C_exact) and _gs_pure:
        ax1.axhline(
            C_exact, color="k", ls=":", lw=1.2,
            label=f"Exact C_GME(n={n_values[-1]}, T=0) = {C_exact:.4f}",
        )

    if gap is not None and gap > 0:
        ax1.axvline(gap, color="gray", ls=":", lw=1.0, alpha=0.6,
                    label=f"Gap Δ = {gap:.4f}")

    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.set_title(
        f"Thermal GME: L={L}, Ne={Ne}, t={t}, U={U} (t/U={t/U:.3f}), mode={mode}{label_suffix}\n"
        f"[{_sector_label(spin_preserving)}]   "
        f"T ∈ [{float(T_arr[0]):.3g}, {float(T_arr[-1]):.3g}]  ({len(T_arr)} points)",
        fontsize=11,
    )
    ax1.set_xlabel("Temperature $T$ (K)", fontsize=12)
    ax1.legend(fontsize=9, ncol=2, loc="upper right", framealpha=0.95)
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(left=float(T_arr[0]), right=T_display_max if T_view_max is None else float(T_view_max))
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def _make_figure_t(
    t_values: np.ndarray,
    U: float,
    C_GME: dict,
    n_values: list,
    L: int,
    mode: str,
    solver: str = "gcg",
    spin_preserving: bool = True,
) -> plt.Figure:
    """Single-panel figure: C_GME vs t/U at T=0."""
    fig, ax = plt.subplots(figsize=(8, 5))
    tu = t_values / U
    ylabel = r"$C_{GME}$ (T=0)" if solver == "gcg" else r"$C_{GME}$ lower bound (T=0)"

    for i, n in enumerate(n_values):
        lbl = f"n={n}" + (" (bipartite)" if n == 2 else "")
        ax.plot(tu, C_GME[n],
                color=_COLORS[i % len(_COLORS)],
                marker=_MARKERS[i % len(_MARKERS)],
                ms=4, label=lbl, lw=2)

    ax.set_xlabel("$t/U$", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(
        f"t-dependence of GME: L={L}, U={U}, mode={mode}, T=0  "
        f"[{_sector_label(spin_preserving)}]\n"
        f"t/U ∈ [{float(t_values[0]/U):.3g}, {float(t_values[-1]/U):.3g}]  ({len(t_values)} points)",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def _value_for_n(mapping: dict, n: int):
    """Return mapping[n], accepting stringified keys from older PKLs."""
    if n in mapping:
        return mapping[n]
    return mapping[str(n)]


def _save_excited_singlet_plots_from_result(
    result: dict,
    *,
    L: int,
    t: float,
    U: float,
    mode: str,
    n_electrons: int,
    output_dir: str,
    output_prefix: str,
    solver: str,
    timestamp: str,
    T_view_max: Optional[float] = None
) -> dict:
    """Regenerate excited-singlet plots from computed or cached values."""
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{output_prefix}_" if output_prefix else ""
    n_values = list(result["n_values"])
    n_max = max(n_values)
    C_GME_pure = result["C_GME_pure"]
    C_exact_singlet = float(result["C_exact_singlet"])
    singlet_idx = int(result.get("singlet_idx", -1))

    bar_path = os.path.join(
        output_dir,
        f"{prefix}task2_excited_singlet_GME_L{L}_t{t}_U{U}_T0_{timestamp}.pdf",
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = [float(_value_for_n(C_GME_pure, n)) for n in n_values]
    bars = ax.bar(
        n_values,
        vals,
        color=[_COLORS[i % len(_COLORS)] for i in range(len(n_values))],
    )
    # Raise labels for near-zero bars so they don't stack on the x-axis.
    val_max = max(vals) if vals else 1.0
    for bar, v in zip(bars, vals):
        label = f"{v:.4f}"
        if v < 0.05 * val_max:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    0.03 * val_max, label,
                    ha="center", va="bottom", fontsize=8, color="dimgray")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.01 * val_max, label,
                    ha="center", va="bottom", fontsize=8)
    if np.isfinite(C_exact_singlet):
        ax.axhline(
            C_exact_singlet, color="k", ls="--", lw=1,
            label=f"Exact C_GME(n={n_max}) = {C_exact_singlet:.4f}",
        )
    ax.set_xlabel("Body order $n$", fontsize=12)
    _ylabel = r"$C_{GME}$" if solver == "gcg" else r"$C_{GME}$ lower bound"
    ax.set_ylabel(_ylabel, fontsize=12)
    ax.set_title(
        f"GME of 1st excited singlet: L={L}, t={t}, U={U} (t/U={t/U:.3f})\n"
        f"T=0  (pure eigenstate, idx={singlet_idx})",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(bottom=0)
    ax.set_xticks(n_values)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    result["bar_path"] = bar_path

    T_results = result.get("T_results") or {}
    T_values = T_results.get("T_values")
    if T_values is not None and len(T_values) > 0:
        T_values = np.asarray(T_values, dtype=float)
        plot_path_T = os.path.join(
            output_dir,
            f"{prefix}task2_excited_singlet_thermal_L{L}_t{t}_U{U}_{_grid_tag(T_values, 'T')}_{timestamp}.pdf",
        )
        sinfo = dict(
            L=L, t=t, U=U, mode=mode, Ne=n_electrons,
            gap=None, C_exact_gs=C_exact_singlet,
            label_suffix=f" — 1st excited singlet (idx={singlet_idx})",
        )
        fig = _make_figure_T(
            T_values,
            T_results["C_GME"],
            T_results["purity"],
            n_values,
            sinfo,
            solver=solver,
            T_view_max=T_view_max
        )
        fig.savefig(plot_path_T, dpi=150, bbox_inches="tight")
        plt.close(fig)
        T_results["plot_path"] = plot_path_T
        result["T_results"] = T_results

    result["run_timestamp"] = timestamp
    return result


def _singlet_payload_matches(
    payload: dict,
    *,
    L: int,
    t: float,
    U: float,
    mode: str,
    n_electrons: int,
    n_min: int,
    solver: str,
    n_restarts: int,
    max_iter: int,
    tol: float,
    k_max_extra: int,
    k: int,
    T_values: Optional[np.ndarray],
) -> bool:
    """Return True when a saved excited-singlet PKL matches this request."""
    meta = payload.get("meta", {})
    result = payload.get("result", {})
    if meta.get("analysis_type") != "excited_singlet":
        return False
    checks = [
        int(meta.get("L", -1)) == int(L),
        int(meta.get("Ne", -1)) == int(n_electrons),
        int(meta.get("n_min", -1)) == int(n_min),
        str(meta.get("mode", "")) == str(mode),
        str(meta.get("solver", "")) == str(solver),
        int(meta.get("n_restarts", -1)) == int(n_restarts),
        int(meta.get("max_iter", -1)) == int(max_iter),
        int(meta.get("k_max_extra", -1)) == int(k_max_extra),
        np.isclose(float(meta.get("t", np.nan)), float(t)),
        np.isclose(float(meta.get("U", np.nan)), float(U)),
        np.isclose(float(meta.get("tol", np.nan)), float(tol)),
    ]
    if not all(checks):
        return False
    if "k" in meta and int(meta["k"]) != int(k):
        return False

    requested_T = None if T_values is None else np.asarray(T_values, dtype=float)
    if requested_T is None or len(requested_T) == 0:
        return True

    cached_T = (result.get("T_results") or {}).get("T_values")
    if cached_T is None:
        return False
    cached_T = np.asarray(cached_T, dtype=float)
    return cached_T.shape == requested_T.shape and np.allclose(cached_T, requested_T)


def _find_legacy_singlet_cache(
    *,
    L: int,
    t: float,
    U: float,
    mode: str,
    n_electrons: int,
    n_min: int,
    solver: str,
    n_restarts: int,
    max_iter: int,
    tol: float,
    k_max_extra: int,
    k: int,
    T_values: Optional[np.ndarray],
) -> tuple[Optional[Path], Optional[dict]]:
    """Find a matching timestamped singlet PKL written by older code."""
    pattern = (
        f"singlet_*_L{L}_Ne{n_electrons}_"
        f"t{_safe_float_tag(t)}_U{_safe_float_tag(U)}_"
        f"mode-{mode}_solver-{solver}.pkl"
    )
    candidates = sorted(
        _thermal_runs_dir().glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            payload = _load_thermal_result_pkl(path)
        except Exception:
            continue
        if _singlet_payload_matches(
            payload,
            L=L, t=t, U=U, mode=mode, n_electrons=n_electrons,
            n_min=n_min, solver=solver, n_restarts=n_restarts,
            max_iter=max_iter, tol=tol, k_max_extra=k_max_extra,
            k=k, T_values=T_values,
        ):
            return path, payload
    return None, None


# ── Public API ────────────────────────────────────────────────────────────────

def run_thermal_gme_analysis(
    L: int,
    t: float,
    U: float,
    T_values: np.ndarray,
    mode: str = "uniform",
    n_electrons: Optional[int] = None,
    t_prime: Optional[float] = None,
    t2: Optional[float] = None,
    t_perp: Optional[float] = None,
    n_min: int = 2,
    output_dir: str = "outputs",
    output_prefix: str = "",
    solver: str = "gcg",
    n_restarts: int = 20,
    max_iter: int = 300,
    tol: float = 1e-9,
    k_max_extra: int = 4,
    verbose: bool = True,
    thermal_energy_cutoff: Optional[float] = 10.0,
    adaptive_cutoff: bool = True,
    cutoff_increment: float = 5.0,
    cutoff_rdm_tol: float = 1e-6,
    use_pkl_cache: bool = False,
    cache_version: str = "v1",
    force_recompute: bool = False,
    save_pkl: bool = True,
    n_workers: int = 1,
    average_site_clusters: bool = True,
    spin_preserving: bool = True,\
        T_view_max: Optional[float] = None
) -> dict:
    """
    Compute C_GME(n, T) for n = n_min..L as a function of temperature T.

    Algorithm
    ---------
    1. Full dense diagonalization → all eigenstates {E_m, ψ_m}.
    2. Precompute Γ^(n)[ψ_m] for all m and n (operator cache reused per n).
    3. For each T: form Γ^(n)(T) = Σ_m p_m(T) Γ^(n)[ψ_m], normalise, then:
       - pure state (purity ≈ 1): exact C_GME via compute_C_GME_pure
       - solver='gcg': convex_roof_gcg (true convex-roof optimization)
       - solver='supervisor': fast analytical lower bound (not convex-roof)
       For T > 0, include eigenstates satisfying
       (E_m - E_0) / T < thermal_energy_cutoff, with k_B = 1.

    Notes
    -----
    - By default, averages over all contiguous site clusters for n < L and
      does this cluster work in parallel when n_workers > 1.  Set
      average_site_clusters=False to use only sites (0,..,n-1).
    - For n=2, mixed states use the convex roof (same as all other n).
    - At T→0, C_GCG(n=L) should match the exact ground-state value.

    Parameters
    ----------
    L            : number of sites
    t            : hopping amplitude
    U            : on-site interaction (set U = 10*t for t/U = 0.1)
    T_values     : 1-D temperature array (in units of t)
    mode         : lattice topology (default 'uniform')
    n_electrons  : total electrons (default L for half-filling)
    t_prime      : NNN hopping for mode='nnn'
    t2           : weak bond hopping for mode='ssh'
    t_perp       : rung hopping for mode='ladder'
    n_min        : minimum body order (default 2)
    output_dir   : directory for output PDF
    output_prefix: filename prefix
    solver       : 'gcg' (default) or 'supervisor'
    n_restarts   : GCG restarts per k (default 20)
    max_iter     : GCG iterations per restart (default 300)
    tol          : GCG convergence tolerance (default 1e-9)
    k_max_extra  : extra k steps beyond rank in GCG (default 4)
    verbose      : print progress
    thermal_energy_cutoff:
                   initial threshold for (E_m - E0)/T; default 10 as
                   recommended by supervisor
    adaptive_cutoff:
                   if True (default), apply the supervisor's convergence
                   criterion: increase cutoff by cutoff_increment until
                   the Frobenius-norm change in all Γ^(n)(T) < cutoff_rdm_tol
    cutoff_increment:
                   step size for adaptive cutoff search (default 5.0)
    cutoff_rdm_tol:
                   Frobenius-norm tolerance for adaptive cutoff convergence
    use_pkl_cache:
                   load/save the complete thermal result under
                   results/rdm_pkl_results/thermal_runs
    cache_version : version tag included in thermal PKL filenames
    force_recompute:
                   ignore an existing thermal PKL and recompute
    save_pkl      : when use_pkl_cache=True, write the computed result to PKL
    n_workers     : parallel worker processes for site-cluster RDM bank and
                    thermal GME jobs
    average_site_clusters:
                   if True, average over all contiguous clusters for n < L;
                   for L=6 this gives 6 cluster jobs per n

    Returns
    -------
    dict
        T_values, C_GME {n: array}, purity {n: array},
        C_exact_gs, energies, gap, n_values, system_info, plot_path,
        solver, thermal_energy_cutoff
    """
    if n_electrons is None:
        n_electrons = L
    # Cluster size cannot exceed the electron count: the n-site RDM trace is
    # ⟨Π_site n̂_site⟩, which is exactly 0 once n > Ne (can't occupy more sites
    # than there are electrons). Cap at min(L, Ne) so doped (Ne<L) runs stop at
    # n=Ne instead of hitting a zero-trace RDM. At half-filling Ne=L → no change.
    n_max = min(L, n_electrons)
    n_values = list(range(n_min, n_max + 1))
    T_values = np.asarray(T_values, dtype=float)

    cache_path = _thermal_cache_path(
        analysis_type="thermal",
        L=L, t=t, U=U, t_prime=t_prime, t2=t2, t_perp=t_perp,
        mode=mode, n_electrons=n_electrons, n_min=n_min,
        solver=solver, n_restarts=n_restarts, max_iter=max_iter, tol=tol,
        k_max_extra=k_max_extra, grid_name="T", grid_values=T_values,
        cache_version=cache_version, thermal_energy_cutoff=thermal_energy_cutoff,
        average_site_clusters=average_site_clusters,
        adaptive_cutoff=adaptive_cutoff, cutoff_increment=cutoff_increment,
        spin_preserving=spin_preserving,
    )

    if use_pkl_cache and not force_recompute and cache_path.exists():
        payload = _load_thermal_result_pkl(cache_path)
        result = payload["result"]
        result["loaded_from_cache"] = True
        result["cache_path"] = str(cache_path)
        result["cache_meta"] = payload.get("meta", {})

        os.makedirs(output_dir, exist_ok=True)
        ts_cache = _run_timestamp()
        prefix = f"{output_prefix}_" if output_prefix else ""
        plot_path = os.path.join(
            output_dir, f"{prefix}task1_thermal_GME_L{L}_t{t}_U{U}_{_sector_tag(spin_preserving)}_{_grid_tag(np.asarray(result['T_values']), 'T')}_{ts_cache}.pdf"
        )
        fig = _make_figure_T(
            np.asarray(result["T_values"]),
            result["C_GME"],
            result["purity"],
            result["n_values"],
            result["system_info"],
            solver=result.get("solver", solver),
            spin_preserving=spin_preserving,
            T_view_max=T_view_max
        )
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        result["plot_path"] = plot_path

        print(f"\n[cache] loaded thermal result PKL → {cache_path}")
        print(f"[cache] regenerated plot → {plot_path}")
        return result

    run_key = hashlib.sha256(str(cache_path.name).encode("utf-8")).hexdigest()[:16]

    print(f"\n{'='*64}")
    print(f"  THERMAL GME ANALYSIS — ground-state thermal ensemble")
    print(f"{'='*64}")
    print(f"  L={L}  t={t}  U={U}  t/U={t/U:.4f}  Ne={n_electrons}  mode={mode}")
    if t_prime is not None or t2 is not None or t_perp is not None:
        print(f"  topology extras: t_prime={t_prime}  t2={t2}  t_perp={t_perp}")
    print(f"  solver={solver}  n_restarts={n_restarts}  max_iter={max_iter}")
    print(f"  RDM workers={max(1, int(n_workers or 1))}  "
          f"average_site_clusters={average_site_clusters}")
    cutoff_msg = "all states" if thermal_energy_cutoff is None else thermal_energy_cutoff
    adaptive_msg = (
        f"adaptive (start={thermal_energy_cutoff}, step={cutoff_increment}, "
        f"tol={cutoff_rdm_tol:.0e})"
        if adaptive_cutoff and thermal_energy_cutoff is not None
        else str(cutoff_msg)
    )
    print(f"  thermal energy cutoff: {adaptive_msg}")
    print(f"  n = {n_min}..{n_max}   T range [{T_values[0]:.4g}, {T_values[-1]:.4g}]"
          f"  ({len(T_values)} points)")
    print(f"{'-'*64}")

    # 1+2. Full diagonalization + RDM bank via the shared, task-independent
    #       generic bank cache (results/rdm_pkl_results/rdm_bank/). On a cache
    #       hit the (expensive) diagonalization is skipped entirely.
    print("\n[1-2] RDM bank (build or load from generic cache) …", flush=True)
    bank_info = build_or_load_rdm_bank(
        L=L, t=t, U=U, mode=mode, n_electrons=n_electrons, n_min=n_min,
        t_prime=t_prime, t2=t2, t_perp=t_perp,
        average_site_clusters=average_site_clusters,
        n_workers=n_workers, spin_preserving=spin_preserving,
        use_pkl_cache=use_pkl_cache, cache_version=cache_version,
        force_recompute=force_recompute, save_pkl=save_pkl, verbose=verbose,
    )
    energies = bank_info["energies"]
    bank = bank_info["bank"]
    n_so = bank_info["n_so"]
    rdm_bank_cache_path = bank_info["cache_path"]
    loaded_rdm_bank_cache = bank_info["loaded_from_cache"]
    n_states = len(energies)
    E0, gap = energies[0], energies[1] - energies[0]
    print(
        f"    {n_states} eigenstates  E0={E0:.6f}  gap Δ={gap:.6f}"
        + ("  [loaded from generic cache]" if loaded_rdm_bank_cache else ""),
        flush=True,
    )

    # Exact C_GME at T=0 for the full (n=L) system
    rho_gs_full = bank[n_max][0]
    tr_gs = np.real(np.trace(rho_gs_full))
    rho_gs_full = rho_gs_full / tr_gs if abs(tr_gs) > 1e-14 else rho_gs_full
    C_exact_gs = (float(compute_C_GME_pure(rho_gs_full, n_max))
                  if compute_purity(rho_gs_full) > 1.0 - 1e-6 else float("nan"))
    print(f"\n    Exact C_GME(n={n_max}, T=0) = {C_exact_gs:.6f}")

    # 3. Temperature sweep
    print(f"\n[3] Temperature sweep ({len(T_values)} points) …", flush=True)
    C_GME: dict[int, np.ndarray] = {n: np.zeros(len(T_values)) for n in n_values}
    purity_dict: dict[int, np.ndarray] = {n: np.zeros(len(T_values)) for n in n_values}
    active_counts = np.zeros(len(T_values), dtype=int)
    converged_cutoffs = np.zeros(len(T_values))
    gme_tasks = []
    gme_job_paths: dict[tuple[int, int], str] = {}
    n_gme_jobs_reused = 0

    for i, T in enumerate(T_values):
        if T <= 1e-14:
            active = np.where(np.abs(energies - E0) < 1e-10)[0]
            probs_active = np.full(len(active), 1.0 / len(active))
            converged_cutoffs[i] = 0.0
        elif adaptive_cutoff and thermal_energy_cutoff is not None:
            cut, active, probs_active = _converged_thermal_cutoff(
                energies, T, E0, bank, n_values,
                initial_cutoff=thermal_energy_cutoff,
                increment=cutoff_increment,
                rdm_tol=cutoff_rdm_tol,
            )
            converged_cutoffs[i] = cut
        else:
            active, probs_active = _thermal_active_probabilities(
                energies, T, E0, thermal_energy_cutoff
            )
            converged_cutoffs[i] = thermal_energy_cutoff if thermal_energy_cutoff is not None else -1.0
        active_counts[i] = len(active)

        for n in n_values:
            Gamma_T = np.einsum("m,mij->ij", probs_active, bank[n][active])
            Gamma_T = normalize_density_matrix(Gamma_T)
            gamma_hash = _array_sha256(Gamma_T)
            job_path = _thermal_gme_job_cache_path(
                run_key=run_key,
                i=i,
                T=float(T),
                n=n,
                gamma_hash=gamma_hash,
            )
            if use_pkl_cache and not force_recompute:
                cached_job = _load_thermal_gme_job_pkl(job_path)
                if cached_job is not None:
                    job_result = cached_job["result"]
                    C_GME[n][i] = float(job_result["C_value"])
                    purity_dict[n][i] = float(job_result["purity"])
                    gme_job_paths[(i, n)] = str(job_path)
                    n_gme_jobs_reused += 1
                    continue
            gme_tasks.append((
                i,
                float(T),
                n,
                Gamma_T,
                solver,
                n_restarts,
                max_iter,
                tol,
                k_max_extra,
                24680 + 1000 * i + n,
                str(job_path),
                gamma_hash,
            ))

    n_gme_workers = max(1, int(n_workers or 1))
    print(
        f"    GME jobs: {len(gme_tasks)} queued, {n_gme_jobs_reused} reused "
        f"(workers={n_gme_workers}, solver={solver})",
        flush=True,
    )

    def _record_gme_job_result(i, T, n, c_val, pur, elapsed, job_path, gamma_hash,
                               detail=None):
        C_GME[n][i] = c_val
        purity_dict[n][i] = pur
        if use_pkl_cache and save_pkl and job_path:
            # Full solver record (bounds, effective settings, per-k / per-restart
            # logs), matching the standalone GME pipeline's job files.
            result_full, decompositions = _gme_job_detail_payload(
                detail, solver=solver, n=n, purity=pur,
            )
            payload = {
                "meta": {
                    "object_type": "hubbard_thermal_gme_job",
                    "saved_at": datetime.now().isoformat(timespec="seconds"),
                    "run_key": run_key,
                    "T_index": int(i),
                    "T": float(T),
                    "n": int(n),
                    "solver": solver,
                    "n_restarts": int(n_restarts),
                    "max_iter": int(max_iter) if max_iter is not None else None,
                    "tol": float(tol) if tol is not None else None,
                    "k_max_extra": int(k_max_extra),
                    "gamma_hash": str(gamma_hash),
                    "elapsed_seconds": float(elapsed),
                    "spin_preserving": bool(spin_preserving),
                    "sector": _sector_label(spin_preserving),
                    "environment": _basic_env_meta(),
                },
                "result": {
                    "C_value": float(c_val),
                    "purity": float(pur),
                    "elapsed_seconds": float(elapsed),
                    "spin_preserving": bool(spin_preserving),
                },
                "result_full": result_full,
                "decompositions": decompositions,
            }
            try:
                p = _save_thermal_gme_job_pkl(
                    cache_path=Path(job_path),
                    payload=payload,
                )
                gme_job_paths[(int(i), int(n))] = p
            except Exception as exc:
                print(f"      [cache] WARN could not save thermal GME job: {exc!r}", flush=True)

    if n_gme_workers > 1 and len(gme_tasks) > 1:
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(it, **kw):  # noqa: E306
                return it

        import multiprocessing as mp
        with mp.Pool(processes=n_gme_workers) as pool:
            iterator = pool.imap_unordered(_thermal_gme_worker, gme_tasks)
            for i, T, n, c_val, pur, elapsed, job_path, gamma_hash, detail in tqdm(
                iterator,
                total=len(gme_tasks),
                desc="    Thermal GME",
                unit="job",
                ncols=90,
            ):
                _record_gme_job_result(
                    i, T, n, c_val, pur, elapsed, job_path, gamma_hash, detail
                )
                if verbose:
                    tag = "C_GCG" if solver == "gcg" else "C_LB"
                    print(
                        f"      done T={T:.4g}, n={n}, {tag}={c_val:.4f}, "
                        f"{elapsed:.2f}s",
                        flush=True,
                    )
    else:
        for task in gme_tasks:
            i, T, n, c_val, pur, elapsed, job_path, gamma_hash, detail = (
                _thermal_gme_worker(task)
            )
            _record_gme_job_result(
                i, T, n, c_val, pur, elapsed, job_path, gamma_hash, detail
            )

    if verbose:
        for i, T in enumerate(T_values):
            if i % max(1, len(T_values) // 10) == 0 or i == len(T_values) - 1:
                tag = "C_GCG" if solver == "gcg" else "C_LB"
                vals = "  ".join(f"{tag}(n={n})={C_GME[n][i]:.4f}" for n in n_values)
                cut_str = (
                    f"  cutoff={converged_cutoffs[i]:.1f}"
                    if adaptive_cutoff and T > 1e-14 else ""
                )
                print(
                    f"    T={T:.4g}  active={active_counts[i]}{cut_str}  {vals}",
                    flush=True,
                )

    # 4. Plot
    os.makedirs(output_dir, exist_ok=True)
    ts = _run_timestamp()
    prefix = f"{output_prefix}_" if output_prefix else ""
    plot_path = os.path.join(
        output_dir, f"{prefix}task1_thermal_GME_L{L}_t{t}_U{U}_{_sector_tag(spin_preserving)}_{_grid_tag(T_values, 'T')}_{ts}.pdf"
    )
    sinfo = dict(L=L, t=t, U=U, mode=mode, Ne=n_electrons,
                 gap=gap, C_exact_gs=C_exact_gs)
    fig = _make_figure_T(T_values, C_GME, purity_dict, n_values, sinfo,
                         solver=solver, spin_preserving=spin_preserving, T_view_max=T_view_max)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n    Plot → {plot_path}")

    result = {
        "T_values":              T_values,
        "C_GME":                 C_GME,
        "purity":                purity_dict,
        "C_exact_gs":            C_exact_gs,
        "energies":              energies,
        "gap":                   gap,
        "n_values":              n_values,
        "system_info":           sinfo,
        "plot_path":             plot_path,
        "solver":                solver,
        "thermal_energy_cutoff": thermal_energy_cutoff,
        "adaptive_cutoff":       bool(adaptive_cutoff),
        "converged_cutoffs":     converged_cutoffs,
        "loaded_from_cache":     False,
        "cache_path":            None,
        "n_workers":             int(n_workers),
        "gme_workers":           int(n_gme_workers),
        "average_site_clusters": bool(average_site_clusters),
        "gme_jobs_reused":       int(n_gme_jobs_reused),
        "gme_jobs_computed":     int(len(gme_tasks)),
        "gme_job_paths":         gme_job_paths,
        "rdm_bank_cache_path":   str(rdm_bank_cache_path),
        "loaded_rdm_bank_cache": bool(loaded_rdm_bank_cache),
    }

    if use_pkl_cache and save_pkl:
        result["cache_path"] = str(cache_path)
        metadata = {
            "analysis_type": "thermal",
            "cache_version": cache_version,
            "L": int(L),
            "t": float(t),
            "U": float(U),
            "mode": str(mode),
            "t_prime": None if t_prime is None else float(t_prime),
            "t2": None if t2 is None else float(t2),
            "t_perp": None if t_perp is None else float(t_perp),
            "Ne": int(n_electrons),
            "n_min": int(n_min),
            "n_max": int(n_max),
            "n_values": list(n_values),
            "T_values": T_values,
            "T_digest": _array_digest(T_values),
            "solver": solver,
            "n_restarts": int(n_restarts),
            "max_iter": int(max_iter) if max_iter is not None else None,
            "tol": float(tol) if tol is not None else None,
            "k_max_extra": int(k_max_extra),
            "thermal_energy_cutoff": thermal_energy_cutoff,
            "adaptive_cutoff": bool(adaptive_cutoff),
            "cutoff_increment": float(cutoff_increment),
            "cutoff_rdm_tol": float(cutoff_rdm_tol),
            "converged_cutoffs": converged_cutoffs.tolist(),
            "average_site_clusters": bool(average_site_clusters),
            "n_workers": int(n_workers),
            "gme_workers": int(n_gme_workers),
            "gme_jobs_reused": int(n_gme_jobs_reused),
            "gme_jobs_computed": int(len(gme_tasks)),
            "gme_job_paths": gme_job_paths,
            "rdm_bank_cache_path": str(rdm_bank_cache_path),
            "loaded_rdm_bank_cache": bool(loaded_rdm_bank_cache),
            "plot_path": plot_path,
            "run_timestamp": ts,
        }
        pkl_path = _save_thermal_result_pkl(
            cache_path=cache_path,
            metadata=metadata,
            result=result,
        )
        result["cache_path"] = pkl_path
        print(f"    Thermal result PKL → {pkl_path}")

    return result


# ── First excited singlet ─────────────────────────────────────────────────────

def find_first_excited_singlet(
    L: int,
    t: float,
    U: float,
    V: float,
    mode: str = "uniform",
    n_electrons: Optional[int] = None,
    k: int = 12,
    spin_preserving: bool = True,
) -> dict:
    """
    Find the first S=0 excited state (first singlet above the triplet manifold).

    Parameters
    ----------
    k : number of eigenstates to compute (increase if singlet not found)
    spin_preserving :
        Which space to diagonalise in — True = S_z=0 sector (default, the
        historical behaviour), False = full Hamiltonian. This MUST match the
        space the caller builds its RDM operators in, otherwise the returned
        eigenvector lives in a different (smaller) Hilbert space.

    Returns
    -------
    dict with keys:
        idx        : 0-based eigenstate index
        energy     : energy of the singlet
        psi        : eigenvector
        energies   : full array of k energies (for inspection)
        eigvecs    : full (dim, k) eigenvector matrix
        spin_table : list of spin-classification dicts
        n_so       : n_spin_orbitals
    """
    if n_electrons is None:
        n_electrons = L

    energies, eigvecs, n_so = compute_k_eigenstates(
        L=L, t=t, U=U, mode=mode, n_electrons=n_electrons, k=k,
        spin_preserving=spin_preserving, V=V,
    )
    spin_table = classify_eigenstates_spin(
        eigvecs=eigvecs, energies=energies,
        L=L, n_electrons=n_electrons, n_spin_orbitals=n_so,
        # the S^2 operator must be built in the SAME space the eigenvectors
        # live in, otherwise <psi|S^2|psi> is a dimension mismatch
        spin_preserving=spin_preserving,
    )

    print("\n  Eigenstate spin classification:")
    print(f"  {'idx':>3}  {'E':>14}  {'<S²>':>8}  {'S':>5}  label")
    print("  " + "-" * 50)
    for row in spin_table:
        print(f"  {row['idx']:>3}  {row['energy']:>14.8f}  "
              f"{row['S2']:>8.4f}  {row['S']:>5.3f}  {row['spin_label']}")

    # Ground state must be singlet; find NEXT singlet (skip triplets etc.)
    gs_singlet = next(
        r for r in spin_table if abs(r["S"]) < 0.05
    )
    excited_singlets = [
        r for r in spin_table
        if abs(r["S"]) < 0.05 and r["idx"] != gs_singlet["idx"]
    ]
    if not excited_singlets:
        raise RuntimeError(
            f"No excited singlet found in the first {k} eigenstates. "
            f"Try increasing k."
        )
    first_singlet = excited_singlets[0]
    idx = first_singlet["idx"]

    print(f"\n  → First excited singlet: idx={idx}  E={first_singlet['energy']:.8f}")
    return {
        "idx":       idx,
        "energy":    first_singlet["energy"],
        "psi":       eigvecs[:, idx],
        "energies":  energies,
        "eigvecs":   eigvecs,
        "spin_table": spin_table,
        "n_so":      n_so,
    }

def run_excited_singlet_gme(
    L: int,
    t: float,
    U: float,
    mode: str = "uniform",
    n_electrons: Optional[int] = None,
    n_min: int = 2,
    output_dir: str = "outputs",
    output_prefix: str = "",
    T_values: Optional[np.ndarray] = None,
    k: int = 12,
    solver: str = "gcg",
    n_restarts: int = 20,
    max_iter: int = 300,
    tol: float = 1e-9,
    k_max_extra: int = 4,
    verbose: bool = True,
    thermal_energy_cutoff: Optional[float] = 10.0,
    adaptive_cutoff: bool = True,
    cutoff_increment: float = 5.0,
    cutoff_rdm_tol: float = 1e-6,
    use_pkl_cache: bool = False,
    cache_version: str = "v1",
    force_recompute: bool = False,
    save_pkl: bool = True,
    n_workers: int = 1,
    spin_preserving: bool = True,
    V: float = 0.0,  # optional nearest-neighbour interaction
    T_view_max: Optional[float] = None,
) -> dict:
    """
    GME analysis of the first singlet excited state.

    Pure-state analysis (T=0): computes nBRDMs of the singlet and reports
    exact C_GME for n=L and GCG convex-roof for n=2..L-1.

    If T_values is provided, also runs a thermal analysis: the standard
    Gibbs ensemble Γ^(n)(T) = Σ_m p_m(T) Γ^(n)[ψ_m] (shifted so the
    singlet is the zero-energy reference).

    Parameters
    ----------
    k        : eigenstates for spin classification (increase if singlet not found)
    solver   : 'gcg' (default) or 'supervisor'
    n_restarts, max_iter, tol, k_max_extra : GCG solver parameters
    T_values : optional temperature array; None → pure-state analysis only
    thermal_energy_cutoff:
               for optional T>0 analysis, include eigenstates with
               (E_m - E_singlet)/T below this threshold; None includes all

    Returns
    -------
    dict with keys:
        singlet_idx, singlet_energy, n_values,
        C_GME_pure {n: float},   (T=0 values from GCG)
        C_exact_singlet          (n=L, exact formula)
        T_results (if T_values provided)
    """
    if n_electrons is None:
        n_electrons = L
    # Cluster size cannot exceed the electron count: the n-site RDM trace is
    # ⟨Π_site n̂_site⟩, which is exactly 0 once n > Ne (can't occupy more sites
    # than there are electrons). Cap at min(L, Ne) so doped (Ne<L) runs stop at
    # n=Ne instead of hitting a zero-trace RDM. At half-filling Ne=L → no change.
    n_max = min(L, n_electrons)
    n_values = list(range(n_min, n_max + 1))
    if T_values is not None:
        T_values = np.asarray(T_values, dtype=float)
    ts = _run_timestamp()
    cache_path = _thermal_cache_path(
        analysis_type="excited_singlet",
        L=L, t=t, U=U,
        t_prime=None, t2=None, t_perp=None,
        mode=mode, n_electrons=n_electrons,
        n_min=n_min, solver=solver,
        n_restarts=n_restarts, max_iter=max_iter, tol=tol,
        k_max_extra=k_max_extra,
        grid_name="T", grid_values=T_values,
        cache_version=cache_version,
        thermal_energy_cutoff=thermal_energy_cutoff,
        adaptive_cutoff=adaptive_cutoff,
        cutoff_increment=cutoff_increment,
        k=k,
        spin_preserving=spin_preserving,
    )

    print(f"\n{'='*64}")
    print(f"  EXCITED SINGLET GME ANALYSIS")
    print(f"{'='*64}")
    print(f"  L={L}  t={t}  U={U}  t/U={t/U:.4f}  Ne={n_electrons}  mode={mode}")

    if use_pkl_cache and not force_recompute:
        loaded_path: Optional[Path] = None
        payload: Optional[dict] = None
        if cache_path.exists():
            payload = _load_thermal_result_pkl(cache_path)
            loaded_path = cache_path
        else:
            loaded_path, payload = _find_legacy_singlet_cache(
                L=L, t=t, U=U, mode=mode, n_electrons=n_electrons,
                n_min=n_min, solver=solver, n_restarts=n_restarts,
                max_iter=max_iter, tol=tol, k_max_extra=k_max_extra,
                k=k, T_values=T_values,
            )

        if payload is not None and loaded_path is not None:
            result = dict(payload["result"])
            result = _save_excited_singlet_plots_from_result(
                result,
                L=L, t=t, U=U, mode=mode, n_electrons=n_electrons,
                output_dir=output_dir, output_prefix=output_prefix,
                solver=solver, timestamp=ts, T_view_max=T_view_max
            )
            result["loaded_from_cache"] = True
            result["cache_path"] = str(cache_path)
            result["pkl_path"] = str(loaded_path)
            if loaded_path != cache_path and save_pkl:
                metadata = {
                    **payload.get("meta", {}),
                    "analysis_type": "excited_singlet",
                    "cache_version": cache_version,
                    "migrated_from": str(loaded_path),
                    "bar_path": result.get("bar_path"),
                    "run_timestamp": ts,
                }
                _save_thermal_result_pkl(
                    cache_path=cache_path, metadata=metadata, result=result
                )
                result["pkl_path"] = str(cache_path)
            print(f"\n[cache] loaded excited singlet PKL → {loaded_path}")
            print(f"    Bar plot → {result['bar_path']}")
            if result.get("T_results", {}).get("plot_path"):
                print(f"    Plot → {result['T_results']['plot_path']}")
            return result

    # Find the first excited singlet
    singlet_info = find_first_excited_singlet(L, t, U,V, mode, n_electrons, k=k)
    singlet_idx = singlet_info["idx"]
    psi_singlet = singlet_info["psi"]
    E_singlet   = singlet_info["energy"]
    n_so        = singlet_info["n_so"]

    # Pure-state nBRDM analysis of the singlet
    print(f"\n[1] Pure-state nBRDM for singlet (idx={singlet_idx})  solver={solver} …", flush=True)
    ops_cache = {}
    C_GME_pure: dict[int, float] = {}
    purity_pure: dict[int, float] = {}

    for n in n_values:
        sites = tuple(range(n))
        if n not in ops_cache:
            ops_cache[n] = _precompute_ops(sites, n_so, n_electrons,
                                         spin_preserving=spin_preserving)
        rho_raw = _rdm_from_ops(ops_cache[n], psi_singlet, n)
        tr = np.real(np.trace(rho_raw))
        rho = rho_raw / tr if abs(tr) > 1e-14 else rho_raw
        c_val, pur, _detail = _gme_lower_bound(
            rho, n,
            solver=solver, n_restarts=n_restarts,
            max_iter=max_iter, tol=tol, k_max_extra=k_max_extra,
        )
        C_GME_pure[n] = c_val
        purity_pure[n] = pur
        tag = "C_GCG" if solver == "gcg" else "C_LB"
        print(f"    n={n}  purity={pur:.6f}  {tag}={c_val:.6f}")

    # Exact C_GME for n=L (pure singlet state)
    sites_full = tuple(range(n_max))
    rho_full_raw = _rdm_from_ops(ops_cache[n_max], psi_singlet, n_max)
    tr_full = np.real(np.trace(rho_full_raw))
    rho_full = rho_full_raw / tr_full if abs(tr_full) > 1e-14 else rho_full_raw
    C_exact_singlet = (float(compute_C_GME_pure(rho_full, n_max))
                       if compute_purity(rho_full) > 1.0 - 1e-6 else float("nan"))
    print(f"\n    Exact C_GME(n={n_max}, singlet) = {C_exact_singlet:.6f}")

    # Thermal analysis relative to singlet energy (optional)
    T_results: dict = {}
    if T_values is not None and len(T_values) > 0:
        print(f"\n[2] Thermal analysis relative to singlet "
              f"(T range [{T_values[0]:.4g}, {T_values[-1]:.4g}]) …", flush=True)
        cutoff_msg = "all states" if thermal_energy_cutoff is None else thermal_energy_cutoff
        print(f"    thermal energy cutoff: {cutoff_msg}", flush=True)

        # Use full eigenstates; E_singlet is passed as reference to Boltzmann weights
        energies_full = singlet_info["energies"]
        eigvecs_full  = singlet_info["eigvecs"]

        bank = _build_rdm_bank(n_values, eigvecs_full, n_so, n_electrons,
                                verbose=verbose, spin_preserving=spin_preserving)

        C_GME_T: dict[int, np.ndarray] = {n: np.zeros(len(T_values)) for n in n_values}
        purity_T: dict[int, np.ndarray] = {n: np.zeros(len(T_values)) for n in n_values}

        # Pre-compute all (T, n) thermal RDMs, then dispatch GCG jobs in parallel.
        jobs_tsweep = []
        for i, T in enumerate(T_values):
            if float(T) <= 1e-14:
                active = np.where(np.abs(energies_full - E_singlet) < 1e-10)[0]
                probs_active = np.full(len(active), 1.0 / len(active))
            elif adaptive_cutoff and thermal_energy_cutoff is not None:
                _, active, probs_active = _converged_thermal_cutoff(
                    energies_full, T, E_singlet, bank, n_values,
                    initial_cutoff=thermal_energy_cutoff,
                    increment=cutoff_increment,
                    rdm_tol=cutoff_rdm_tol,
                )
            else:
                active, probs_active = _thermal_active_probabilities(
                    energies_full, T, E_singlet, thermal_energy_cutoff
                )
            for n in n_values:
                Gamma_T = np.einsum("m,mij->ij", probs_active, bank[n][active])
                Gamma_T = normalize_density_matrix(Gamma_T)
                jobs_tsweep.append((i, float(T), n, Gamma_T,
                                    solver, n_restarts, max_iter, tol, k_max_extra,
                                    False, np.random.randint(0, 2**31),
                                    "no_cache", "no_hash"))

        n_jobs = len(jobs_tsweep)
        n_w = max(1, int(n_workers or 1))
        print(f"\n    T-sweep GME: {n_jobs} jobs (workers={n_w}) …", flush=True)

        try:
            from tqdm import tqdm as _tqdm
        except ImportError:
            def _tqdm(it, **kw):
                return it

        if n_w > 1 and n_jobs > 1:
            import multiprocessing as mp
            with mp.Pool(processes=n_w) as pool:
                for i, T, n, c_val, pur, _bip, _dt, _jp, _gh in _tqdm(
                    pool.imap_unordered(_thermal_gme_worker, jobs_tsweep),
                    total=n_jobs, desc="  Singlet T-sweep GME", unit="job", ncols=90,
                ):
                    C_GME_T[n][i] = c_val
                    purity_T[n][i] = pur
        else:
            for job in _tqdm(jobs_tsweep, desc="  Singlet T-sweep GME", unit="job", ncols=90):
                i, T, n, Gamma_T, *rest = job
                c_val, pur, _detail = _gme_lower_bound(
                    Gamma_T, n, solver=solver, n_restarts=n_restarts,
                    max_iter=max_iter, tol=tol, k_max_extra=k_max_extra,
                )
                C_GME_T[n][i] = c_val
                purity_T[n][i] = pur

        T_results = {
            "T_values": T_values,
            "C_GME":    C_GME_T,
            "purity":   purity_T,
            "thermal_energy_cutoff": thermal_energy_cutoff,
        }

        os.makedirs(output_dir, exist_ok=True)
        prefix = f"{output_prefix}_" if output_prefix else ""
        plot_path_T = os.path.join(
            output_dir,
            f"{prefix}task2_excited_singlet_thermal_L{L}_t{t}_U{U}_{_grid_tag(T_values, 'T')}_{ts}.pdf",
        )
        sinfo = dict(L=L, t=t, U=U, mode=mode, Ne=n_electrons,
                     gap=None, C_exact_gs=C_exact_singlet,
                     label_suffix=f" — 1st excited singlet (idx={singlet_idx})")
        fig = _make_figure_T(T_values, C_GME_T, purity_T, n_values, sinfo, solver=solver, spin_preserving=spin_preserving, T_view_max=T_view_max)
        fig.savefig(plot_path_T, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n    Plot → {plot_path_T}")
        T_results["plot_path"] = plot_path_T

    result = {
        "singlet_idx":      singlet_idx,
        "singlet_energy":   E_singlet,
        "n_values":         n_values,
        "C_GME_pure":       C_GME_pure,
        "purity_pure":      purity_pure,
        "C_exact_singlet":  C_exact_singlet,
        "spin_table":       singlet_info["spin_table"],
        "T_results":        T_results,
        "run_timestamp":    ts,
    }
    result = _save_excited_singlet_plots_from_result(
        result,
        L=L, t=t, U=U, mode=mode, n_electrons=n_electrons,
        output_dir=output_dir, output_prefix=output_prefix,
        solver=solver, timestamp=ts, T_view_max=T_view_max
    )
    print(f"    Bar plot → {result['bar_path']}")

    if save_pkl:
        metadata = {
            "analysis_type": "excited_singlet",
            "cache_version": cache_version,
            "L": int(L), "t": float(t), "U": float(U),
            "mode": str(mode), "Ne": int(n_electrons),
            "n_min": int(n_min), "n_max": int(n_max),
            "n_values": list(n_values),
            "k": int(k),
            "singlet_idx": int(singlet_idx),
            "singlet_energy": float(E_singlet),
            "solver": solver, "n_restarts": int(n_restarts),
            "max_iter": int(max_iter), "tol": float(tol),
            "k_max_extra": int(k_max_extra),
            "thermal_energy_cutoff": thermal_energy_cutoff,
            "adaptive_cutoff": bool(adaptive_cutoff),
            "cutoff_increment": float(cutoff_increment),
            "cutoff_rdm_tol": float(cutoff_rdm_tol),
            "bar_path": result["bar_path"],
            "run_timestamp": ts,
        }
        _save_thermal_result_pkl(
            cache_path=cache_path, metadata=metadata, result=result
        )
        result["cache_path"] = str(cache_path)
        result["pkl_path"] = str(cache_path)
        print(f"    Excited singlet PKL → {cache_path}")

    return result


# ── t-dependence scan ─────────────────────────────────────────────────────────

def _tscan_cache_path(
    *,
    L: int,
    U: float,
    mode: str,
    n_electrons: int,
    n_min: int,
    solver: str,
    n_restarts: int,
    max_iter: int,
    tol: float,
    k_max_extra: int,
    t_values: np.ndarray,
    cache_version: str,
    spin_preserving: bool = True,
    excited_singlet: bool = False,
    V: float = 0.0,
) -> Path:
    """Deterministic PKL path for one t-scan run."""
    t_arr = np.asarray(t_values, dtype=float)
    filename = (
        f"tscan_{cache_version}_"
        f"L{L}_Ne{n_electrons}_U{_safe_float_tag(U)}_"
        f"mode-{mode}_nmin{n_min}_solver-{solver}_"
        f"{_sector_tag(spin_preserving)}"
        # tags added ONLY when set, so existing ground-state / V=0 PKLs keep
        # their names and stay cache-reusable
        f"{'_es' if excited_singlet else ''}"
        f"{('_V' + _safe_float_tag(float(V))) if abs(V) > 1e-12 else ''}_"
        f"restarts{int(n_restarts)}_iter{int(max_iter)}_"
        f"kmax{int(k_max_extra)}_tol{_safe_float_tag(float(tol))}_"
        f"{_grid_tag(t_arr, 't')}-{_array_digest(t_arr)}.pkl"
    )
    return _thermal_runs_dir() / filename


def plot_scan_finalized(result: dict, plot_path: str,
                        spin_preserving: bool = True,
                        save_png: bool = False) -> str:
    """
    Draw the FINALIZED (replot-style) scan figure directly from a result dict
    and save it to ``plot_path``. Handles both t-scans (key ``t_values``) and
    U-scans (key ``U_values``). This is the single source of truth for scan
    figures, used by run_t_dependence_gme / run_U_dependence_gme and by the
    standalone replot script, so every scan automatically produces the same
    finalized style (8-colour palette, n=L labelled pure, Wootters reference
    for n=2). Returns ``plot_path``.
    """
    si = result.get("system_info", {}) or {}
    L = int(si.get("L", 0))
    mode = si.get("mode", "uniform")
    C = result["C_GME"]
    # ns = sorted(C)
    Ne = int(si.get("Ne", L) or L)
    ns = [n for n in sorted(C) if n <= Ne]     # never draw n > Ne (zero-trace RDM)

    is_t = "t_values" in result
    is_v = "V_values" in result
    V = float(si.get("V", 0.0))

    if is_v:                                     # V-scan: fix U & t, sweep V
        t = float(si.get("t", 1.0)); U = float(si.get("U", 0.0))
        x = (np.asarray(result["V_over_t"], float) if "V_over_t" in result
             else np.asarray(result["V_values"], float) / t)
        xlabel, fixed, xkind = r"$V/t$", f"U={U}, t={t}", "V"
    elif is_t:                                   # t-scan: fix U, sweep t
        U = float(si.get("U", 1.0))
        x = np.asarray(result["t_values"], float) / U
        xlabel, fixed, xkind = r"$t/U$", f"U={U}", "t"
    else:                                        # U-scan: fix t, sweep U
        t = float(si.get("t", 1.0))
        x = (np.asarray(result["U_over_t"], float) if "U_over_t" in result
             else np.asarray(result["U_values"], float) / t)
        xlabel, fixed, xkind = r"$U/t$", f"t={t}", "U"

    # show a fixed V in the title on t-/U-scans (V is the axis on a V-scan, so
    # it is not repeated there); only when V != 0 so V=0 plots are unchanged
    v_tag = "" if is_v else (f", V={V:g}" if abs(V) > 1e-12 else "")

    sector = ("full Hamiltonian (all $S_z$)" if not spin_preserving
              else "$S_z$=0 sector")

    fig, ax = plt.subplots(figsize=(9, 6))
    for k, n in enumerate(ns):
        lab = f"n={n}"
        if n == 2:
            lab += " (bipartite)"
        elif n == L:
            lab += " (whole ring, pure)"
        ax.plot(x, np.asarray(C[n], float),
                color=_COLORS[k % len(_COLORS)],
                marker=_MARKERS[k % len(_MARKERS)], ms=5, lw=1.6, label=lab)

    ex = result.get("C_exact_nL")
    if ex is not None:
        ex = np.asarray(ex, float).copy()
        # Drop the pure-formula reference wherever the largest-cluster RDM is
        # MIXED (stored purity < 1) — the pure formula is valid only for a pure
        # RDM. Reads the purity already saved in the PKL, so OLD scans are
        # corrected at plot time without recomputing.
        pur_nmax = (result.get("purity", {}) or {}).get(max(ns))
        if pur_nmax is not None:
            ex = np.where(np.asarray(pur_nmax, float) > 1.0 - 1e-6, ex, np.nan)
    if ex is not None and np.isfinite(ex).any():
        ax.plot(x, ex, color="k", ls="--", lw=1.4,
                label=f"Exact $C_{{GME}}$(n={max(ns)})")

    # n=2 analytic reference: the Wootters concurrence, which for two qubits IS
    # the convex roof exactly. If an older PKL does not store it, fall back to
    # the n=2 C_GME curve (identical by construction).
    # NOTE: the Werner form sqrt(1-sqrt(1-C^2)) is NOT a valid GME for n=2 and
    # is deliberately not plotted — only the convex roof / Wootters is correct.
    if 2 in C:
        w = result.get("C_wootters_n2")
        w = (np.asarray(w, float) if w is not None
             and np.isfinite(np.asarray(w, float)).any()
             else np.asarray(C[2], float))
        ax.plot(x, w, color="k", ls=":", lw=2.2,
                label=r"n=2 Wootters $C$  (= n=2 convex roof)")

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(r"$C_{GME}$  (T=0)", fontsize=13)
    # say WHICH state the curves describe — a ground-state and an
    # excited-singlet figure are otherwise indistinguishable
    _state = ("1st excited singlet" if result.get("excited_singlet")
              else "ground state")
    ax.set_title(f"{xkind}-dependence of GME: L={L}, Ne={Ne}, {fixed}, mode={mode}{v_tag}, "
                 f"T=0, {_state}  [{sector}]\n"
                 f"{xlabel.strip('$')} $\\in$ [{x.min():.3f}, {x.max():.3f}]  "
                 f"({len(x)} points)", fontsize=11)
    ax.grid(alpha=0.3)
    # C_GME lives in [0, 1], but curves sitting exactly ON a bound (n=2 at 0 for
    # excited states, n=L at 1 for a pure whole-ring state) get their markers
    # clipped in half by the axes frame — pad so every curve is fully visible.
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=9, ncol=2, loc="best", framealpha=0.95)
    fig.tight_layout()
    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    if save_png:
        fig.savefig(os.path.splitext(plot_path)[0] + ".png", dpi=140,
                    bbox_inches="tight")
    plt.close(fig)
    return plot_path


def run_t_dependence_gme(
    L: int,
    t_values: np.ndarray,
    U: float,
    V: float = 0.0,
    mode: str = "uniform",
    n_electrons: Optional[int] = None,
    n_min: int = 2,
    output_dir: str = "outputs",
    output_prefix: str = "",
    solver: str = "gcg",
    n_restarts: int = 20,
    max_iter: int = 300,
    tol: float = 1e-9,
    k_max_extra: int = 4,
    verbose: bool = True,
    save_pkl: bool = True,
    use_pkl_cache: bool = False,
    cache_version: str = "v1",
    force_recompute: bool = False,
    spin_preserving: bool = True,
    excited_singlet: bool = False,
    singlet_k: int = 12,
) -> dict:
    """
    Compute C_GME(n) at T = 0 for each value of t (fixing U).

    For n = L (full system) the PRB result shows no t-dependence.
    For n < L a decreasing trend with t is expected.

    The x-axis is t/U so the plot is in terms of the fundamental
    coupling ratio.

    Parameters
    ----------
    t_values : 1-D array of hopping amplitudes to scan
    U        : fixed on-site interaction
    n_min    : minimum body order (default 2)
    use_pkl_cache   : load/save the complete result under thermal_runs/
    cache_version   : version tag included in PKL filename
    force_recompute : ignore an existing PKL and recompute

    Returns
    -------
    dict
        t_values, t_over_U, C_LB {n: array}, purity {n: array},
        system_info, plot_path
    """
    if n_electrons is None:
        n_electrons = L
    # Cluster size cannot exceed the electron count: the n-site RDM trace is
    # ⟨Π_site n̂_site⟩, which is exactly 0 once n > Ne (can't occupy more sites
    # than there are electrons). Cap at min(L, Ne) so doped (Ne<L) runs stop at
    # n=Ne instead of hitting a zero-trace RDM. At half-filling Ne=L → no change.
    n_max = min(L, n_electrons)
    n_values = list(range(n_min, n_max + 1))
    t_values = np.asarray(t_values, dtype=float)
    ts = _run_timestamp()

    cache_path = _tscan_cache_path(
        L=L, U=U, mode=mode, n_electrons=n_electrons,
        n_min=n_min, solver=solver, n_restarts=n_restarts,
        max_iter=max_iter, tol=tol, k_max_extra=k_max_extra,
        t_values=t_values, cache_version=cache_version,
        spin_preserving=spin_preserving,
        excited_singlet=excited_singlet,
        V=V,
    )

    if use_pkl_cache and not force_recompute and cache_path.exists():
        payload = _load_thermal_result_pkl(cache_path)
        result = dict(payload["result"])
        result["loaded_from_cache"] = True
        result["cache_path"] = str(cache_path)
        os.makedirs(output_dir, exist_ok=True)
        prefix = f"{output_prefix}_" if output_prefix else ""
        plot_path = os.path.join(
            output_dir, f"{prefix}task3_tscan_GME_L{L}_U{U}_{_sector_tag(spin_preserving)}_{_grid_tag(np.asarray(result['t_values']), 't')}_{ts}.pdf"
        )
        plot_scan_finalized(result, plot_path, spin_preserving=spin_preserving)
        result["plot_path"] = plot_path
        print(f"\n[cache] loaded t-scan PKL → {cache_path}")
        print(f"[cache] regenerated plot → {plot_path}")
        return result

    print(f"\n{'='*64}")
    print(f"  t-DEPENDENCE GME SCAN (T=0)")
    print(f"{'='*64}")
    print(f"  L={L}  U={U}  Ne={n_electrons}  mode={mode}")
    print(f"  t range [{t_values[0]:.4g}, {t_values[-1]:.4g}]  ({len(t_values)} points)")
    print(f"  n = {n_min}..{n_max}")
    print(f"{'-'*64}")

    C_GME: dict[int, np.ndarray] = {n: np.zeros(len(t_values)) for n in n_values}
    purity_dict: dict[int, np.ndarray] = {n: np.zeros(len(t_values)) for n in n_values}
    entropy_dict: dict[int, np.ndarray] = {n: np.zeros(len(t_values)) for n in n_values}
    rank_dict: dict[int, np.ndarray] = {n: np.zeros(len(t_values), dtype=int) for n in n_values}
    convergence: dict[int, list] = {n: [None] * len(t_values) for n in n_values}
    # which eigenstate was actually used at each grid point (excited-singlet mode);
    # its index moves with the parameter through level crossings, so it is recorded
    singlet_idx: np.ndarray = np.full(len(t_values), -1, dtype=int)
    singlet_energy: np.ndarray = np.full(len(t_values), np.nan)
    C_exact: np.ndarray = np.full(len(t_values), np.nan)
    C_analytical_n2: np.ndarray = np.zeros(len(t_values))
    C_wootters_n2:   np.ndarray = np.zeros(len(t_values))
    tag = "C_GCG" if solver == "gcg" else "C_LB"

    for i, t in enumerate(t_values):
        if verbose:
            print(f"\n  t={t:.4g} (t/U={t/U:.4f}) …", flush=True)

        energies, eigvecs, n_so = compute_all_eigenstates(
            L, t, U, mode, n_electrons, spin_preserving=spin_preserving, V=V,
            # GS branch prints the table here; excited path prints its own via
            # find_first_excited_singlet, so gate to avoid double-printing.
            classify_spin=(verbose and not excited_singlet), classify_k=singlet_k,
        )
        if excited_singlet:
            # Same spin-resolved search as the U-scan: the singlet's index
            # moves with t through level crossings, so re-locate it per point.
            info = find_first_excited_singlet(
                L=L, t=float(t), U=U, V=V, mode=mode,
                n_electrons=n_electrons, k=singlet_k,
                spin_preserving=spin_preserving,   # must match the RDM ops
            )
            E0 = float(info["energy"])
            psi0 = np.asarray(info["psi"])
            singlet_idx[i] = int(info["idx"])
            singlet_energy[i] = E0
            if verbose:
                print(f"    first excited singlet: idx={info['idx']}  "
                      f"E={E0:.8f}", flush=True)
        else:
            E0 = energies[0]
            psi0 = eigvecs[:, 0]

        degen = np.where(np.abs(energies - E0) < 1e-10)[0]
        if len(degen) > 1 and verbose:
            _lbl = "excited singlet" if excited_singlet else "ground state"
            print(f"    [WARN] {len(degen)}-fold degenerate {_lbl}")

        for n in n_values:
            sites = tuple(range(n))
            ops = _precompute_ops(sites, n_so, n_electrons,
                                  spin_preserving=spin_preserving)

            if len(degen) <= 1:
                # <=1: non-degenerate state — use psi0 directly. This also
                # guards the excited-singlet path where E0 comes from a
                # separate diagonalisation: if it fails to match any entry in
                # `energies` (degen empty), fall back to psi0 instead of
                # dividing a zero matrix by 0 (which produced silent NaNs).
                rho_raw = _rdm_from_ops(ops, psi0, n)
            else:
                dim = 2 ** n
                rho_raw = np.zeros((dim, dim), dtype=complex)
                for d_idx in degen:
                    rho_raw += _rdm_from_ops(ops, eigvecs[:, d_idx], n)
                rho_raw /= len(degen)

            tr = np.real(np.trace(rho_raw))
            rho = rho_raw / tr if abs(tr) > 1e-14 else rho_raw

            c_val, pur, _detail = _gme_lower_bound(
                rho, n,
                solver=solver, n_restarts=n_restarts,
                max_iter=max_iter, tol=tol, k_max_extra=k_max_extra,
            )
            C_GME[n][i] = c_val
            purity_dict[n][i] = pur
            # entropy + rank stored alongside purity so the PKL is self-contained
            entropy_dict[n][i] = float(compute_entropy(rho))
            rank_dict[n][i] = int(np.sum(np.real(np.linalg.eigvalsh(rho)) > 1e-10))
            convergence[n][i] = _detail    # full solver detail (standard format)

            if n == n_max and compute_purity(rho) > 1.0 - 1e-6:
                # pure formula is valid only for a pure RDM (same check as
                # _gme_lower_bound); a mixed n_max RDM would give a spurious line
                C_exact[i] = float(compute_C_GME_pure(rho, n_max))

            if n == 2:
                C_analytical_n2[i] = _cgme_analytical_n2(rho)
                C_wootters_n2[i]   = _wootters_concurrence(rho)

        if verbose:
            print("    " + "  ".join(
                f"{tag}(n={n})={C_GME[n][i]:.4f}" for n in n_values
            ))
            if 2 in n_values:
                print(f"    C_analytical(n=2)={C_analytical_n2[i]:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{output_prefix}_" if output_prefix else ""
    plot_path = os.path.join(
        output_dir, f"{prefix}task3_tscan_GME_L{L}_U{U}_{_sector_tag(spin_preserving)}_{_grid_tag(t_values, 't')}_{ts}.pdf"
    )

    result = {
        "t_values":           t_values,
        "t_over_U":           t_values / U,
        "C_GME":              C_GME,
        "C_exact_nL":         C_exact,
        "C_analytical_n2":    C_analytical_n2,
        "C_wootters_n2":      C_wootters_n2,
        "purity":             purity_dict,
        "entropy":            entropy_dict,
        "rank":               rank_dict,
        "convergence":        convergence,   # {n: [solver detail per t] } — standard _gme_lower_bound format
        "n_values":           n_values,
        "solver":             solver,
        "system_info":        dict(L=L, U=U, V=V, mode=mode, Ne=n_electrons),
        # which state each grid point used: ground state, or the first
        # excited singlet (whose index moves through level crossings)
        "excited_singlet": bool(excited_singlet),
        "singlet_k":       int(singlet_k),
        "singlet_idx":     singlet_idx,
        "singlet_energy":  singlet_energy,
        "plot_path":          plot_path,
        "run_timestamp":      ts,
    }

    # Finalized (replot-style) figure, produced automatically from the result
    plot_scan_finalized(result, plot_path, spin_preserving=spin_preserving)
    print(f"\n    Plot → {plot_path}")

    if save_pkl or use_pkl_cache:
        metadata = {
            "analysis_type": "t_dependence",
            "cache_version": cache_version,
            "L": int(L), "U": float(U), "V": float(V),
            "mode": str(mode), "Ne": int(n_electrons),
            "n_min": int(n_min), "n_max": int(n_max),
            "n_values": list(n_values),
            "t_values": t_values.tolist(),
            "t_digest": _array_digest(t_values),
            "solver": solver, "n_restarts": int(n_restarts),
            "max_iter": int(max_iter), "tol": float(tol),
            "k_max_extra": int(k_max_extra),
            "plot_path": plot_path,
            "run_timestamp": ts,
        }
        _save_thermal_result_pkl(
            cache_path=cache_path, metadata=metadata, result=result
        )
        result["pkl_path"] = str(cache_path)
        result["cache_path"] = str(cache_path)
        print(f"    t-scan PKL → {cache_path}")

    return result


# ── t/U × Temperature 3D surface ──────────────────────────────────────────────

def _tT_surface_cache_path(
    *,
    L: int,
    U: float,
    mode: str,
    n_electrons: int,
    n_min: int,
    solver: str,
    n_restarts: int,
    max_iter: int,
    tol: float,
    k_max_extra: int,
    t_values: np.ndarray,
    T_values: np.ndarray,
    spin_preserving: bool,
    cache_version: str,
    n_max: Optional[int] = None,
) -> Path:
    """Deterministic PKL path for one (t, T) surface run."""
    t_arr = np.asarray(t_values, dtype=float)
    T_arr = np.asarray(T_values, dtype=float)
    sp_tag = _sector_tag(spin_preserving)
    nmax_tag = "" if (n_max is None or n_max >= L) else f"_nmax{int(n_max)}"
    filename = (
        f"tTsurface_{cache_version}_"
        f"L{L}_Ne{n_electrons}_U{_safe_float_tag(U)}_"
        f"mode-{mode}_nmin{n_min}{nmax_tag}_solver-{solver}_{sp_tag}_"
        f"restarts{int(n_restarts)}_iter{int(max_iter)}_"
        f"kmax{int(k_max_extra)}_tol{_safe_float_tag(float(tol))}_"
        f"{_grid_tag(t_arr, 't')}-{_array_digest(t_arr)}_"
        f"{_grid_tag(T_arr, 'T')}-{_array_digest(T_arr)}.pkl"
    )
    return _thermal_runs_dir() / filename


def _surface_gme_job_cache_path(
    *,
    run_key: str,
    ti: int,
    Ti: int,
    n: int,
) -> Path:
    """
    Deterministic path for one surface GME job result at grid point
    (t-index ti, T-index Ti, body order n).

    Keyed by run_key + integer grid indices (not the density-matrix hash), so a
    resumed run can tell a t-slice is fully done WITHOUT rebuilding its RDM bank.
    run_key already encodes all run parameters (L, U, grids, solver, sector).
    """
    filename = f"surfgme_{run_key}_ti{int(ti)}_Ti{int(Ti)}_n{int(n)}.pkl"
    return _thermal_job_cache_dir() / filename


def _werner_gme_from_concurrence(C):
    """
    Supervisor's n=2 GME measure for a Werner state: sqrt(1 - sqrt(1 - C^2)),
    where C is the Wootters concurrence (= the GCG convex roof for n=2).
    """
    C = np.clip(np.asarray(C, dtype=float), 0.0, 1.0)
    return np.sqrt(np.clip(1.0 - np.sqrt(np.clip(1.0 - C ** 2, 0.0, 1.0)),
                           0.0, 1.0))


def _surface_n2_werner(C_surface: dict, n_values: list, werner_n2: bool) -> dict:
    """Return a copy of C_surface with the n=2 panel mapped to the Werner GME."""
    if not werner_n2 or 2 not in n_values:
        return C_surface
    out = dict(C_surface)
    out[2] = _werner_gme_from_concurrence(np.asarray(C_surface[2], dtype=float))
    return out


def _make_figure_tT_surface(
    t_values: np.ndarray,
    T_values: np.ndarray,
    U: float,
    C_surface: dict,
    n_values: list,
    L: int,
    mode: str,
    solver: str = "gcg",
    spin_preserving: bool = True,
    werner_n2: bool = False,   # Werner is NOT a valid n=2 GME; show the convex roof
    n_electrons: Optional[int] = None,   # for "Ne=" in the title
) -> plt.Figure:
    """
    3D surface plot C_GME(t/U, T): one 3D panel per body-order n.

    x-axis = t/U, y-axis = Temperature T, z-axis = C_GME. Every panel shows the
    convex roof as stored, n=2 included (for n=2 that equals the Wootters
    concurrence). The legacy Werner mapping is off by default and kept only to
    reproduce old figures.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d proj)
    from matplotlib.ticker import MaxNLocator

    C_surface = _surface_n2_werner(C_surface, n_values, werner_n2)

    tu = np.asarray(t_values, dtype=float) / U
    T_arr = np.asarray(T_values, dtype=float)
    # meshgrid with 'ij' so grids match C_surface[n] shape (len(t), len(T))
    TU, TT = np.meshgrid(tu, T_arr, indexing="ij")

    # publication font sizes
    LBL, TICK, TTL = 15, 11, 16

    n_panels = len(n_values)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig = plt.figure(figsize=(6.8 * ncols, 6.0 * nrows))

    zlabel = r"$C_{\mathrm{GME}}$" if solver == "gcg" else r"$C_{\mathrm{GME}}$ (LB)"
    for idx, n in enumerate(n_values):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
        Z = np.asarray(C_surface[n], dtype=float)
        surf = ax.plot_surface(
            TU, TT, Z,
            cmap="viridis", edgecolor="none",
            antialiased=True, rstride=1, cstride=1,
        )
        ax.set_xlabel(r"$t/U$", fontsize=LBL, labelpad=8)
        ax.set_ylabel(r"Temperature $T$ (K)", fontsize=LBL, labelpad=8)
        ax.set_zlabel(zlabel, fontsize=LBL, labelpad=6)
        _tsuf = (" (bipartite, Werner)" if (n == 2 and werner_n2)
                 else " (bipartite)" if n == 2 else "")
        ax.set_title(f"$n={n}$" + _tsuf, fontsize=TTL, y=1.03)
        # ── thin, non-overlapping ticks (≤5 per axis) + larger tick font ──
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis="both", labelsize=TICK, pad=2)
        ax.set_zlim(bottom=0)
        ax.view_init(elev=26, azim=-118)
        cb = fig.colorbar(surf, ax=ax, shrink=0.58, aspect=12, pad=0.12)
        cb.ax.tick_params(labelsize=TICK)

    sector = _sector_label(spin_preserving)
    ne_tag = f"Ne={n_electrons}, " if n_electrons is not None else ""
    fig.suptitle(
        f"Thermal GME surface  $C_{{\\mathrm{{GME}}}}(t/U,\\,T)$ — "
        f"L={L}, {ne_tag}U={U}, mode={mode}, {sector}\n"
        f"$t/U \\in [{tu[0]:.3g}, {tu[-1]:.3g}]$  $\\times$  "
        f"$T \\in [{T_arr[0]:.3g}, {T_arr[-1]:.3g}]$",
        fontsize=TTL + 1,
    )
    # tight_layout / bbox_inches='tight' both mis-crop 3D axis labels, so use
    # explicit, generous margins and save the figure WITHOUT a tight bbox.
    fig.subplots_adjust(
        left=0.08, right=0.96, top=0.90, bottom=0.10,
        wspace=0.22, hspace=0.34,
    )
    return fig


def _make_figure_tT_heatmap(
    t_values: np.ndarray,
    T_values: np.ndarray,
    U: float,
    C_surface: dict,
    n_values: list,
    L: int,
    mode: str,
    solver: str = "gcg",
    spin_preserving: bool = True,
    werner_n2: bool = False,   # Werner is NOT a valid n=2 GME; show the convex roof
    n_electrons: Optional[int] = None,   # for "Ne=" in the title
) -> plt.Figure:
    """
    2D heatmap of C_GME(t/U, T): one panel per body-order n.
    x-axis = t/U, y-axis = Temperature T, colour = C_GME. Cleaner than the 3D
    surface for print. Every panel shows the convex roof as stored, n=2 included
    (for n=2 that equals the Wootters concurrence). The legacy Werner mapping is
    off by default and kept only to reproduce old figures.
    """
    from matplotlib.ticker import MaxNLocator

    C_surface = _surface_n2_werner(C_surface, n_values, werner_n2)
    tu = np.asarray(t_values, dtype=float) / U
    T_arr = np.asarray(T_values, dtype=float)
    LBL, TICK, TTL = 14, 11, 15

    n_panels = len(n_values)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.3 * nrows),
                             squeeze=False)
    zlabel = r"$C_{\mathrm{GME}}$" if solver == "gcg" else r"$C_{\mathrm{GME}}$ (LB)"

    for idx, n in enumerate(n_values):
        ax = axes[idx // ncols][idx % ncols]
        # C_surface[n] is (len(t), len(T)); pcolormesh wants (len(y), len(x))
        Z = np.asarray(C_surface[n], dtype=float).T
        im = ax.pcolormesh(tu, T_arr, Z, cmap="viridis", shading="gouraud",
                           vmin=0.0)
        ax.set_xlabel(r"$t/U$", fontsize=LBL)
        ax.set_ylabel(r"Temperature $T$ (K)", fontsize=LBL)
        _tsuf = (" (bipartite, Werner)" if (n == 2 and werner_n2)
                 else " (bipartite)" if n == 2 else "")
        ax.set_title(f"$n={n}$" + _tsuf, fontsize=TTL)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(labelsize=TICK)
        cb = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.03)
        cb.set_label(zlabel, fontsize=LBL); cb.ax.tick_params(labelsize=TICK)

    for j in range(n_panels, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    sector = _sector_label(spin_preserving)
    ne_tag = f"Ne={n_electrons}, " if n_electrons is not None else ""
    fig.suptitle(
        f"Thermal GME heatmap  $C_{{\\mathrm{{GME}}}}(t/U,\\,T)$ — "
        f"L={L}, {ne_tag}U={U}, mode={mode}, {sector}",
        fontsize=TTL + 1,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def render_surface_from_pkl(
    pkl_path: str,
    output_dir: Optional[str] = None,
    output_prefix: str = "",
) -> dict:
    """
    Re-render the 3D surface AND the 2D heatmap from a cached tTsurface PKL,
    WITHOUT recomputing. Handy for turning a long cluster run into paper-ready
    figures (or just the 2D heatmap) in seconds.

    Returns {"surface_path": ..., "heatmap_path": ...}.
    """
    payload = _load_thermal_result_pkl(Path(pkl_path))
    result = payload["result"]
    meta = payload.get("meta", {})
    si = result.get("system_info", {})
    L = int(meta.get("L", si.get("L")))
    Ne = int(meta.get("Ne", si.get("Ne", L)))
    U = float(meta.get("U", si.get("U")))
    mode = str(meta.get("mode", si.get("mode", "uniform")))
    sv = result.get("solver", "gcg")
    spv = bool(result.get("spin_preserving", meta.get("spin_preserving", True)))
    tv = np.asarray(result["t_values"], dtype=float)
    Tv = np.asarray(result["T_values"], dtype=float)
    Cs = result["C_surface"]
    nv = list(result["n_values"])

    outdir = output_dir or os.path.dirname(os.path.abspath(pkl_path))
    os.makedirs(outdir, exist_ok=True)
    ts = _run_timestamp()
    prefix = f"{output_prefix}_" if output_prefix else ""
    grids = f"{_grid_tag(tv, 't')}_{_grid_tag(Tv, 'T')}_{ts}"
    base = f"L{L}_U{U}_{_sector_tag(spv)}_{grids}"

    surf_path = os.path.join(outdir, f"{prefix}task3_tT_surface_GME_{base}.pdf")
    heat_path = os.path.join(outdir, f"{prefix}task3_tT_heatmap_GME_{base}.pdf")
    fig = _make_figure_tT_surface(tv, Tv, U, Cs, nv, L, mode,
                                  solver=sv, spin_preserving=spv, n_electrons=Ne)
    # 3D: NO bbox_inches='tight' — it mis-crops mplot3d axis labels (clips the
    # left-column / bottom-row "Temperature T (K)"); rely on the figure margins.
    fig.savefig(surf_path,  dpi=600); plt.close(fig)
    fig2 = _make_figure_tT_heatmap(tv, Tv, U, Cs, nv, L, mode,
                                   solver=sv, spin_preserving=spv, n_electrons=Ne)
    fig2.savefig(heat_path, dpi=600, bbox_inches="tight"); plt.close(fig2)
    return {"surface_path": surf_path, "heatmap_path": heat_path}


def merge_tT_surface_pkls(
    pkl_paths,
    output_dir: Optional[str] = None,
    output_prefix: str = "",
    save_pkl: bool = True,
) -> dict:
    """
    Stitch several per-n (t,T)-surface PKLs — each produced by a single-body-order
    job (``--min-n N --tT-surface-max-n N``) — into ONE combined surface, and
    render the multi-panel 3D + 2D figures over ALL n together. This is the merge
    step for running one SLURM job per n.

    Accepts explicit paths and/or shell globs. Every PKL must share the same L, U,
    sector, and (t, T) grids. Writes a merged PKL (re-usable by
    render_surface_from_pkl / render_surface_html.py). Returns
    {"surface_path", "heatmap_path", "merged_pkl", "n_values"}.
    """
    import glob as _glob

    paths: list[str] = []
    for p in pkl_paths:
        hits = sorted(_glob.glob(p))
        paths.extend(hits if hits else [p])
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        raise FileNotFoundError(f"no surface PKLs matched: {pkl_paths}")

    results = [_load_thermal_result_pkl(Path(p))["result"] for p in paths]
    ref = results[0]
    tv = np.asarray(ref["t_values"], dtype=float)
    Tv = np.asarray(ref["T_values"], dtype=float)
    si = ref.get("system_info", {})
    L = int(si.get("L"))
    Ne = int(si.get("Ne", L))
    U = float(si.get("U"))
    mode = str(si.get("mode", "uniform"))
    solver = ref.get("solver", "gcg")
    spv = bool(ref.get("spin_preserving", True))

    merged_C: dict = {}
    merged_P: dict = {}
    for r, p in zip(results, paths):
        rt = np.asarray(r["t_values"], dtype=float)
        rT = np.asarray(r["T_values"], dtype=float)
        if rt.shape != tv.shape or not np.allclose(rt, tv) \
           or rT.shape != Tv.shape or not np.allclose(rT, Tv):
            raise ValueError(
                f"grid mismatch in {p}: all per-n PKLs must share identical "
                f"t and T grids to be merged.")
        rsi = r.get("system_info", {})
        if int(rsi.get("L", L)) != L or bool(r.get("spin_preserving", spv)) != spv:
            raise ValueError(f"L/sector mismatch in {p}; cannot merge.")
        for n, arr in r["C_surface"].items():
            merged_C[int(n)] = np.asarray(arr, dtype=float)
        for n, arr in r.get("purity_surface", {}).items():
            merged_P[int(n)] = np.asarray(arr, dtype=float)

    n_values = sorted(merged_C)

    outdir = output_dir or os.path.dirname(os.path.abspath(paths[0]))
    os.makedirs(outdir, exist_ok=True)
    ts = _run_timestamp()
    prefix = f"{output_prefix}_" if output_prefix else ""
    grids = f"{_grid_tag(tv, 't')}_{_grid_tag(Tv, 'T')}_{ts}"
    ntag = "-".join(str(n) for n in n_values)
    base = f"L{L}_U{U}_{_sector_tag(spv)}_n{ntag}_{grids}"

    surf_path = os.path.join(outdir, f"{prefix}task3_tT_surface_GME_{base}.pdf")
    heat_path = os.path.join(outdir, f"{prefix}task3_tT_heatmap_GME_{base}.pdf")
    fig = _make_figure_tT_surface(tv, Tv, U, merged_C, n_values, L, mode,
                                  solver=solver, spin_preserving=spv, n_electrons=Ne)
    fig.savefig(surf_path, dpi=600); plt.close(fig)             # no tight (3D labels)
    fig2 = _make_figure_tT_heatmap(tv, Tv, U, merged_C, n_values, L, mode,
                                   solver=solver, spin_preserving=spv, n_electrons=Ne)
    fig2.savefig(heat_path, dpi=600, bbox_inches="tight"); plt.close(fig2)

    out = {"surface_path": surf_path, "heatmap_path": heat_path,
           "n_values": n_values, "merged_pkl": None}

    if save_pkl:
        merged_result = dict(ref)
        merged_result["C_surface"] = merged_C
        merged_result["purity_surface"] = merged_P
        merged_result["n_values"] = n_values
        merged_result["merged_from"] = [os.path.abspath(p) for p in paths]
        merged_pkl = Path(outdir) / f"{prefix}tTsurface_merged_{base}.pkl"
        _save_thermal_result_pkl(
            cache_path=merged_pkl,
            metadata={"L": L, "U": U, "mode": mode, "spin_preserving": spv,
                      "merged_from": [os.path.basename(p) for p in paths]},
            result=merged_result,
        )
        out["merged_pkl"] = str(merged_pkl)
    return out


def run_tT_surface_gme(
    L: int,
    t_values: np.ndarray,
    T_values: np.ndarray,
    U: float,
    mode: str = "uniform",
    n_electrons: Optional[int] = None,
    n_min: int = 2,
    output_dir: str = "outputs",
    output_prefix: str = "",
    solver: str = "gcg",
    n_restarts: int = 20,
    max_iter: int = 300,
    tol: float = 1e-9,
    k_max_extra: int = 4,
    thermal_energy_cutoff: Optional[float] = 10.0,
    verbose: bool = True,
    n_workers: int = 1,
    average_site_clusters: bool = True,
    spin_preserving: bool = True,
    save_pkl: bool = True,
    use_pkl_cache: bool = False,
    cache_version: str = "v1",
    force_recompute: bool = False,
    n_max: Optional[int] = None,
    V: float = 0.0,
    excited_singlet: bool = False,
    singlet_k: int = 0,
) -> dict:
    """
    Compute the 3D thermal-GME surface C_GME(n; t/U, T).

    For each t value the Hubbard Hamiltonian is diagonalised, the thermal
    RDM bank is built over all eigenstates, and the temperature is swept.
    Produces one 3D surface per body-order n with axes:
        x = t/U,  y = Temperature T,  z = C_GME.

    Parameters mirror run_t_dependence_gme / run_thermal_gme_analysis.
    T_values may include T=0 (handled as the degenerate ground-state limit).

    Returns
    -------
    dict
        t_values, T_values, t_over_U, C_surface {n: (n_t, n_T)},
        purity_surface {n: (n_t, n_T)}, n_values, system_info, plot_path
    """
    if n_electrons is None:
        n_electrons = L
    # Cap cluster size at min(L, Ne): an n-site RDM with n > Ne has zero trace.
    n_max = min(L, n_electrons) if n_max is None else min(int(n_max), L, n_electrons)
    n_values = list(range(n_min, n_max + 1))
    t_values = np.asarray(t_values, dtype=float)
    T_values = np.asarray(T_values, dtype=float)
    ts = _run_timestamp()

    cache_path = _tT_surface_cache_path(
        L=L, U=U, mode=mode, n_electrons=n_electrons,
        n_min=n_min, solver=solver, n_restarts=n_restarts,
        max_iter=max_iter, tol=tol, k_max_extra=k_max_extra,
        t_values=t_values, T_values=T_values,
        spin_preserving=spin_preserving, cache_version=cache_version,
        n_max=n_max,
    )

    def _regenerate_plot(result: dict) -> str:
        os.makedirs(output_dir, exist_ok=True)
        prefix = f"{output_prefix}_" if output_prefix else ""
        sp_tag = _sector_tag(spin_preserving)
        grids = f"{_grid_tag(t_values, 't')}_{_grid_tag(T_values, 'T')}_{ts}"
        tv = np.asarray(result["t_values"]); Tv = np.asarray(result["T_values"])
        sv = result.get("solver", solver)
        spv = result.get("spin_preserving", spin_preserving)
        # 3D surface
        surf_path = os.path.join(
            output_dir, f"{prefix}task3_tT_surface_GME_L{L}_U{U}_{sp_tag}_{grids}.pdf")
        fig = _make_figure_tT_surface(tv, Tv, U, result["C_surface"],
                                      result["n_values"], L, mode,
                                      solver=sv, spin_preserving=spv,
                                      n_electrons=n_electrons)
        # no bbox_inches='tight' — it clips 3D axis labels; rely on the margins
        fig.savefig(surf_path, dpi=600); plt.close(fig)
        # 2D heatmap (print-friendly)
        heat_path = os.path.join(
            output_dir, f"{prefix}task3_tT_heatmap_GME_L{L}_U{U}_{sp_tag}_{grids}.pdf")
        fig2 = _make_figure_tT_heatmap(tv, Tv, U, result["C_surface"],
                                       result["n_values"], L, mode,
                                       solver=sv, spin_preserving=spv,
                                       n_electrons=n_electrons)
        fig2.savefig(heat_path, dpi=600, bbox_inches="tight"); plt.close(fig2)
        result["heatmap_path"] = heat_path
        return surf_path

    if use_pkl_cache and not force_recompute and cache_path.exists():
        payload = _load_thermal_result_pkl(cache_path)
        result = dict(payload["result"])
        result["loaded_from_cache"] = True
        result["cache_path"] = str(cache_path)
        plot_path = _regenerate_plot(result)
        result["plot_path"] = plot_path
        print(f"\n[cache] loaded (t,T)-surface PKL → {cache_path}")
        print(f"[cache] regenerated 3D plot → {plot_path}")
        return result

    sector = "Sz=0 sector" if spin_preserving else "full Hamiltonian (all Sz)"
    print(f"\n{'='*64}")
    print(f"  (t/U, T) 3D SURFACE GME SCAN")
    print(f"{'='*64}")
    print(f"  L={L}  U={U}  Ne={n_electrons}  mode={mode}  [{sector}]")
    print(f"  t range [{t_values[0]:.4g}, {t_values[-1]:.4g}]  ({len(t_values)} points)")
    print(f"  T range [{T_values[0]:.4g}, {T_values[-1]:.4g}]  ({len(T_values)} points)")
    print(f"  n = {n_min}..{n_max}   solver={solver}")
    print(f"{'-'*64}")

    n_t, n_T = len(t_values), len(T_values)
    C_surface: dict[int, np.ndarray] = {n: np.zeros((n_t, n_T)) for n in n_values}
    purity_surface: dict[int, np.ndarray] = {n: np.zeros((n_t, n_T)) for n in n_values}
    # entropy / rank / solver-convergence stored on the same (n_t, n_T) grid, so
    # the surface PKL is as self-contained as the t-scan and U-scan PKLs
    entropy_surface: dict[int, np.ndarray] = {
        n: np.full((n_t, n_T), np.nan) for n in n_values}
    rank_surface: dict[int, np.ndarray] = {
        n: np.full((n_t, n_T), -1, dtype=int) for n in n_values}
    convergence_surface: dict[int, list] = {
        n: [[None] * n_T for _ in range(n_t)] for n in n_values}
    tag = "C_GCG" if solver == "gcg" else "C_LB"

    # Per-job checkpointing: each (ti, Ti, n) GME result is written to its own
    # PKL keyed on run_key + grid indices. A resubmitted run (e.g. after a
    # walltime kill) reloads finished grid points and — when a whole t-slice is
    # already done — skips its diagonalization + RDM-bank build entirely.
    run_key = hashlib.sha256(str(cache_path.name).encode("utf-8")).hexdigest()[:16]
    do_cache = bool(use_pkl_cache)
    n_reused, n_computed = 0, 0

    def _persist_surface_job(ti_, Ti_, n_, c_val, pur, elapsed, detail=None,
                             ent=float("nan"), rank=-1):
        if not (do_cache and save_pkl):
            return
        p = _surface_gme_job_cache_path(run_key=run_key, ti=ti_, Ti=Ti_, n=n_)
        result_full, decompositions = _gme_job_detail_payload(
            detail, solver=solver, n=n_, purity=pur,
        )
        payload = {
            "meta": {
                "object_type": "hubbard_surface_gme_job",
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "run_key": run_key,
                "ti": int(ti_), "Ti": int(Ti_), "n": int(n_),
                "t": float(t_values[ti_]), "T": float(T_values[Ti_]),
                "solver": solver,
                "n_restarts": int(n_restarts),
                "max_iter": int(max_iter) if max_iter is not None else None,
                "tol": float(tol) if tol is not None else None,
                "k_max_extra": int(k_max_extra),
                "elapsed_seconds": float(elapsed),
                "spin_preserving": bool(spin_preserving),
                "sector": _sector_label(spin_preserving),
            },
            "result": {
                "C_value": float(c_val), "purity": float(pur),
                # entropy + rank stored per job so a cache reload restores the
                # full metric set without touching the RDM again
                "entropy": float(ent), "rank": int(rank),
                "elapsed_seconds": float(elapsed),
                "spin_preserving": bool(spin_preserving),
            },
            "result_full": result_full,
            "decompositions": decompositions,
        }
        try:
            _save_thermal_gme_job_pkl(cache_path=p, payload=payload)
        except Exception as exc:
            print(f"      [cache] WARN could not save surface job: {exc!r}", flush=True)

    for ti, t in enumerate(t_values):
        # 1. Reload any (Ti, n) jobs for this t-slice already in the per-job cache.
        missing: list[tuple[int, int]] = []
        n_slice_reused = 0
        for Ti in range(n_T):
            for n in n_values:
                cached = None
                if do_cache and not force_recompute:
                    cached = _load_thermal_gme_job_pkl(
                        _surface_gme_job_cache_path(
                            run_key=run_key, ti=ti, Ti=Ti, n=n)
                    )
                if cached is not None:
                    C_surface[n][ti, Ti] = float(cached["result"]["C_value"])
                    purity_surface[n][ti, Ti] = float(cached["result"]["purity"])
                    # older job caches predate entropy/rank — fall back to NaN/-1
                    _res = cached["result"]
                    entropy_surface[n][ti, Ti] = float(
                        _res.get("entropy", float("nan")))
                    rank_surface[n][ti, Ti] = int(_res.get("rank", -1))
                    convergence_surface[n][ti][Ti] = cached.get("result_full")
                    n_slice_reused += 1
                else:
                    missing.append((Ti, n))
        n_reused += n_slice_reused

        # 2. Whole slice done → skip the expensive diagonalization + RDM bank.
        if not missing:
            if verbose:
                print(f"\n  [{ti+1}/{n_t}] t={t:.4g} (t/U={t/U:.4f}) … "
                      f"all {n_T*len(n_values)} jobs cached, skipped", flush=True)
            continue

        if verbose:
            print(f"\n  [{ti+1}/{n_t}] t={t:.4g} (t/U={t/U:.4f}) … "
                  f"{n_slice_reused} cached, {len(missing)} to compute", flush=True)

        # 3. RDM bank for THIS t — build or load from the shared generic cache
        #    (each t-value is a distinct Hamiltonian → its own cached bank).
        bank_info = build_or_load_rdm_bank(
            L=L, t=t, U=U, mode=mode, n_electrons=n_electrons, n_min=n_min,
            average_site_clusters=average_site_clusters,
            n_workers=n_workers, spin_preserving=spin_preserving,
            use_pkl_cache=use_pkl_cache, cache_version=cache_version,
            force_recompute=force_recompute, save_pkl=save_pkl, verbose=False,
            n_max=n_max,
        )
        energies = bank_info["energies"]
        bank = bank_info["bank"]
        E0 = energies[0]

        # Boltzmann weights depend only on T (Ti); compute once per needed Ti.
        active_by_Ti: dict[int, tuple] = {}
        for Ti in sorted({Ti_ for Ti_, _ in missing}):
            T = float(T_values[Ti])
            if T <= 1e-14:
                active = np.where(np.abs(energies - E0) < 1e-10)[0]
                probs_active = np.full(len(active), 1.0 / len(active))
            else:
                active, probs_active = _thermal_active_probabilities(
                    energies, T, E0, thermal_energy_cutoff
                )
            active_by_Ti[Ti] = (active, probs_active)

        # 4. Build ONLY the missing GME jobs for this t-slice.
        jobs = []
        # entropy/rank come straight from the RDM, so compute them here where
        # Gamma_T exists rather than shipping the matrix back from the worker
        metrics_by_job: dict[tuple[int, int], tuple[float, int]] = {}
        for (Ti, n) in missing:
            active, probs_active = active_by_Ti[Ti]
            Gamma_T = np.einsum("m,mij->ij", probs_active, bank[n][active])
            Gamma_T = normalize_density_matrix(Gamma_T)
            metrics_by_job[(Ti, n)] = (
                float(compute_entropy(Gamma_T)),
                int(np.sum(np.real(np.linalg.eigvalsh(Gamma_T)) > 1e-10)),
            )
            jobs.append((
                Ti, float(T_values[Ti]), n, Gamma_T,
                solver, n_restarts, max_iter, tol, k_max_extra,
                13579 + 1000 * ti + 100 * Ti + n, "no_cache", "no_hash",
            ))

        # 5. Dispatch, checkpointing each result as it finishes.
        n_w = max(1, int(n_workers or 1))
        if n_w > 1 and len(jobs) > 1:
            try:
                from tqdm import tqdm as _tqdm
            except ImportError:
                def _tqdm(it, **kw):
                    return it
            import multiprocessing as mp
            with mp.Pool(processes=n_w) as pool:
                for Ti, T, n, c_val, pur, elapsed, _jp, _gh, detail in _tqdm(
                    pool.imap_unordered(_thermal_gme_worker, jobs),
                    total=len(jobs), desc=f"    t={t:.3g} GME",
                    unit="job", ncols=90,
                ):
                    C_surface[n][ti, Ti] = c_val
                    purity_surface[n][ti, Ti] = pur
                    _ent, _rk = metrics_by_job[(Ti, n)]
                    entropy_surface[n][ti, Ti] = _ent
                    rank_surface[n][ti, Ti] = _rk
                    convergence_surface[n][ti][Ti] = detail
                    _persist_surface_job(ti, Ti, n, c_val, pur, elapsed, detail,
                                         ent=_ent, rank=_rk)
                    n_computed += 1
        else:
            for job in jobs:
                Ti, T, n, c_val, pur, elapsed, _jp, _gh, detail = _thermal_gme_worker(job)
                C_surface[n][ti, Ti] = c_val
                purity_surface[n][ti, Ti] = pur
                _ent, _rk = metrics_by_job[(Ti, n)]
                entropy_surface[n][ti, Ti] = _ent
                rank_surface[n][ti, Ti] = _rk
                convergence_surface[n][ti][Ti] = detail
                _persist_surface_job(ti, Ti, n, c_val, pur, elapsed, detail,
                                     ent=_ent, rank=_rk)
                n_computed += 1

        if verbose:
            lo, hi = T_values[0], T_values[-1]
            print(
                "    " + "  ".join(
                    f"{tag}(n={n}): T={lo:.2g}→{C_surface[n][ti,0]:.3f}, "
                    f"T={hi:.2g}→{C_surface[n][ti,-1]:.3f}"
                    for n in n_values
                ),
                flush=True,
            )

    if verbose and do_cache:
        print(f"\n  GME jobs: {n_computed} computed, {n_reused} reused from cache",
              flush=True)

    # Plot
    result = {
        "t_values":        t_values,
        "T_values":        T_values,
        "t_over_U":        t_values / U,
        "C_surface":       C_surface,
        "purity_surface":  purity_surface,
        # same self-contained metric set the t-scan / U-scan PKLs carry
        "entropy_surface": entropy_surface,   # {n: (n_t, n_T)}
        "rank_surface":    rank_surface,      # {n: (n_t, n_T)}
        "convergence":     convergence_surface,  # {n: [[detail per T] per t]} — standard _gme_lower_bound format
        "n_values":        n_values,
        "solver":          solver,
        "spin_preserving": bool(spin_preserving),
        "system_info":     dict(L=L, U=U,V=V, mode=mode, Ne=n_electrons),
        "run_timestamp":   ts,
        "loaded_from_cache": False,
        "cache_path":      None,
    }
    plot_path = _regenerate_plot(result)
    result["plot_path"] = plot_path
    print(f"\n    3D surface plot → {plot_path}")

    if save_pkl or use_pkl_cache:
        metadata = {
            "analysis_type": "tT_surface",
            "cache_version": cache_version,
            "L": int(L), "U": float(U), "V": float(V),
            "mode": str(mode), "Ne": int(n_electrons),
            "n_min": int(n_min), "n_max": int(n_max),
            "n_values": list(n_values),
            "t_values": t_values.tolist(),
            "T_values": T_values.tolist(),
            "t_digest": _array_digest(t_values),
            "T_digest": _array_digest(T_values),
            "spin_preserving": bool(spin_preserving),
            "solver": solver, "n_restarts": int(n_restarts),
            "max_iter": int(max_iter), "tol": float(tol),
            "k_max_extra": int(k_max_extra),
            "thermal_energy_cutoff": thermal_energy_cutoff,
            "plot_path": plot_path,
            "run_timestamp": ts,
        }
        _save_thermal_result_pkl(
            cache_path=cache_path, metadata=metadata, result=result
        )
        result["pkl_path"] = str(cache_path)
        result["cache_path"] = str(cache_path)
        print(f"    (t,T)-surface PKL → {cache_path}")

    return result


# ── U-dependence scan ─────────────────────────────────────────────────────────

def _uscan_cache_path(
    *,
    L: int,
    t: float,
    mode: str,
    n_electrons: int,
    n_min: int,
    solver: str,
    n_restarts: int,
    max_iter: int,
    tol: float,
    k_max_extra: int,
    U_values: np.ndarray,
    cache_version: str,
    spin_preserving: bool = True,
    excited_singlet: bool = False,
    V: float = 0.0,
) -> Path:
    """Deterministic PKL path for one U-scan run."""
    U_arr = np.asarray(U_values, dtype=float)
    filename = (
        f"uscan_{cache_version}_"
        f"L{L}_Ne{n_electrons}_t{_safe_float_tag(t)}_"
        f"mode-{mode}_nmin{n_min}_solver-{solver}_"
        f"{_sector_tag(spin_preserving)}"
        # tags added ONLY when set, so existing ground-state / V=0 PKLs keep
        # their names and stay cache-reusable
        f"{'_es' if excited_singlet else ''}"
        f"{('_V' + _safe_float_tag(float(V))) if abs(V) > 1e-12 else ''}_"
        f"restarts{int(n_restarts)}_iter{int(max_iter)}_"
        f"kmax{int(k_max_extra)}_tol{_safe_float_tag(float(tol))}_"
        f"{_grid_tag(U_arr, 'U')}-{_array_digest(U_arr)}.pkl"
    )
    return _thermal_runs_dir() / filename


def _vscan_cache_path(
    *,
    L: int,
    U: float,
    t: float,
    mode: str,
    n_electrons: int,
    n_min: int,
    solver: str,
    n_restarts: int,
    max_iter: int,
    tol: float,
    k_max_extra: int,
    V_values: np.ndarray,
    cache_version: str,
    spin_preserving: bool = True,
    excited_singlet: bool = False,
) -> Path:
    """Deterministic PKL path for one V-scan run (fixed U and t; V swept)."""
    V_arr = np.asarray(V_values, dtype=float)
    filename = (
        f"vscan_{cache_version}_"
        f"L{L}_Ne{n_electrons}_U{_safe_float_tag(U)}_t{_safe_float_tag(t)}_"
        f"mode-{mode}_nmin{n_min}_solver-{solver}_"
        f"{_sector_tag(spin_preserving)}"
        f"{'_es' if excited_singlet else ''}_"
        f"restarts{int(n_restarts)}_iter{int(max_iter)}_"
        f"kmax{int(k_max_extra)}_tol{_safe_float_tag(float(tol))}_"
        f"{_grid_tag(V_arr, 'V')}-{_array_digest(V_arr)}.pkl"
    )
    return _thermal_runs_dir() / filename


def _make_figure_U(
    U_values: np.ndarray,
    t: float,
    C_LB: dict,
    n_values: list,
    L: int,
    mode: str,
    solver: str = "gcg",
    spin_preserving: bool = True,
) -> plt.Figure:
    """Single-panel figure: C_GME vs U/t at T=0."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ut = U_values / t
    ylabel = r"$C_{GME}$ (T=0)" if solver == "gcg" else r"$C_{GME}$ lower bound (T=0)"

    for i, n in enumerate(n_values):
        ax.plot(ut, C_LB[n],
                color=_COLORS[i % len(_COLORS)],
                marker=_MARKERS[i % len(_MARKERS)],
                ms=4, label=f"n={n}", lw=2)

    ax.set_xlabel("$U/t$", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(
        f"U-dependence of GME: L={L}, t={t}, mode={mode}, T=0  "
        f"[{_sector_label(spin_preserving)}]\n"
        f"U/t ∈ [{float(U_values[0]/t):.3g}, {float(U_values[-1]/t):.3g}]  ({len(U_values)} points)",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def run_U_dependence_gme(
    L: int,
    U_values: np.ndarray,
    t: float,
    V: float = 0.0,
    mode: str = "uniform",
    n_electrons: Optional[int] = None,
    n_min: int = 2,
    output_dir: str = "outputs",
    output_prefix: str = "",
    solver: str = "gcg",
    n_restarts: int = 20,
    max_iter: int = 300,
    tol: float = 1e-9,
    k_max_extra: int = 4,
    verbose: bool = True,
    save_pkl: bool = True,
    use_pkl_cache: bool = False,
    cache_version: str = "v1",
    force_recompute: bool = False,
    spin_preserving: bool = True,
    excited_singlet: bool = False,
    singlet_k: int = 12,
) -> dict:
    """
    Compute C_GME(n) at T=0 for each value of U (fixing t).

    The x-axis is U/t so the plot shows the interaction-to-hopping ratio.
    For n=L the PRB result shows no U-dependence (global entanglement is
    fixed by the many-body state structure).
    For n < L a U-dependence is expected.

    Parameters
    ----------
    U_values : 1-D array of on-site interactions to scan
    t        : fixed hopping amplitude
    n_min    : minimum body order (default 2)
    use_pkl_cache   : load/save the complete result under thermal_runs/
    cache_version   : version tag included in PKL filename
    force_recompute : ignore an existing PKL and recompute

    Returns
    -------
    dict
        U_values, U_over_t, C_GME {n: array}, purity {n: array},
        system_info, plot_path
    """
    if n_electrons is None:
        n_electrons = L
    # Cluster size cannot exceed the electron count: the n-site RDM trace is
    # ⟨Π_site n̂_site⟩, which is exactly 0 once n > Ne (can't occupy more sites
    # than there are electrons). Cap at min(L, Ne) so doped (Ne<L) runs stop at
    # n=Ne instead of hitting a zero-trace RDM. At half-filling Ne=L → no change.
    n_max = min(L, n_electrons)
    n_values = list(range(n_min, n_max + 1))
    U_values = np.asarray(U_values, dtype=float)
    ts = _run_timestamp()

    cache_path = _uscan_cache_path(
        L=L, t=t, mode=mode, n_electrons=n_electrons,
        n_min=n_min, solver=solver, n_restarts=n_restarts,
        max_iter=max_iter, tol=tol, k_max_extra=k_max_extra,
        U_values=U_values, cache_version=cache_version,
        spin_preserving=spin_preserving,
        excited_singlet=excited_singlet,
        V=V,
    )

    if use_pkl_cache and not force_recompute and cache_path.exists():
        payload = _load_thermal_result_pkl(cache_path)
        result = dict(payload["result"])
        result["loaded_from_cache"] = True
        result["cache_path"] = str(cache_path)
        os.makedirs(output_dir, exist_ok=True)
        prefix = f"{output_prefix}_" if output_prefix else ""
        plot_path = os.path.join(
            output_dir,
            f"{prefix}task4_uscan_GME_L{L}_t{t}_{_sector_tag(spin_preserving)}_{_grid_tag(np.asarray(result['U_values']), 'U')}_{ts}.pdf"
        )
        plot_scan_finalized(result, plot_path, spin_preserving=spin_preserving)
        result["plot_path"] = plot_path
        print(f"\n[cache] loaded U-scan PKL → {cache_path}")
        print(f"[cache] regenerated plot → {plot_path}")
        return result

    print(f"\n{'='*64}")
    print(f"  U-DEPENDENCE GME SCAN (T=0)")
    print(f"{'='*64}")
    print(f"  L={L}  t={t}  Ne={n_electrons}  mode={mode}")
    print(f"  U range [{U_values[0]:.4g}, {U_values[-1]:.4g}]  ({len(U_values)} points)")
    print(f"  n = {n_min}..{n_max}")
    print(f"{'-'*64}")

    C_GME: dict[int, np.ndarray] = {n: np.zeros(len(U_values)) for n in n_values}
    purity_dict: dict[int, np.ndarray] = {n: np.zeros(len(U_values)) for n in n_values}
    entropy_dict: dict[int, np.ndarray] = {n: np.zeros(len(U_values)) for n in n_values}
    rank_dict: dict[int, np.ndarray] = {n: np.zeros(len(U_values), dtype=int) for n in n_values}
    convergence: dict[int, list] = {n: [None] * len(U_values) for n in n_values}
    # which eigenstate was actually used at each grid point (excited-singlet mode);
    # its index moves with the parameter through level crossings, so it is recorded
    singlet_idx: np.ndarray = np.full(len(U_values), -1, dtype=int)
    singlet_energy: np.ndarray = np.full(len(U_values), np.nan)
    C_exact: np.ndarray = np.full(len(U_values), np.nan)
    C_wootters_n2: np.ndarray = np.full(len(U_values), np.nan)
    C_analytical_n2: np.ndarray = np.full(len(U_values), np.nan)
    tag = "C_GCG" if solver == "gcg" else "C_LB"

    for i, U in enumerate(U_values):
        if verbose:
            print(f"\n  U={U:.4g} (U/t={U/t:.4f}) …", flush=True)

        energies, eigvecs, n_so = compute_all_eigenstates(
            L, t, U, mode, n_electrons, spin_preserving=spin_preserving, V=V,
            classify_spin=(verbose and not excited_singlet), classify_k=singlet_k,
        )
        if excited_singlet:
            # Re-use the existing spin-resolved search. The index of the first
            # excited singlet is NOT fixed — level crossings move it as U
            # varies — so it must be located at every grid point.
            info = find_first_excited_singlet(
                L=L, t=t, U=float(U),V=V, mode=mode,
                n_electrons=n_electrons, k=singlet_k,
                spin_preserving=spin_preserving,   # must match the RDM ops
            )
            E0 = float(info["energy"])
            psi0 = np.asarray(info["psi"])
            singlet_idx[i] = int(info["idx"])
            singlet_energy[i] = E0
            if verbose:
                print(f"    first excited singlet: idx={info['idx']}  "
                      f"E={E0:.8f}", flush=True)
        else:
            E0 = energies[0]
            psi0 = eigvecs[:, 0]

        degen = np.where(np.abs(energies - E0) < 1e-10)[0]
        if len(degen) > 1 and verbose:
            _lbl = "excited singlet" if excited_singlet else "ground state"
            print(f"    [WARN] {len(degen)}-fold degenerate {_lbl}")

        for n in n_values:
            sites = tuple(range(n))
            ops = _precompute_ops(sites, n_so, n_electrons,
                                  spin_preserving=spin_preserving)

            if len(degen) <= 1:
                # <=1: non-degenerate state — use psi0 directly. This also
                # guards the excited-singlet path where E0 comes from a
                # separate diagonalisation: if it fails to match any entry in
                # `energies` (degen empty), fall back to psi0 instead of
                # dividing a zero matrix by 0 (which produced silent NaNs).
                rho_raw = _rdm_from_ops(ops, psi0, n)
            else:
                dim = 2 ** n
                rho_raw = np.zeros((dim, dim), dtype=complex)
                for d_idx in degen:
                    rho_raw += _rdm_from_ops(ops, eigvecs[:, d_idx], n)
                rho_raw /= len(degen)

            tr = np.real(np.trace(rho_raw))
            rho = rho_raw / tr if abs(tr) > 1e-14 else rho_raw

            c_val, pur, _detail = _gme_lower_bound(
                rho, n,
                solver=solver, n_restarts=n_restarts,
                max_iter=max_iter, tol=tol, k_max_extra=k_max_extra,
            )
            C_GME[n][i] = c_val
            purity_dict[n][i] = pur
            # entropy + rank stored alongside purity so the PKL is self-contained
            entropy_dict[n][i] = float(compute_entropy(rho))
            rank_dict[n][i] = int(np.sum(np.real(np.linalg.eigvalsh(rho)) > 1e-10))
            convergence[n][i] = _detail    # full solver detail (standard format)

            if n == n_max and compute_purity(rho) > 1.0 - 1e-6:
                # pure formula is valid only for a pure RDM (same check as
                # _gme_lower_bound); a mixed n_max RDM would give a spurious line
                C_exact[i] = float(compute_C_GME_pure(rho, n_max))

            if n == 2:                       # analytic bipartite references
                C_analytical_n2[i] = _cgme_analytical_n2(rho)
                C_wootters_n2[i]   = _wootters_concurrence(rho)

        if verbose:
            print("    " + "  ".join(
                f"{tag}(n={n})={C_GME[n][i]:.4f}" for n in n_values
            ))

    # Plot
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{output_prefix}_" if output_prefix else ""
    plot_path = os.path.join(
        output_dir, f"{prefix}task4_uscan_GME_L{L}_t{t}_{_sector_tag(spin_preserving)}_{_grid_tag(U_values, 'U')}_{ts}.pdf"
    )

    result = {
        "U_values":      U_values,
        "U_over_t":      U_values / t,
        "C_GME":         C_GME,
        "C_exact_nL":    C_exact,
        "C_wootters_n2": C_wootters_n2,
        "C_analytical_n2": C_analytical_n2,
        "purity":        purity_dict,
        "entropy":       entropy_dict,
        "rank":          rank_dict,
        "convergence":   convergence,   # {n: [solver detail per U]} — standard _gme_lower_bound format
        "n_values":      n_values,
        "solver":        solver,
        "system_info":   dict(L=L, t=t,V=V, mode=mode, Ne=n_electrons),
        # which state each grid point used: ground state, or the first
        # excited singlet (whose index moves through level crossings)
        "excited_singlet": bool(excited_singlet),
        "singlet_k":       int(singlet_k),
        "singlet_idx":     singlet_idx,
        "singlet_energy":  singlet_energy,
        "plot_path":     plot_path,
        "run_timestamp": ts,
    }

    # Finalized (replot-style) figure, produced automatically from the result
    plot_scan_finalized(result, plot_path, spin_preserving=spin_preserving)
    print(f"\n    Plot → {plot_path}")

    if save_pkl or use_pkl_cache:
        metadata = {
            "analysis_type": "U_dependence",
            "cache_version": cache_version,
            "L": int(L), "t": float(t), "V": float(V),
            "mode": str(mode), "Ne": int(n_electrons),
            "n_min": int(n_min), "n_max": int(n_max),
            "n_values": list(n_values),
            "U_values": U_values.tolist(),
            "U_digest": _array_digest(U_values),
            "solver": solver, "n_restarts": int(n_restarts),
            "max_iter": int(max_iter), "tol": float(tol),
            "k_max_extra": int(k_max_extra),
            "plot_path": plot_path,
            "run_timestamp": ts,
        }
        _save_thermal_result_pkl(
            cache_path=cache_path, metadata=metadata, result=result
        )
        result["pkl_path"] = str(cache_path)
        result["cache_path"] = str(cache_path)
        print(f"    U-scan PKL → {cache_path}")

    return result


def run_N_dependence_gme(
    L: int,
    V_values: np.ndarray,
    t: float,
    U: float,
    mode: str = "uniform",
    n_electrons: Optional[int] = None,
    n_min: int = 2,
    output_dir: str = "outputs",
    output_prefix: str = "",
    solver: str = "gcg",
    n_restarts: int = 20,
    max_iter: int = 300,
    tol: float = 1e-9,
    k_max_extra: int = 4,
    verbose: bool = True,
    save_pkl: bool = True,
    use_pkl_cache: bool = False,
    cache_version: str = "v1",
    force_recompute: bool = False,
    spin_preserving: bool = True,
    excited_singlet: bool = False,
    singlet_k: int = 12,
) -> dict:
    """
    Compute C_GME(n) at T=0 for each value of the nearest-neighbour interaction
    V (fixing t and U) — the extended-Hubbard V-scan.

    The x-axis is V/t. As V grows past ~U/2 the ring crosses from the Mott/SDW
    regime into a charge-density-wave (CDW) state, and short-range C_GME
    collapses — the transition shows up directly as a curve.

    Structurally identical to ``run_U_dependence_gme`` with U and V swapped:
    here U is fixed and V is swept.

    Returns
    -------
    dict
        V_values, V_over_t, C_GME {n: array}, purity {n: array},
        system_info, plot_path
    """
    if n_electrons is None:
        n_electrons = L
    # Cluster size cannot exceed the electron count: the n-site RDM trace is
    # ⟨Π_site n̂_site⟩, which is exactly 0 once n > Ne (can't occupy more sites
    # than there are electrons). Cap at min(L, Ne) so doped (Ne<L) runs stop at
    # n=Ne instead of hitting a zero-trace RDM. At half-filling Ne=L → no change.
    n_max = min(L, n_electrons)
    n_values = list(range(n_min, n_max + 1))
    V_values = np.asarray(V_values, dtype=float)
    ts = _run_timestamp()

    cache_path = _vscan_cache_path(
        L=L, U=U, t=t, mode=mode, n_electrons=n_electrons,
        n_min=n_min, solver=solver, n_restarts=n_restarts,
        max_iter=max_iter, tol=tol, k_max_extra=k_max_extra,
        V_values=V_values, cache_version=cache_version,
        spin_preserving=spin_preserving, excited_singlet=excited_singlet,
    )

    if use_pkl_cache and not force_recompute and cache_path.exists():
        payload = _load_thermal_result_pkl(cache_path)
        result = dict(payload["result"])
        result["loaded_from_cache"] = True
        result["cache_path"] = str(cache_path)
        os.makedirs(output_dir, exist_ok=True)
        prefix = f"{output_prefix}_" if output_prefix else ""
        plot_path = os.path.join(
            output_dir,
            f"{prefix}task4_vscan_GME_L{L}_U{U}_t{t}_{_sector_tag(spin_preserving)}_{_grid_tag(np.asarray(result['V_values']), 'V')}_{ts}.pdf"
        )
        plot_scan_finalized(result, plot_path, spin_preserving=spin_preserving)
        result["plot_path"] = plot_path
        print(f"\n[cache] loaded V-scan PKL → {cache_path}")
        print(f"[cache] regenerated plot → {plot_path}")
        return result

    print(f"\n{'='*64}")
    print(f"  V-DEPENDENCE GME SCAN (T=0)")
    print(f"{'='*64}")
    print(f"  L={L}  U={U}  t={t}  Ne={n_electrons}  mode={mode}")
    print(f"  V range [{V_values[0]:.4g}, {V_values[-1]:.4g}]  ({len(V_values)} points)")
    print(f"  n = {n_min}..{n_max}")
    print(f"{'-'*64}")

    C_GME: dict[int, np.ndarray] = {n: np.zeros(len(V_values)) for n in n_values}
    purity_dict: dict[int, np.ndarray] = {n: np.zeros(len(V_values)) for n in n_values}
    entropy_dict: dict[int, np.ndarray] = {n: np.zeros(len(V_values)) for n in n_values}
    rank_dict: dict[int, np.ndarray] = {n: np.zeros(len(V_values), dtype=int) for n in n_values}
    convergence: dict[int, list] = {n: [None] * len(V_values) for n in n_values}
    singlet_idx: np.ndarray = np.full(len(V_values), -1, dtype=int)
    singlet_energy: np.ndarray = np.full(len(V_values), np.nan)
    C_exact: np.ndarray = np.full(len(V_values), np.nan)
    C_wootters_n2: np.ndarray = np.full(len(V_values), np.nan)
    C_analytical_n2: np.ndarray = np.full(len(V_values), np.nan)
    tag = "C_GCG" if solver == "gcg" else "C_LB"

    for i, V in enumerate(V_values):
        if verbose:
            print(f"\n  V={V:.4g} (V/t={V/t:.4f}) …", flush=True)

        energies, eigvecs, n_so = compute_all_eigenstates(
            L, t, U, mode, n_electrons, spin_preserving=spin_preserving, V=V,
            classify_spin=(verbose and not excited_singlet), classify_k=singlet_k,
        )
        if excited_singlet:
            # The first-excited-singlet index moves with V through level
            # crossings, so re-locate it at every grid point.
            info = find_first_excited_singlet(
                L=L, t=t, U=float(U), V=V, mode=mode,
                n_electrons=n_electrons, k=singlet_k,
                spin_preserving=spin_preserving,
            )
            E0 = float(info["energy"])
            psi0 = np.asarray(info["psi"])
            singlet_idx[i] = int(info["idx"])
            singlet_energy[i] = E0
            if verbose:
                print(f"    first excited singlet: idx={info['idx']}  "
                      f"E={E0:.8f}", flush=True)
        else:
            E0 = energies[0]
            psi0 = eigvecs[:, 0]

        degen = np.where(np.abs(energies - E0) < 1e-10)[0]
        if len(degen) > 1 and verbose:
            _lbl = "excited singlet" if excited_singlet else "ground state"
            print(f"    [WARN] {len(degen)}-fold degenerate {_lbl}")

        for n in n_values:
            sites = tuple(range(n))
            ops = _precompute_ops(sites, n_so, n_electrons,
                                  spin_preserving=spin_preserving)

            if len(degen) <= 1:
                # <=1: non-degenerate state — use psi0 directly. This also
                # guards the excited-singlet path where E0 comes from a
                # separate diagonalisation: if it fails to match any entry in
                # `energies` (degen empty), fall back to psi0 instead of
                # dividing a zero matrix by 0 (which produced silent NaNs).
                rho_raw = _rdm_from_ops(ops, psi0, n)
            else:
                dim = 2 ** n
                rho_raw = np.zeros((dim, dim), dtype=complex)
                for d_idx in degen:
                    rho_raw += _rdm_from_ops(ops, eigvecs[:, d_idx], n)
                rho_raw /= len(degen)

            tr = np.real(np.trace(rho_raw))
            rho = rho_raw / tr if abs(tr) > 1e-14 else rho_raw

            c_val, pur, _detail = _gme_lower_bound(
                rho, n,
                solver=solver, n_restarts=n_restarts,
                max_iter=max_iter, tol=tol, k_max_extra=k_max_extra,
            )
            C_GME[n][i] = c_val
            purity_dict[n][i] = pur
            entropy_dict[n][i] = float(compute_entropy(rho))
            rank_dict[n][i] = int(np.sum(np.real(np.linalg.eigvalsh(rho)) > 1e-10))
            convergence[n][i] = _detail

            if n == n_max and compute_purity(rho) > 1.0 - 1e-6:
                # pure formula is valid only for a pure RDM (same check as
                # _gme_lower_bound); a mixed n_max RDM would give a spurious line
                C_exact[i] = float(compute_C_GME_pure(rho, n_max))

            if n == 2:
                C_analytical_n2[i] = _cgme_analytical_n2(rho)
                C_wootters_n2[i]   = _wootters_concurrence(rho)

        if verbose:
            print("    " + "  ".join(
                f"{tag}(n={n})={C_GME[n][i]:.4f}" for n in n_values
            ))

    # Plot
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{output_prefix}_" if output_prefix else ""
    plot_path = os.path.join(
        output_dir, f"{prefix}task4_vscan_GME_L{L}_U{U}_t{t}_{_sector_tag(spin_preserving)}_{_grid_tag(V_values, 'V')}_{ts}.pdf"
    )

    result = {
        "V_values":      V_values,
        "V_over_t":      V_values / t,
        "C_GME":         C_GME,
        "C_exact_nL":    C_exact,
        "C_wootters_n2": C_wootters_n2,
        "C_analytical_n2": C_analytical_n2,
        "purity":        purity_dict,
        "entropy":       entropy_dict,
        "rank":          rank_dict,
        "convergence":   convergence,
        "n_values":      n_values,
        "solver":        solver,
        "system_info":   dict(L=L, t=t, U=U, mode=mode, Ne=n_electrons),
        "excited_singlet": bool(excited_singlet),
        "singlet_k":       int(singlet_k),
        "singlet_idx":     singlet_idx,
        "singlet_energy":  singlet_energy,
        "plot_path":     plot_path,
        "run_timestamp": ts,
    }

    plot_scan_finalized(result, plot_path, spin_preserving=spin_preserving)
    print(f"\n    Plot → {plot_path}")

    if save_pkl or use_pkl_cache:
        metadata = {
            "analysis_type": "V_dependence",
            "cache_version": cache_version,
            "L": int(L), "t": float(t), "U": float(U),
            "mode": str(mode), "Ne": int(n_electrons),
            "n_min": int(n_min), "n_max": int(n_max),
            "n_values": list(n_values),
            "V_values": V_values.tolist(),
            "V_digest": _array_digest(V_values),
            "solver": solver, "n_restarts": int(n_restarts),
            "max_iter": int(max_iter), "tol": float(tol),
            "k_max_extra": int(k_max_extra),
            "plot_path": plot_path,
            "run_timestamp": ts,
        }
        _save_thermal_result_pkl(
            cache_path=cache_path, metadata=metadata, result=result
        )
        result["pkl_path"] = str(cache_path)
        result["cache_path"] = str(cache_path)
        print(f"    V-scan PKL → {cache_path}")

    return result




def run_V_dependence_gme(
    L: int,
    V_values: np.ndarray,
    t: float,
    U: float,
    mode: str = "uniform",
    n_electrons: Optional[int] = None,
    n_min: int = 2,
    output_dir: str = "outputs",
    output_prefix: str = "",
    solver: str = "gcg",
    n_restarts: int = 20,
    max_iter: int = 300,
    tol: float = 1e-9,
    k_max_extra: int = 4,
    verbose: bool = True,
    save_pkl: bool = True,
    use_pkl_cache: bool = False,
    cache_version: str = "v1",
    force_recompute: bool = False,
    spin_preserving: bool = True,
    excited_singlet: bool = False,
    singlet_k: int = 12,
) -> dict:
    """
    Compute C_GME(n) at T=0 for each value of the nearest-neighbour interaction
    V (fixing t and U) — the extended-Hubbard V-scan.

    The x-axis is V/t. As V grows past ~U/2 the ring crosses from the Mott/SDW
    regime into a charge-density-wave (CDW) state, and short-range C_GME
    collapses — the transition shows up directly as a curve.

    Structurally identical to ``run_U_dependence_gme`` with U and V swapped:
    here U is fixed and V is swept.

    Returns
    -------
    dict
        V_values, V_over_t, C_GME {n: array}, purity {n: array},
        system_info, plot_path
    """
    if n_electrons is None:
        n_electrons = L
    # Cluster size cannot exceed the electron count: the n-site RDM trace is
    # ⟨Π_site n̂_site⟩, which is exactly 0 once n > Ne (can't occupy more sites
    # than there are electrons). Cap at min(L, Ne) so doped (Ne<L) runs stop at
    # n=Ne instead of hitting a zero-trace RDM. At half-filling Ne=L → no change.
    n_max = min(L, n_electrons)
    n_values = list(range(n_min, n_max + 1))
    V_values = np.asarray(V_values, dtype=float)
    ts = _run_timestamp()

    cache_path = _vscan_cache_path(
        L=L, U=U, t=t, mode=mode, n_electrons=n_electrons,
        n_min=n_min, solver=solver, n_restarts=n_restarts,
        max_iter=max_iter, tol=tol, k_max_extra=k_max_extra,
        V_values=V_values, cache_version=cache_version,
        spin_preserving=spin_preserving, excited_singlet=excited_singlet,
    )

    if use_pkl_cache and not force_recompute and cache_path.exists():
        payload = _load_thermal_result_pkl(cache_path)
        result = dict(payload["result"])
        result["loaded_from_cache"] = True
        result["cache_path"] = str(cache_path)
        os.makedirs(output_dir, exist_ok=True)
        prefix = f"{output_prefix}_" if output_prefix else ""
        plot_path = os.path.join(
            output_dir,
            f"{prefix}task4_vscan_GME_L{L}_U{U}_t{t}_{_sector_tag(spin_preserving)}_{_grid_tag(np.asarray(result['V_values']), 'V')}_{ts}.pdf"
        )
        plot_scan_finalized(result, plot_path, spin_preserving=spin_preserving)
        result["plot_path"] = plot_path
        print(f"\n[cache] loaded V-scan PKL → {cache_path}")
        print(f"[cache] regenerated plot → {plot_path}")
        return result

    print(f"\n{'='*64}")
    print(f"  V-DEPENDENCE GME SCAN (T=0)")
    print(f"{'='*64}")
    print(f"  L={L}  U={U}  t={t}  Ne={n_electrons}  mode={mode}")
    print(f"  V range [{V_values[0]:.4g}, {V_values[-1]:.4g}]  ({len(V_values)} points)")
    print(f"  n = {n_min}..{n_max}")
    print(f"{'-'*64}")

    C_GME: dict[int, np.ndarray] = {n: np.zeros(len(V_values)) for n in n_values}
    purity_dict: dict[int, np.ndarray] = {n: np.zeros(len(V_values)) for n in n_values}
    entropy_dict: dict[int, np.ndarray] = {n: np.zeros(len(V_values)) for n in n_values}
    rank_dict: dict[int, np.ndarray] = {n: np.zeros(len(V_values), dtype=int) for n in n_values}
    convergence: dict[int, list] = {n: [None] * len(V_values) for n in n_values}
    singlet_idx: np.ndarray = np.full(len(V_values), -1, dtype=int)
    singlet_energy: np.ndarray = np.full(len(V_values), np.nan)
    C_exact: np.ndarray = np.full(len(V_values), np.nan)
    C_wootters_n2: np.ndarray = np.full(len(V_values), np.nan)
    C_analytical_n2: np.ndarray = np.full(len(V_values), np.nan)
    tag = "C_GCG" if solver == "gcg" else "C_LB"

    for i, V in enumerate(V_values):
        if verbose:
            print(f"\n  V={V:.4g} (V/t={V/t:.4f}) …", flush=True)

        energies, eigvecs, n_so = compute_all_eigenstates(
            L, t, U, mode, n_electrons, spin_preserving=spin_preserving, V=V,
            classify_spin=(verbose and not excited_singlet), classify_k=singlet_k,
        )
        if excited_singlet:
            # The first-excited-singlet index moves with V through level
            # crossings, so re-locate it at every grid point.
            info = find_first_excited_singlet(
                L=L, t=t, U=float(U), V=V, mode=mode,
                n_electrons=n_electrons, k=singlet_k,
                spin_preserving=spin_preserving,
            )
            E0 = float(info["energy"])
            psi0 = np.asarray(info["psi"])
            singlet_idx[i] = int(info["idx"])
            singlet_energy[i] = E0
            if verbose:
                print(f"    first excited singlet: idx={info['idx']}  "
                      f"E={E0:.8f}", flush=True)
        else:
            E0 = energies[0]
            psi0 = eigvecs[:, 0]

        degen = np.where(np.abs(energies - E0) < 1e-10)[0]
        if len(degen) > 1 and verbose:
            _lbl = "excited singlet" if excited_singlet else "ground state"
            print(f"    [WARN] {len(degen)}-fold degenerate {_lbl}")

        for n in n_values:
            sites = tuple(range(n))
            ops = _precompute_ops(sites, n_so, n_electrons,
                                  spin_preserving=spin_preserving)

            if len(degen) <= 1:
                # <=1: non-degenerate state — use psi0 directly. This also
                # guards the excited-singlet path where E0 comes from a
                # separate diagonalisation: if it fails to match any entry in
                # `energies` (degen empty), fall back to psi0 instead of
                # dividing a zero matrix by 0 (which produced silent NaNs).
                rho_raw = _rdm_from_ops(ops, psi0, n)
            else:
                dim = 2 ** n
                rho_raw = np.zeros((dim, dim), dtype=complex)
                for d_idx in degen:
                    rho_raw += _rdm_from_ops(ops, eigvecs[:, d_idx], n)
                rho_raw /= len(degen)

            tr = np.real(np.trace(rho_raw))
            rho = rho_raw / tr if abs(tr) > 1e-14 else rho_raw

            c_val, pur, _detail = _gme_lower_bound(
                rho, n,
                solver=solver, n_restarts=n_restarts,
                max_iter=max_iter, tol=tol, k_max_extra=k_max_extra,
            )
            C_GME[n][i] = c_val
            purity_dict[n][i] = pur
            entropy_dict[n][i] = float(compute_entropy(rho))
            rank_dict[n][i] = int(np.sum(np.real(np.linalg.eigvalsh(rho)) > 1e-10))
            convergence[n][i] = _detail

            if n == n_max and compute_purity(rho) > 1.0 - 1e-6:
                # pure formula is valid only for a pure RDM (same check as
                # _gme_lower_bound); a mixed n_max RDM would give a spurious line
                C_exact[i] = float(compute_C_GME_pure(rho, n_max))

            if n == 2:
                C_analytical_n2[i] = _cgme_analytical_n2(rho)
                C_wootters_n2[i]   = _wootters_concurrence(rho)

        if verbose:
            print("    " + "  ".join(
                f"{tag}(n={n})={C_GME[n][i]:.4f}" for n in n_values
            ))

    # Plot
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{output_prefix}_" if output_prefix else ""
    plot_path = os.path.join(
        output_dir, f"{prefix}task4_vscan_GME_L{L}_U{U}_t{t}_{_sector_tag(spin_preserving)}_{_grid_tag(V_values, 'V')}_{ts}.pdf"
    )

    result = {
        "V_values":      V_values,
        "V_over_t":      V_values / t,
        "C_GME":         C_GME,
        "C_exact_nL":    C_exact,
        "C_wootters_n2": C_wootters_n2,
        "C_analytical_n2": C_analytical_n2,
        "purity":        purity_dict,
        "entropy":       entropy_dict,
        "rank":          rank_dict,
        "convergence":   convergence,
        "n_values":      n_values,
        "solver":        solver,
        "system_info":   dict(L=L, t=t, U=U, mode=mode, Ne=n_electrons),
        "excited_singlet": bool(excited_singlet),
        "singlet_k":       int(singlet_k),
        "singlet_idx":     singlet_idx,
        "singlet_energy":  singlet_energy,
        "plot_path":     plot_path,
        "run_timestamp": ts,
    }

    plot_scan_finalized(result, plot_path, spin_preserving=spin_preserving)
    print(f"\n    Plot → {plot_path}")

    if save_pkl or use_pkl_cache:
        metadata = {
            "analysis_type": "V_dependence",
            "cache_version": cache_version,
            "L": int(L), "t": float(t), "U": float(U),
            "mode": str(mode), "Ne": int(n_electrons),
            "n_min": int(n_min), "n_max": int(n_max),
            "n_values": list(n_values),
            "V_values": V_values.tolist(),
            "V_digest": _array_digest(V_values),
            "solver": solver, "n_restarts": int(n_restarts),
            "max_iter": int(max_iter), "tol": float(tol),
            "k_max_extra": int(k_max_extra),
            "plot_path": plot_path,
            "run_timestamp": ts,
        }
        _save_thermal_result_pkl(
            cache_path=cache_path, metadata=metadata, result=result
        )
        result["pkl_path"] = str(cache_path)
        result["cache_path"] = str(cache_path)
        print(f"    V-scan PKL → {cache_path}")

    return result
