"""
pipeline.py
-----------
End-to-end analysis drivers that wire all modules together.

Two public entry points
-----------------------
run_pure_analysis(...)
    Diagonalise → compute RDMs → generate PDF → return result dict.
    Suitable for a quick single-run or programmatic use.

run_full_GME_pipeline(...)
    Full pipeline: run_pure_analysis + spin observables + one PDF per
    n-body order (n = 3 … max_n_body).

New Features (Consolidated Convex Roof & Landscape)
────────────────────────────────────────────────────
• Unified convex roof analysis for mixed states (n < N)
  - 5 methods: paper bound, supervisor bound, upper bound, numerical, CVXPY SDP
  - Single call: full_convex_roof_analysis(rho, sites, ...)

• Unified landscape analysis for pure states (n = N)
  - Discrete scan, 1D continuous, 2D surface
  - Single call: landscape_full_report(rho, n, system_info, ...)

• Parallel convex-roof via n_workers (multiprocessing.Pool + tqdm)

• Eigenstate selection via state_idx (0 = ground state)

• New CLI flags: --landscape, --landscape-no-2d
"""

from __future__ import annotations

import os
import time
import pickle
import platform
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .hamiltonian import (
    compute_ground_and_excited_states,
    compute_k_eigenstates,
    compute_state_at_index,
    classify_eigenstates_spin,
    double_occupancy_operator,
)
from .rdm import compute_all_pure_rdms, compare_all_subsystem_rdms, print_rdm_results, generate_site_clusters
from .rdm import (
    get_rdm_pkl_results_dir,
    make_system_cache_key,
    load_or_compute_system_state_pkl,
    load_or_compute_rdm_pkl,
    save_rdm_pkl,
    _safe_float_tag,
    _ensure_tuple_int,
    _basic_env_meta,
)
from .spin import compute_spin_properties
from .visualization import generate_gme_pdf, generate_mixed_state_pdf
from .convex_roof import full_convex_roof_analysis  # ← CONSOLIDATED
from .landscape import landscape_full_report        # ← NEW

# The three expensive numerical convex-roof solvers controlled independently by
# the per-method PKL policy. Keep these keys identical to convex_roof.py output.
GME_SOLVER_KEYS = ("stiefel", "gcg", "stiefel_gcg")

# Resolved defaults used by convex_roof.py when CLI options are omitted.
# These values are written to cache filenames and PKL metadata so saved files
# show real settings instead of ambiguous tags such as "auto".
DEFAULT_GME_TOL = 1e-9
DEFAULT_GME_TOL_K = 1e-6
DEFAULT_GCG_MAX_ITER = 300
DEFAULT_STIEFEL_GCG_MAX_ITER = 300


def _effective_tol(tol: Optional[float]) -> float:
    return DEFAULT_GME_TOL if tol is None else float(tol)


def _effective_max_iter_meta(max_iter: Optional[int]) -> Dict[str, Any]:
    if max_iter is None:
        return {
            "stiefel": "adaptive_by_k",
            "gcg": DEFAULT_GCG_MAX_ITER,
            "stiefel_gcg": DEFAULT_STIEFEL_GCG_MAX_ITER,
        }
    v = int(max_iter)
    return {"stiefel": v, "gcg": v, "stiefel_gcg": v}


def _gme_iter_tag(max_iter: Any) -> str:
    if isinstance(max_iter, dict):
        return (
            f"stiefel-{_gme_safe_tag(max_iter.get('stiefel'))}_"
            f"gcg{max_iter.get('gcg')}_sgcg{max_iter.get('stiefel_gcg')}"
        )
    if max_iter is None:
        return f"stiefel-adaptive_gcg{DEFAULT_GCG_MAX_ITER}_sgcg{DEFAULT_STIEFEL_GCG_MAX_ITER}"
    return str(int(max_iter))


def _resolve_gme_solver_cache_actions(
    *,
    default_action: str = "reload",
    stiefel_action: str = "default",
    gcg_action: str = "default",
    stiefel_gcg_action: str = "default",
    reuse_gme_pkl: bool = True,
    force_gme_recompute: bool = False,
) -> Dict[str, str]:
    """Resolve global + per-method cache settings into one action per solver.

    Returned actions are only ``reload`` or ``recompute``.
    ``reload`` means: load from the cluster PKL when that solver exists there;
    if missing, recompute that solver and mark it as cache-missing/recomputed.
    """
    valid_default = {"reload", "recompute"}
    valid_override = {"default", "reload", "recompute"}

    if default_action not in valid_default:
        raise ValueError(f"gme_cache_default must be one of {sorted(valid_default)}")
    overrides = {
        "stiefel": stiefel_action,
        "gcg": gcg_action,
        "stiefel_gcg": stiefel_gcg_action,
    }
    for key, action in overrides.items():
        if action not in valid_override:
            raise ValueError(
                f"{key} cache action must be one of {sorted(valid_override)}"
            )

    # Backward-compatible hard global recompute switches.
    if force_gme_recompute or not reuse_gme_pkl:
        return {key: "recompute" for key in GME_SOLVER_KEYS}

    return {
        key: (default_action if action == "default" else action)
        for key, action in overrides.items()
    }


def _status_kind(status_obj: object) -> str:
    """Return a compact status string from either old string or new dict metadata."""
    if isinstance(status_obj, dict):
        return str(status_obj.get("status", "unknown"))
    return str(status_obj)


def _array_sha256(arr: Optional[np.ndarray]) -> Optional[str]:
    """Stable hash for numpy arrays stored only as metadata."""
    if arr is None:
        return None
    arr_c = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(arr_c.shape).encode("utf-8"))
    h.update(str(arr_c.dtype).encode("utf-8"))
    h.update(arr_c.view(np.uint8))
    return h.hexdigest()


def _rdm_integrity_metadata(rho: Optional[np.ndarray], metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Metadata saved with GME PKLs so a result can be audited later."""
    if rho is None:
        return {"rho_available_in_memory": False}
    eigvals = np.linalg.eigvalsh(rho)
    eigvals_real = np.real(eigvals)
    out: Dict[str, Any] = {
        "rho_available_in_memory": True,
        "rho_shape": tuple(rho.shape),
        "rho_dtype": str(rho.dtype),
        "rho_hash_sha256": _array_sha256(rho),
        "trace": float(np.real(np.trace(rho))),
        "purity": float(np.real(np.trace(rho @ rho))),
        "rank_threshold": 1e-10,
        "rank": int(np.sum(eigvals_real > 1e-10)),
        "eigvals": eigvals_real.tolist(),
        "eigvals_nonzero": eigvals_real[eigvals_real > 1e-10].tolist(),
        "lambda_min": float(np.min(eigvals_real)) if eigvals_real.size else None,
        "lambda_max": float(np.max(eigvals_real)) if eigvals_real.size else None,
        "hermitian_max_error": float(np.max(np.abs(rho - rho.conj().T))),
        "trace_error": float(abs(np.real(np.trace(rho)) - 1.0)),
    }
    if metrics:
        out["metrics_snapshot"] = metrics
    return out


def _solver_result_slim_summary(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Small audit summary extracted from a full cluster result."""
    if not isinstance(result, dict):
        return {}
    return {
        "sites": result.get("sites"),
        "n": result.get("n"),
        "global_purity": result.get("global_purity"),
        "C_LB_supervisor": result.get("C_LB_supervisor"),
        "C_LB_best": result.get("C_LB_best"),
        "C_stiefel": result.get("C_stiefel"),
        "C_gcg": result.get("C_gcg"),
        "C_stiefel_gcg": result.get("C_stiefel_gcg"),
        "C_UB": result.get("C_UB"),
        "solver_runtime": result.get("solver_runtime"),
        "solver_success": result.get("solver_success"),
        "solver_cache_status": result.get("solver_cache_status"),
        "effective_settings": result.get("effective_settings"),
        "elapsed_seconds": result.get("elapsed_seconds"),
    }
from .metrics import compute_purity
from .gme import compute_C_GME_pure, supervisor_gme_lower_bound
from .scan import run_U_scan, run_Ut_scan
from .plots import generate_scan_pdf
from openfermion.linalg import get_number_preserving_sparse_operator

from .convex_roof_multimethod import multi_method_convex_roof_analysis, ALL_METHODS
from .report_multimethod import generate_comparison_pdf, save_full_pkl


# ── GME-run persistence (mirrors convergence_benchmark.py, reuses rdm.py) ─────
#
# Central cache layout (extends the existing tree from rdm.py; nothing moved):
#
#     <project_root>/results/rdm_pkl_results/
#     +-- state_cache/         (rdm.py, reused for system/eigenstate caching)
#     +-- rdm_cache/           (rdm.py, reused for n-RDM caching)
#     +-- convergence_runs/    (convergence_benchmark.py, untouched)
#     +-- gme_runs/            (THIS pipeline)
#         +-- job_cache/       one PKL per (system, state, cluster, settings)
#
# Granularity: one job PKL per CLUSTER, with per-solver reuse inside that file.
# A run can reload stiefel from PKL, recompute gcg, reload stiefel_gcg, etc.
# The merged cluster result is saved back with per-method source metadata plus
# every k-record, restart_log, and decomposition for each solver.

def _gme_runs_dir() -> Path:
    d = get_rdm_pkl_results_dir() / "gme_runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _gme_job_cache_dir() -> Path:
    d = _gme_runs_dir() / "job_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _gme_safe_tag(x: Any) -> str:
    """Filename-safe string tag (keeps '-' for names like L-BFGS-B)."""
    s = str(x)
    for a, b in ((" ", ""), ("/", "-"), ("\\", "-"), (".", "p"), ("+", "")):
        s = s.replace(a, b)
    return s


def _gme_sites_tag(sites: Tuple[int, ...]) -> str:
    return "-".join(str(s) for s in _ensure_tuple_int(sites))


def _gme_state_label(state_idx: int) -> str:
    """0 -> 'gs' (keeps ground-state cache names stable); k>0 -> 'state{k}'."""
    return "gs" if int(state_idx) == 0 else f"state{int(state_idx)}"


def make_gme_cluster_filename(
    *,
    L: int, Ne: int, n_so: int, U: float, t: float, mode: str,
    state_label: str, sites: Tuple[int, ...],
    method: str, optimizer: str,
    n_restarts: int, max_iter: Optional[int], k_max_extra: int,
    tol: Optional[float], cache_version: str,
) -> str:
    iter_tag = _gme_iter_tag(max_iter)
    tol_tag = _safe_float_tag(_effective_tol(tol))
    return (
        f"gme_{cache_version}_"
        f"L{L}_Ne{Ne}_nso{n_so}_"
        f"U{_safe_float_tag(U)}_t{_safe_float_tag(t)}_"
        f"mode-{_gme_safe_tag(mode)}_state-{_gme_safe_tag(state_label)}_"
        f"sites-{_gme_sites_tag(sites)}_n{len(_ensure_tuple_int(sites))}_"
        f"meth-{_gme_safe_tag(method)}_opt-{_gme_safe_tag(optimizer)}_"
        f"restarts{n_restarts}_iter{iter_tag}_kmax{k_max_extra}_tol{tol_tag}.pkl"
    )


def build_gme_cluster_meta(
    *,
    L: int, Ne: int, n_so: int, U: float, t: float, mode: str,
    state_label: str, sites: Tuple[int, ...],
    method: str, optimizer: str,
    n_restarts: int, max_iter: Optional[int], k_max_extra: int,
    tol: Optional[float], cache_version: str,
) -> Dict[str, Any]:
    """Identity of one cluster job. Drives both filename and equality checks."""
    sites = _ensure_tuple_int(sites)
    return {
        "L": int(L), "Ne": int(Ne), "n_so": int(n_so),
        "U": float(U), "t": float(t), "mode": str(mode),
        "state_label": str(state_label), "sites": sites, "n": len(sites),
        "method": str(method), "optimizer": str(optimizer),
        "n_restarts": int(n_restarts),
        "max_iter_input": None if max_iter is None else int(max_iter),
        "max_iter": _effective_max_iter_meta(max_iter),
        "k_max_extra": int(k_max_extra),
        "tol_input": None if tol is None else float(tol),
        "tol": _effective_tol(tol),
        "tol_k": DEFAULT_GME_TOL_K,
        "cache_version": str(cache_version),
    }


def _extract_all_decompositions(cluster_result: Dict[str, Any]) -> Dict[str, list]:
    """
    Pull every decomposition out of a full_convex_roof_analysis result, for all
    three solver keys. Each entry is forward-compatible: it copies whatever the
    solver stored per component (at least {p, C_GME}; also psi when present).
    """
    out: Dict[str, list] = {}
    for key in ("stiefel", "gcg", "stiefel_gcg"):
        solver_result = cluster_result.get(key)
        if not isinstance(solver_result, dict):
            continue
        per_k = []
        for rec in solver_result.get("k_records", []) or []:
            decomp = rec.get("decomp", []) or []
            per_k.append({
                "k": rec.get("k"),
                "C_GME": rec.get("C_GME"),
                "n_components": len(decomp),
                "decomp": decomp,
                "restart_log": rec.get("restart_log"),
            })
        out[key] = per_k
    return out


def save_gme_cluster_result(
    *,
    cluster_result: Dict[str, Any],
    job_meta: Dict[str, Any],
    elapsed: Optional[float] = None,
    system_state_ref: Optional[Dict[str, Any]] = None,
    rdm_ref: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save one finished cluster result atomically (resumable across runs)."""
    job_cache_dir = _gme_job_cache_dir()
    filename = make_gme_cluster_filename(
        L=job_meta["L"], Ne=job_meta["Ne"], n_so=job_meta["n_so"],
        U=job_meta["U"], t=job_meta["t"], mode=job_meta["mode"],
        state_label=job_meta["state_label"], sites=job_meta["sites"],
        method=job_meta["method"], optimizer=job_meta["optimizer"],
        n_restarts=job_meta["n_restarts"], max_iter=job_meta["max_iter"],
        k_max_extra=job_meta["k_max_extra"], tol=job_meta["tol"],
        cache_version=job_meta["cache_version"],
    )
    path = job_cache_dir / filename
    _audit_summary = _solver_result_slim_summary(cluster_result)
    payload = {
        "meta": {
            **job_meta,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "environment": _basic_env_meta(),
        },
        "system_state_ref": system_state_ref,
        "rdm_ref": rdm_ref,
        "elapsed": None if elapsed is None else float(elapsed),
        "cluster_audit_summary": _audit_summary,
        "integrity": {
            "rho_hash_sha256": (rdm_ref or {}).get("rho_hash_sha256"),
            "rho_cache_path": (rdm_ref or {}).get("cache_path"),
            "system_state_cache_path": (system_state_ref or {}).get("cache_path"),
            "result_rho_hash_sha256": cluster_result.get("rho_hash_sha256") if isinstance(cluster_result, dict) else None,
            "hash_match": (
                (rdm_ref or {}).get("rho_hash_sha256") == cluster_result.get("rho_hash_sha256")
                if isinstance(cluster_result, dict) and (rdm_ref or {}).get("rho_hash_sha256") is not None
                else None
            ),
        },
        # Whole cluster result: scalar C-values, bounds, and the full per-method
        # dicts (k_records, best_record, decomp, restart_log) under the keys
        # "stiefel" / "gcg" / "stiefel_gcg".
        "result_full": cluster_result,
        "decompositions": _extract_all_decompositions(cluster_result),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return path


def load_gme_cluster_result(job_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Load an exact cached cluster result if present, else None."""
    job_cache_dir = _gme_job_cache_dir()
    filename = make_gme_cluster_filename(
        L=job_meta["L"], Ne=job_meta["Ne"], n_so=job_meta["n_so"],
        U=job_meta["U"], t=job_meta["t"], mode=job_meta["mode"],
        state_label=job_meta["state_label"], sites=job_meta["sites"],
        method=job_meta["method"], optimizer=job_meta["optimizer"],
        n_restarts=job_meta["n_restarts"], max_iter=job_meta["max_iter"],
        k_max_extra=job_meta["k_max_extra"], tol=job_meta["tol"],
        cache_version=job_meta["cache_version"],
    )
    path = job_cache_dir / filename
    if not path.exists():
        return None
    with open(path, "rb") as f:
        payload = pickle.load(f)
    payload["loaded_from_job_cache"] = True
    payload["job_cache_path"] = str(path)
    return payload


def save_gme_full_pkl(
    *,
    config: Dict[str, Any],
    system_state_ref: Optional[Dict[str, Any]],
    rdm_refs: Dict[Tuple[int, ...], Dict[str, Any]],
    results_by_sites: Dict[Tuple[int, ...], Dict[str, Any]],
    job_cache_paths: Optional[Dict[Tuple[int, ...], str]] = None,
    timing: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Per-invocation full GME-run PKL. Links to the system-state PKL and the
    per-cluster RDM PKLs (paths only; rho/psi are never duplicated) and stores
    every cluster result plus a flat decomposition index keyed by
    (sites, method_key, k).
    """
    gme_dir = _gme_runs_dir()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    decomposition_index: Dict[Any, Any] = {}
    for sites, cluster in results_by_sites.items():
        sites_t = _ensure_tuple_int(sites)
        for key, per_k in _extract_all_decompositions(cluster).items():
            for entry in per_k:
                decomposition_index[(sites_t, key, entry["k"])] = {
                    "C_GME": entry["C_GME"],
                    "n_components": entry["n_components"],
                    "decomp": entry["decomp"],
                }

    run_solver_runtime = {
        _ensure_tuple_int(s): c.get("solver_runtime")
        for s, c in results_by_sites.items() if isinstance(c, dict)
    }
    run_solver_success = {
        _ensure_tuple_int(s): c.get("solver_success")
        for s, c in results_by_sites.items() if isinstance(c, dict)
    }
    run_rho_hashes = {
        _ensure_tuple_int(s): c.get("rho_hash_sha256")
        for s, c in results_by_sites.items() if isinstance(c, dict)
    }

    full_data = {
        "meta": {
            "object_type": "gme_full_run",
            "created_at": stamp,
            "environment": _basic_env_meta(),
        },
        "config": config,
        "system_state_ref": system_state_ref,
        "rdm_refs": {_ensure_tuple_int(s): r for s, r in rdm_refs.items()},
        "results_by_sites": {_ensure_tuple_int(s): c for s, c in results_by_sites.items()},
        "job_cache_paths": job_cache_paths or {},
        "decomposition_index": decomposition_index,
        "run_solver_runtime": run_solver_runtime,
        "run_solver_success": run_solver_success,
        "run_rho_hashes": run_rho_hashes,
        "timing": timing or {},
    }

    iter_tag = _gme_iter_tag(config.get("max_iter"))
    filename = (
        f"gmefull_{config.get('cache_version', 'v1')}_"
        f"L{config['L']}_Ne{config['Ne']}_nso{config['n_so']}_"
        f"U{_safe_float_tag(config['U'])}_t{_safe_float_tag(config['t'])}_"
        f"mode-{_gme_safe_tag(config['mode'])}_state-{_gme_safe_tag(config['state_label'])}_"
        f"n{config['min_n_body']}-{config['max_n_body']}_"
        f"meth-{_gme_safe_tag(config.get('method', 'cgme'))}_"
        f"opt-{_gme_safe_tag(config.get('optimizer', 'L-BFGS-B'))}_"
        f"restarts{config['n_restarts']}_iter{iter_tag}_{stamp}.pkl"
    )
    path = gme_dir / filename
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(full_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return path


# ── Worker function for parallel convex-roof analysis ─────────────────────────
# Defined at module level so it can be pickled by multiprocessing.

def _convex_roof_worker(args):
    """
    Run full_convex_roof_analysis on one cluster.

    Each worker re-seeds NumPy with a per-cluster value so that:
      - different clusters explore different random restarts in parallel
      - the same input always produces the same output (reproducibility)
    """
    (
        sites, rho, cr_restarts, seed, n_restarts, k_max_extra, max_iter, tol,
        optimizer, method, active_solvers, precomputed_solvers, solver_cache_status,
    ) = args
    np.random.seed(seed)
    _worker_t0 = time.time()
    result = full_convex_roof_analysis(
        rho, sites,
        n_restarts=n_restarts, k_max_extra=k_max_extra, max_iter=max_iter,
        tol=tol, method=method, optimizer=optimizer,
        use_de=False,
        use_cvxpy=False,
        verbose=False,
        active_solvers=active_solvers,
        precomputed_solvers=precomputed_solvers,
        solver_cache_status=solver_cache_status,
    )
    result["worker_elapsed_seconds"] = float(time.time() - _worker_t0)
    result["worker_seed"] = int(seed)
    return sites, result


# ── Helper: build system_info dict ────────────────────────────────────────────

def _build_system_info(
    L: int,
    t: float,
    U: float,
    mode: str,
    n_electrons: int,
    psi0: np.ndarray,
    E0: float,
    E1: float,
    n_spin_orbitals: int,
) -> dict:
    """
    Compute spin observables and double occupancy, then package everything
    into a flat dictionary for use in PDF title pages.
    """
    # Double occupancy
    D_op = double_occupancy_operator(n_spin_orbitals, L)
    D_sparse = get_number_preserving_sparse_operator(
        D_op, n_spin_orbitals, n_electrons, spin_preserving=True
    )
    double_occ = float(np.vdot(psi0, D_sparse @ psi0).real)

    # Spin properties
    df = compute_spin_properties(
        U=U, t=t, L=L,
        n_spin_orbitals=n_spin_orbitals,
        eigvec=psi0,
        actual_electrons=n_electrons,
        ground_state_eng=E0,
        excited_state1_eng=E1,
        double_occ=double_occ,
    )

    return {
        "L":                   L,
        "Ne":                  n_electrons,
        "t":                   t,
        "U":                   U,
        "mode":                mode,
        "E0":                  float(df["ψ₀ Energy"].iloc[0]),
        "E1":                  float(df["ψ₁ Energy"].iloc[0]),
        "S2":                  float(df["⟨S²⟩"].iloc[0]),
        "S":                   float(df["S"].iloc[0]),
        "Sz":                  float(df["⟨Sz⟩"].iloc[0]),
        "multiplicity":        float(df["Multiplicity (2S+1)"].iloc[0]),
        "spin_label":          df["Spin State"].iloc[0],
        "double_occ":          float(df["⟨D⟩ total"].iloc[0]),
        "double_occ_per_site": float(df["⟨D⟩ / site"].iloc[0]),
    }


# ── Primary driver ─────────────────────────────────────────────────────────────

def run_pure_analysis(
    L: int,
    t: float,
    U: float,
    mode: str = "uniform",
    n_electrons: int = None,
    n_restarts: int = 20,
    k_max_extra: int   = 4,
    max_iter: int | None = None,
    optimizer:    str       = "L-BFGS-B",
    method:       str       = "c_gme",
    max_n_body: int = 4,
    min_n_body: int = 1,
    output_dir: str = "outputs",
    generate_pdf: bool = True,
    cr_restarts: int = 60,
    convex_roof_all: bool = False,
    verify_n_equals_N: bool = True,
    continuous_optimization: bool = False,
    continuous_real_only: bool = False,
    continuous_restarts: int = 20,
    landscape_analysis: bool = False,
    landscape_no_2d: bool = False,
    state_idx: int = 0,
    n_workers: int = 1,
    t_prime: float = None,
    t2:      float = None,
    t_perp:  float = None,
    is_multi:     bool = False,
    # ── PKL persistence / reuse (mirrors convergence_benchmark.py) ──
    use_pkl_cache: bool = False,
    cache_version: str = "v1",
    force_state_recompute: bool = False,
    force_rdm_recompute: bool = False,
    reuse_gme_pkl: bool = True,
    force_gme_recompute: bool = False,
    gme_cache_default: str = "reload",
    stiefel_cache_action: str = "default",
    gcg_cache_action: str = "default",
    stiefel_gcg_cache_action: str = "default",
    no_load_from_full_runs: bool = False,
    save_gme_pkl: bool = True,
    tol: float | None = None,
) -> dict:
    """
    Full RDM + GME + convex-roof + landscape analysis pipeline.

    Steps
    -----
    1. Exact diagonalisation → E₀, ψ₀, E₁, ψ₁
       (supports higher eigenstates via state_idx)
    2. n-body RDMs for n = min_n_body … max_n_body
    3. Convex-roof analysis (unified: paper, supervisor, upper, numerical, SDP)
       for all mixed n≥3 RDMs — serial or parallel via n_workers
    4. Landscape analysis (optional, for n=N pure states only)
       - Discrete scan: all basis-state pairs
       - 1D continuous: Bloch sphere interpolation
       - 2D surface: parameter space map
    5. n=N verification: if n_electrons == max_n_body, compare exact C_GME
       from eq.(1) against the supervisor lower bound from eqs.(6-9)
    6. PDF reports (optional):
          GME_{n}body_...pdf           — heatmap + GME + convex roof bounds
          ConvexRoof_{n}body_...pdf    — dedicated mixed-state report
          landscape_L{L}_Ne{Ne}_...pdf — landscape analysis (if --landscape)

    Parameters
    ----------
    L                  : Number of sites.
    t                  : Hopping parameter.
    U                  : On-site interaction.
    mode               : 'uniform' | 'open' | 'nnn' | 'ssh' | 'ladder'
    n_electrons        : Total electrons (default: half-filling = L).
    max_n_body         : Maximum RDM body order to compute.
    min_n_body         : Minimum RDM body order to retain.
    output_dir         : Directory where PDFs are written.
    generate_pdf       : If False, skip PDF generation.
    cr_restarts        : Random restarts for convex-roof numerical minimisation.
    convex_roof_all    : If True, run convex-roof on ALL RDMs (not just mixed).
    verify_n_equals_N  : If True and n_electrons == max_n_body, run verification.
    continuous_optimization : If True, run continuous supervisor optimisation.
    continuous_real_only    : If True, restrict continuous optimisation to real states.
    continuous_restarts     : Random restarts for continuous optimisation.
    landscape_analysis : If True, generate 3-page landscape PDF (n=N only).
    landscape_no_2d    : If True, skip 2D surface (faster landscape).
    state_idx          : Eigenstate index to analyse (0 = ground state).
    n_workers          : Number of parallel workers for convex-roof (1 = serial).
    t_prime / t2 / t_perp : Topological hopping parameters (passed to hamiltonian).

    Returns
    -------
    dict with keys:
        E0, E1, gap, psi0, psi1,
        rdm_gs, rdm_ex, metrics,
        cr_results, system_info,
        n_equals_N_results  (dict or None),
        continuous_results  (dict or None),
        landscape_results   (dict or None)
    """
    

    if n_electrons is None:
        n_electrons = L  # half-filling

    # Detect n = N condition
    is_n_equals_N = (max_n_body == L)

    # Validate landscape option
    if landscape_analysis and not is_n_equals_N:
        print(f"\n  [WARN] --landscape requires n=N (Ne={n_electrons} == max-n={max_n_body})")
        print(f"         Disabling landscape analysis.\n")
        landscape_analysis = False

    print("\n" + "=" * 60)
    print("  PURE RDM ANALYSIS PIPELINE")
    print("=" * 60)
    print(f"  L={L}  t={t}  U={U}  mode={mode}  Ne={n_electrons}")
    print(f"  n-body range: {min_n_body} … {max_n_body}")
    if state_idx > 0:
        print(f"  *** Using eigenstate index {state_idx}  "
              f"(= state {state_idx+1} in 1-indexed convention) ***")
    if is_n_equals_N:
        print(f"  *** n = N = {n_electrons}  →  pure state RDM  →  verification will run ***")
    if landscape_analysis:
        print(f"  *** Landscape analysis ENABLED ***")
    if n_workers and n_workers > 1:
        print(f"  *** Parallel convex-roof: {n_workers} workers ***")
    print("-" * 60)

    # ── 1. Diagonalise Hamiltonian ──
    print("\n[1/5] Diagonalising Hamiltonian …")

    if state_idx == 0:
        # Default — ground state.  Original behaviour, unchanged.
        # When PKL caching is enabled AND no topology extras are set (the cache
        # key does not encode t_prime/t2/t_perp, so those are computed directly
        # to avoid key collisions), reuse/save the system state via rdm.py.
        _topology_plain = (t_prime is None and t2 is None and t_perp is None)
        if use_pkl_cache and _topology_plain:
            print("      [cache] system-state PKL reuse enabled")
            _state = load_or_compute_system_state_pkl(
                compute_fn=compute_ground_and_excited_states,
                compute_kwargs=dict(
                    L=L, t=t, U=U, mode=mode, n_electrons=n_electrons,
                ),
                L=L, Ne=n_electrons, U=U, t=t, mode=mode,
                cache_version=cache_version,
                force_recompute=force_state_recompute,
            )
            E0 = _state["E0"]; psi0 = _state["psi0"]
            E1 = _state["E1"]; psi1 = _state["psi1"]
            n_spin_orbitals = _state["n_so"]
            _state_cache_path = _state["cache_path"]
            print(f"      [cache] loaded_from_cache={_state['loaded_from_cache']}  "
                  f"path={_state_cache_path}")
        else:
            E0, psi0, E1, psi1, n_spin_orbitals = compute_ground_and_excited_states(
                L=L, t=t, U=U, mode=mode, n_electrons=n_electrons,
                t_prime=t_prime, t2=t2, t_perp=t_perp,
            )
            _state_cache_path = None
        all_energies = None
        all_eigvecs  = None
    else:
        # Higher eigenstate requested.  Diagonalise enough states and pick
        # the requested one, then classify all lower states by spin so the
        # user can verify which is the first excited singlet etc.
        E0, psi0, E1, psi1, all_energies, all_eigvecs, n_spin_orbitals = (
            compute_state_at_index(
                L=L, t=t, U=U, mode=mode, n_electrons=n_electrons,
                state_idx=state_idx,
                t_prime=t_prime, t2=t2, t_perp=t_perp,
            )
        )
        _state_cache_path = None

        spin_table = classify_eigenstates_spin(
            eigvecs=all_eigvecs,
            energies=all_energies,
            L=L,
            n_electrons=n_electrons,
            n_spin_orbitals=n_spin_orbitals,
        )
        print()
        print("      Eigenstate spin classification")
        print("      ──────────────────────────────────────────────────────────")
        print(f"      {'idx':>3}  {'1-idx':>5}  {'energy':>14}  "
              f"{'<S^2>':>8}  {'S':>6}  spin label")
        print("      " + "-" * 60)
        for row in spin_table:
            arrow = "  <-- selected" if row["idx"] == state_idx else ""
            print(f"      {row['idx']:>3}  {row['idx']+1:>5}  "
                  f"{row['energy']:>14.8f}  {row['S2']:>8.4f}  "
                  f"{row['S']:>6.3f}  {row['spin_label']}{arrow}")
        print()

    print(f"      E₀ = {E0:.10f}    (selected state)")
    print(f"      E₁ = {E1:.10f}    (next state)")
    print(f"      Gap Δ = {E1 - E0:.6f}")

    # ── 2. Compute/load RDMs ──
    print(f"\n[2/5] Computing RDMs (n = {min_n_body} … {max_n_body}) …")

    # RDM PKL caching must be handled BEFORE computing the RDMs.
    # The old path computed all RDMs first and only saved them afterwards,
    # so a second run still recomputed them.  This branch uses rdm.py's
    # load_or_compute_rdm_pkl(...) helper cluster-by-cluster.
    rdm_cache_paths: dict = {}
    _rdm_cache_active = use_pkl_cache and (t_prime is None and t2 is None and t_perp is None)

    if _rdm_cache_active:
        rdm_gs, rdm_ex, metrics = {}, {}, {}
        n_gs_loaded = n_gs_computed = 0
        n_ex_loaded = n_ex_computed = 0
        gs_label = _gme_state_label(state_idx)
        ex_label = f"{gs_label}_next"

        print("      [cache] RDM PKL load/compute enabled")
        if force_rdm_recompute:
            print("      [cache] force_rdm_recompute=True -> recomputing RDMs and overwriting cache")

        for n_ord in range(min_n_body, max_n_body + 1):
            clusters = generate_site_clusters(L, n_ord)
            print(f"  Loading/computing {n_ord}-body RDMs ({len(clusters)} clusters) …", flush=True)

            for sites in clusters:
                sites_t = _ensure_tuple_int(sites)

                gs_res = load_or_compute_rdm_pkl(
                    psi=psi0,
                    L=L, Ne=n_electrons, U=U, t=t, mode=mode,
                    sites=sites_t,
                    n_so=n_spin_orbitals, n_electrons=n_electrons,
                    state_label=gs_label,
                    state_energy=E0,
                    parent_state_cache_path=_state_cache_path,
                    cache_version=cache_version,
                    force_recompute=force_rdm_recompute,
                )
                rdm_gs[sites_t] = gs_res["rho"]
                metrics[sites_t] = gs_res["metrics"]
                rdm_cache_paths[sites_t] = gs_res["cache_path"]
                if gs_res.get("loaded_from_cache", False):
                    n_gs_loaded += 1
                else:
                    n_gs_computed += 1

                # Keep the existing API return value rdm_ex populated.
                # It is cached separately so repeated full runs do not recompute psi1 RDMs either.
                ex_res = load_or_compute_rdm_pkl(
                    psi=psi1,
                    L=L, Ne=n_electrons, U=U, t=t, mode=mode,
                    sites=sites_t,
                    n_so=n_spin_orbitals, n_electrons=n_electrons,
                    state_label=ex_label,
                    state_energy=E1,
                    parent_state_cache_path=_state_cache_path,
                    cache_version=cache_version,
                    force_recompute=force_rdm_recompute,
                )
                rdm_ex[sites_t] = ex_res["rho"]
                if ex_res.get("loaded_from_cache", False):
                    n_ex_loaded += 1
                else:
                    n_ex_computed += 1

        print(
            "      [cache] RDM cache summary: "
            f"GS loaded={n_gs_loaded}, GS computed={n_gs_computed}; "
            f"next-state loaded={n_ex_loaded}, next-state computed={n_ex_computed}"
        )

    else:
        rdm_gs, rdm_ex, metrics = compute_all_pure_rdms(
            L=L, psi0=psi0, psi1=psi1,
            n_spin_orbitals=n_spin_orbitals,
            n_electrons=n_electrons,
            min_n=min_n_body,
            max_n=max_n_body,
        )

    # FILTER: Keep only RDMs with body order >= min_n_body
    rdm_gs = {sites: rho for sites, rho in rdm_gs.items() if len(sites) >= min_n_body}
    rdm_ex   = {sites: rho for sites, rho in rdm_ex.items()   if len(sites) >= min_n_body}
    metrics  = {sites: m   for sites, m   in metrics.items()  if len(sites) >= min_n_body}

    # check_result = compare_all_subsystem_rdms(rdm_gs)
    # print_rdm_results(check_result)
    # print(f"      Done. Total RDMs available: {len(rdm_gs)}")

    # ── 3. System info ──
    system_info = _build_system_info(
        L, t, U, mode, n_electrons, psi0, E0, E1, n_spin_orbitals
    )
    # Record which eigenstate was used so PDFs / outputs can reflect it.
    system_info["state_idx"] = state_idx

    # ── 4. UNIFIED CONVEX-ROOF ANALYSIS ──
    # ──────────────────────────────────────
    cr_results: dict = {}

    # GME PKL caching uses a key that does NOT encode topology extras, so it is
    # only enabled for plain topologies. For nnn/ssh/ladder with extras set,
    # convex-roof runs normally without GME caching.
    _topo_plain = (t_prime is None and t2 is None and t_perp is None)
    _gme_state_label_str = _gme_state_label(state_idx)
    _gme_cache_active = _topo_plain and (reuse_gme_pkl or save_gme_pkl)
    if not _topo_plain and (reuse_gme_pkl or save_gme_pkl):
        print("      [cache] GME PKL caching disabled for topology runs "
              "(t_prime/t2/t_perp set).")

    _solver_cache_actions = _resolve_gme_solver_cache_actions(
        default_action=gme_cache_default,
        stiefel_action=stiefel_cache_action,
        gcg_action=gcg_cache_action,
        stiefel_gcg_action=stiefel_gcg_cache_action,
        reuse_gme_pkl=reuse_gme_pkl,
        force_gme_recompute=force_gme_recompute,
    )
    if not _gme_cache_active:
        _solver_cache_actions = {key: "recompute" for key in GME_SOLVER_KEYS}

    print("      [cache] per-solver GME policy: " +
          ", ".join(f"{k}={v}" for k, v in _solver_cache_actions.items()))
    print("      [cache] effective solver settings: "
          f"n_restarts={n_restarts}, k_max_extra={k_max_extra}, "
          f"tol={_effective_tol(tol):.1e}, tol_k={DEFAULT_GME_TOL_K:.1e}, "
          f"max_iter={_effective_max_iter_meta(max_iter)}")

    # Light reference to the system-state PKL (paths only; psi never duplicated).
    _system_state_ref = None
    if _gme_cache_active and _state_cache_path is not None:
        _system_state_ref = {
            "cache_path": _state_cache_path,
            "state_label": _gme_state_label_str,
            "E0": float(E0), "E1": float(E1),
            "n_so": int(n_spin_orbitals),
        }

    # Build the task list — all clusters that need convex-roof.
    # New behaviour: each expensive solver is handled independently. A cluster
    # PKL may provide stiefel while gcg and stiefel_gcg are recomputed, or any
    # other combination. The final result is merged and saved back as one
    # cluster PKL so the next run can reuse the updated per-solver mixture.
    cr_tasks = []
    cr_job_meta: dict = {}     # sites -> job_meta
    cr_job_paths: dict = {}    # sites -> job_cache path (for the full PKL)
    n_fully_cached = 0

    for i, (sites, rho) in enumerate(rdm_gs.items()):
        if len(sites) < 3:
            continue
        if not convex_roof_all:
            # Only mixed RDMs are processed by convex-roof in this mode.
            if compute_purity(rho) >= 1 - 1e-6:
                continue

        sites_t = _ensure_tuple_int(sites)
        job_meta = None
        cached_payload = None
        cached_result = None
        cached_path = None

        if _gme_cache_active:
            job_meta = build_gme_cluster_meta(
                L=L, Ne=n_electrons, n_so=n_spin_orbitals, U=U, t=t, mode=mode,
                state_label=_gme_state_label_str, sites=sites,
                method=method, optimizer=optimizer,
                n_restarts=n_restarts, max_iter=max_iter,
                k_max_extra=k_max_extra, tol=tol, cache_version=cache_version,
            )
            cr_job_meta[sites_t] = job_meta

            if reuse_gme_pkl and not force_gme_recompute:
                cached_payload = load_gme_cluster_result(job_meta)
                if cached_payload is not None:
                    cached_result = cached_payload.get("result_full", {})
                    cached_path = cached_payload.get("job_cache_path")

        precomputed_solvers: Dict[str, Dict] = {}
        solver_status: Dict[str, Dict[str, Any]] = {}
        active_solvers = []

        for solver_key in GME_SOLVER_KEYS:
            requested_action = _solver_cache_actions[solver_key]
            cached_solver = None
            if isinstance(cached_result, dict):
                cached_solver = cached_result.get(solver_key)

            if requested_action == "reload" and isinstance(cached_solver, dict):
                precomputed_solvers[solver_key] = cached_solver
                solver_status[solver_key] = {
                    "action": "reload",
                    "status": "reused_pkl",
                    "path": cached_path,
                }
            else:
                active_solvers.append(solver_key)
                solver_status[solver_key] = {
                    "action": requested_action,
                    "status": "cache_missing_recomputed"
                              if requested_action == "reload" else "recomputed",
                    "path": cached_path,
                }

        if not active_solvers:
            # All three solver results came from the PKL. Copy the cached result
            # and attach fresh source metadata for visualization/reporting.
            merged = dict(cached_result)
            merged["solver_cache_status"] = solver_status
            merged["cache_status"] = solver_status
            merged["loaded_from_job_cache"] = True
            merged["job_cache_path"] = cached_path
            cr_results[sites] = merged
            cr_job_paths[sites_t] = cached_path
            n_fully_cached += 1
            continue

        # Per-cluster random seed: 12345 base + index for reproducibility.
        cr_tasks.append((
            sites, rho, cr_restarts, 12345 + i, n_restarts, k_max_extra,
            max_iter, tol, optimizer, method, tuple(active_solvers),
            precomputed_solvers, solver_status,
        ))

    if n_fully_cached:
        print(f"\n[3/5] Fully reused {n_fully_cached} cluster result(s) from gme_runs/job_cache.")

    def _save_one_cluster(sites, result, elapsed=None):
        """Persist one finished cluster result to job_cache (if caching active)."""
        if not (_gme_cache_active and save_gme_pkl):
            return
        sites_t = _ensure_tuple_int(sites)
        jm = cr_job_meta.get(sites_t)
        if jm is None:
            return
        _rho_for_ref = rdm_gs.get(sites)
        _rdm_integrity = _rdm_integrity_metadata(_rho_for_ref, metrics.get(sites))
        rdm_ref = {
            "sites": sites_t,
            "cache_path": rdm_cache_paths.get(sites_t),
            "metrics": metrics.get(sites),
            "rho_saved_in_this_file": False,
            **_rdm_integrity,
        }
        try:
            p = save_gme_cluster_result(
                cluster_result=result, job_meta=jm, elapsed=elapsed,
                system_state_ref=_system_state_ref, rdm_ref=rdm_ref,
            )
            cr_job_paths[sites_t] = str(p)
        except Exception as _e:
            print(f"      [cache] WARN could not save GME result for sites={sites}: {_e!r}")

    if cr_tasks:
        scope_str = "ALL" if convex_roof_all else "MIXED"
        print(f"\n[3/5] Running convex-roof analysis on {len(cr_tasks)} "
              f"{scope_str} clusters  (workers = {n_workers}) …")
        print("      recompute queue: " +
              "; ".join(
                  f"sites={_ensure_tuple_int(task[0])}:" + "+".join(task[10])
                  for task in cr_tasks[:8]
              ) + ("; ..." if len(cr_tasks) > 8 else ""))

        # tqdm is a soft dependency — fall back to a no-op iterator.
        try:
            from tqdm import tqdm
        except ImportError:
            print("      [info] tqdm not installed — running without progress bar.")
            def tqdm(it, **kw):   # noqa: E306
                return it

        if n_workers and n_workers > 1:
            # ── Parallel execution ────────────────────────────────────────────
            import multiprocessing as mp
            with mp.Pool(processes=n_workers) as pool:
                iterator = pool.imap_unordered(_convex_roof_worker, cr_tasks)
                for sites, result in tqdm(
                    iterator,
                    total=len(cr_tasks),
                    desc="      Convex-roof",
                    unit="cluster",
                    ncols=90,
                ):
                    cr_results[sites] = result
                    _save_one_cluster(sites, result, elapsed=result.get("worker_elapsed_seconds"))
        else:
            # ── Serial execution ──────────────────────────────────────────────
            for task in tqdm(
                cr_tasks,
                total=len(cr_tasks),
                desc="      Convex-roof",
                unit="cluster",
                ncols=90,
            ):
                _t0 = time.time()
                sites, result = _convex_roof_worker(task)
                cr_results[sites] = result
                _save_one_cluster(sites, result, elapsed=time.time() - _t0)

        print(f"      Convex-roof complete. ({len(cr_results)} clusters analysed)")
    elif n_fully_cached:
        print(f"      All requested clusters served from cache.")
    else:
        print(f"\n[3/5] No clusters require convex-roof — skipping.")

    if cr_results:
        solver_counts = {key: {"reused_pkl": 0, "recomputed": 0,
                               "cache_missing_recomputed": 0, "skipped": 0}
                         for key in GME_SOLVER_KEYS}
        for res in cr_results.values():
            st = res.get("solver_cache_status", res.get("cache_status", {}))
            for key in GME_SOLVER_KEYS:
                kind = _status_kind(st.get(key, "skipped")) if isinstance(st, dict) else "unknown"
                solver_counts[key].setdefault(kind, 0)
                solver_counts[key][kind] += 1
        print("      [cache] solver reuse/recompute summary:")
        for key in GME_SOLVER_KEYS:
            c = solver_counts[key]
            print(f"        {key:12s}: PKL={c.get('reused_pkl',0)}  "
                  f"new={c.get('recomputed',0)}  "
                  f"missing→new={c.get('cache_missing_recomputed',0)}")

    # ── 4a. Full GME-run PKL (one per invocation) ──
    if _gme_cache_active and save_gme_pkl and cr_results:
        try:
            _rdm_refs = {
                _ensure_tuple_int(s): {
                    "sites": _ensure_tuple_int(s),
                    "cache_path": rdm_cache_paths.get(_ensure_tuple_int(s)),
                    "metrics": metrics.get(s),
                    "rho_saved_in_this_file": False,
                    **_rdm_integrity_metadata(rdm_gs.get(s), metrics.get(s)),
                }
                for s in cr_results.keys()
            }
            _gme_config = {
                "L": L, "Ne": n_electrons, "n_so": n_spin_orbitals,
                "U": U, "t": t, "mode": mode,
                "state_label": _gme_state_label_str,
                "min_n_body": min_n_body, "max_n_body": max_n_body,
                "method": method, "optimizer": optimizer,
                "n_restarts": n_restarts,
                "max_iter_input": None if max_iter is None else int(max_iter),
                "max_iter": _effective_max_iter_meta(max_iter),
                "k_max_extra": k_max_extra,
                "tol_input": None if tol is None else float(tol),
                "tol": _effective_tol(tol),
                "tol_k": DEFAULT_GME_TOL_K,
                "convex_roof_all": convex_roof_all,
                "solver_cache_actions": _solver_cache_actions,
                "cache_version": cache_version,
            }
            _full_path = save_gme_full_pkl(
                config=_gme_config,
                system_state_ref=_system_state_ref,
                rdm_refs=_rdm_refs,
                results_by_sites=cr_results,
                job_cache_paths=cr_job_paths,
            )
            print(f"      [cache] full GME-run PKL → {_full_path}")
        except Exception as _e:
            print(f"      [cache] WARN could not write full GME PKL: {_e!r}")


    # ── Multi-method comparison (all clusters n >= 3) ──
    mm_rho_dict = {
        sites: rho
        for sites, rho in rdm_gs.items()
        if len(sites) >= 3
    }

    if mm_rho_dict and generate_pdf and is_multi:
        print(f"\n  Running multi-method convex-roof on {len(mm_rho_dict)} clusters ...")
        mm_results = multi_method_convex_roof_analysis(
            rho_dict   = mm_rho_dict,
            n_restarts = cr_restarts,
            methods    = ALL_METHODS,
            verbose    = True,
            L=L
        )

        # File-name suffix encoding the chosen eigenstate (only added when
        # state_idx > 0 so existing ground-state filenames are unchanged).
        state_suffix = f"_state{state_idx}" if state_idx > 0 else ""

        tag    = f"L{L}_Ne{n_electrons}_U{U}_t{t}{state_suffix}"
        pdf_mm = os.path.join(output_dir, f"comparison_{tag}.pdf")
        pkl_mm = os.path.join(output_dir, f"run_{tag}.pkl")

        generate_comparison_pdf(
            results     = mm_results,
            system_info = system_info,
            output_path = pdf_mm,
            n_restarts  = cr_restarts,
        )

        save_full_pkl(
            results      = mm_results,
            system_info  = system_info,
            output_path  = pkl_mm,
            ground_state = {"psi0": psi0, "psi1": psi1},
            rdm          = rdm_gs,
            metrics      = metrics,
            gme          = None,
        )

    # ── 4b. Continuous optimization (supervisor suggestion) ──
    continuous_results: dict = {}
    if continuous_optimization:
        from .gme import supervisor_gme_lower_bound_continuous
        n_cont = sum(
            1 for sites in rdm_gs
            if len(sites) >= 3
        )
        print(f"\n  Running continuous optimization "
              f"({'real' if continuous_real_only else 'complex'} states, "
              f"{n_cont} RDMs) …")
        for sites, rho in rdm_gs.items():
            if len(sites) < 3:
                continue
            cont_res = supervisor_gme_lower_bound_continuous(
                rho, len(sites),
                real_only=continuous_real_only,
                n_restarts=continuous_restarts,
                verbose=False,
            )
            continuous_results[sites] = cont_res
            print(f"      sites={sites}  C_LB={cont_res['C_LB']:.6f}  "
                  f"gain={cont_res['C_LB']-cont_res['C_LB_basis']:+.6f}")
        print(f"  Continuous optimization complete.")

    # ── 5. n = N verification ──
    n_equals_N_results: dict = {}
    if is_n_equals_N and verify_n_equals_N:
        print(f"\n[4/5] n = N verification  (pure state test) …")
        from .gme import supervisor_test_pure_state

        for n_ord in range(3, max_n_body + 1):
            if n_ord != n_electrons:
                continue

            rho_nN   = None
            sites_nN = None
            for sites, rho in rdm_gs.items():
                if len(sites) == n_ord:
                    rho_nN   = rho
                    sites_nN = sites
                    break

            if rho_nN is None:
                print(f"  [WARN] No {n_ord}-body RDM found — skipping verification.")
                continue

            pur = compute_purity(rho_nN)
            print(f"\n  Sites {sites_nN}  Tr(rho^2)={pur:.8f}")
            if abs(pur - 1.0) > 1e-4:
                print(f"  [WARN] RDM is not pure (Tr(rho^2)={pur:.6f}).")
                print(f"         This may happen if the ground state has degeneracy.")
                continue

            print(f"\n  Running supervisor verification for n={n_ord} body RDM …")
            verify_result = supervisor_test_pure_state(rho_nN, n_ord, verbose=False)
            n_equals_N_results[n_ord] = verify_result
            print(f"    C_exact={verify_result['C_exact']:.8f}  "
                  f"C_LB={verify_result['C_LB']:.8f}  "
                  f"gap={verify_result['difference']:.8f}")
        step_label = "[5/5]"
    else:
        step_label = "[4/4]"

    # ── 6. LANDSCAPE ANALYSIS ──
    landscape_results: dict = {}
    if landscape_analysis and is_n_equals_N:
        print(f"\n  Generating landscape analysis (discrete + 1D + 2D) …")

        rho_nN = None
        for sites, rho in rdm_gs.items():
            if len(sites) == n_electrons:
                rho_nN = rho
                break

        if rho_nN is not None:
            C_exact    = compute_C_GME_pure(rho_nN, n_electrons)
            sup_result = supervisor_gme_lower_bound(rho_nN, n_electrons)
            C_LB_basis = sup_result["C_LB"]

            ls_suffix      = f"_state{state_idx}" if state_idx > 0 else ""
            landscape_path = os.path.join(
                output_dir,
                f"landscape_L{L}_Ne{n_electrons}_U{U}{ls_suffix}.pdf"
            )

            print(f"\n  Generating landscape PDF → {landscape_path}")
            landscape_full_report(
                rho_nN,
                n=n_electrons,
                system_info=system_info,
                output_path=landscape_path,
                C_exact=C_exact,
                C_LB_basis=C_LB_basis,
            )

            landscape_results = {
                "C_exact":        C_exact,
                "C_LB_basis":     C_LB_basis,
                "landscape_path": landscape_path,
            }
            print(f"  Landscape analysis complete.")

    # ── 7. PDFs ──
    if generate_pdf:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n{step_label} Generating PDFs → {output_dir}/")

        # Suffix to keep ground-state output filenames unchanged.
        state_suffix = f"_state{state_idx}" if state_idx > 0 else ""

        for n in range(3, max_n_body + 1):

            # (a) Main GME PDF — heatmap + convex roof + supervisor bounds
            fname = os.path.join(
                output_dir,
                f"GME_{n}body_L{L}_{mode}_t{t}_U{U}{state_suffix}.pdf"
            )
            generate_gme_pdf(
                rdm_gs, metrics, n, fname, system_info,
                cr_results=cr_results,
                n_equals_N_results=n_equals_N_results if n_equals_N_results else None,
                continuous_results=continuous_results if continuous_results else None,
            )

            # # (b) Mixed-state / convex-roof PDF (only if mixed RDMs exist)
            # if cr_results:
            #     fname_cr = os.path.join(
            #         output_dir,
            #         f"ConvexRoof_{n}body_L{L}_{mode}_t{t}_U{U}{state_suffix}.pdf"
            #     )
            #     generate_mixed_state_pdf(
            #         rdm_gs, metrics, cr_results, n, fname_cr, system_info
            #     )

            # (c) n = N verification PDF — currently disabled; the main GME
            #     PDF already shows all convex-roof methods.
            # if is_n_equals_N and verify_n_equals_N and n in n_equals_N_results:
            #     ...
    else:
        print(f"\n{step_label} PDF generation skipped.")

    print("\n" + "=" * 60)
    print("  ANALYSIS COMPLETE")
    print("=" * 60 + "\n")

    return {
        "E0":                 E0,
        "E1":                 E1,
        "gap":                E1 - E0,
        "psi0":               psi0,
        "psi1":               psi1,
        "rdm_gs":           rdm_gs,
        "rdm_ex":             rdm_ex,
        "metrics":            metrics,
        "cr_results":         cr_results,
        "system_info":        system_info,
        "n_equals_N_results": n_equals_N_results,
        "continuous_results": continuous_results,
        "landscape_results":  landscape_results,
    }


# ── Full GME pipeline ──────────────────────────────────────────────────────────

def run_full_GME_pipeline(
    L: int,
    t: float,
    U: float,
    mode: str = "uniform",
    n_electrons: int = None,
    n_restarts: int = 20,
    k_max_extra: int   = 4,
    max_iter: int | None = None,
    optimizer:    str       = "L-BFGS-B",
    method:       str       = "c_gme",
    max_n_body: int = 4,
    min_n_body: int = 1,
    output_dir: str = "outputs",
    verify_n_equals_N: bool = True,
    continuous_optimization: bool = False,
    continuous_real_only: bool = False,
    continuous_restarts: int = 20,
    landscape_analysis: bool = False,
    landscape_no_2d: bool = False,
    convex_roof_all: bool = False,
    state_idx: int = 0,
    n_workers: int = 1,
    t_prime: float = None,
    t2:      float = None,
    t_perp:  float = None,
    # ── PKL persistence / reuse ──
    use_pkl_cache: bool = False,
    cache_version: str = "v1",
    force_state_recompute: bool = False,
    force_rdm_recompute: bool = False,
    reuse_gme_pkl: bool = True,
    force_gme_recompute: bool = False,
    gme_cache_default: str = "reload",
    stiefel_cache_action: str = "default",
    gcg_cache_action: str = "default",
    stiefel_gcg_cache_action: str = "default",
    no_load_from_full_runs: bool = False,
    save_gme_pkl: bool = True,
    tol: float | None = None,
) -> dict:
    """
    Convenience wrapper: run_pure_analysis with PDF generation enabled.

    Adds support for:
    • Unified convex roof analysis (5 methods)
    • Landscape analysis (discrete + 1D + 2D)
    • Eigenstate selection via state_idx (0 = ground state)
    • Parallel convex-roof via n_workers
    """
    return run_pure_analysis(
        L=L, t=t, U=U, mode=mode,
        n_electrons=n_electrons,
        n_restarts=n_restarts,
        k_max_extra=k_max_extra,
        max_iter=max_iter,
        optimizer=optimizer,
        method=method,
        max_n_body=max_n_body,
        min_n_body=min_n_body,
        output_dir=output_dir,
        generate_pdf=True,
        verify_n_equals_N=verify_n_equals_N,
        continuous_optimization=continuous_optimization,
        continuous_real_only=continuous_real_only,
        continuous_restarts=continuous_restarts,
        landscape_analysis=landscape_analysis,
        landscape_no_2d=landscape_no_2d,
        convex_roof_all=convex_roof_all,
        state_idx=state_idx,
        n_workers=n_workers,
        t_prime=t_prime, t2=t2, t_perp=t_perp,
        use_pkl_cache=use_pkl_cache,
        cache_version=cache_version,
        force_state_recompute=force_state_recompute,
        force_rdm_recompute=force_rdm_recompute,
        reuse_gme_pkl=reuse_gme_pkl,
        force_gme_recompute=force_gme_recompute,
        gme_cache_default=gme_cache_default,
        stiefel_cache_action=stiefel_cache_action,
        gcg_cache_action=gcg_cache_action,
        stiefel_gcg_cache_action=stiefel_gcg_cache_action,
        no_load_from_full_runs=no_load_from_full_runs,
        save_gme_pkl=save_gme_pkl,
        tol=tol,
    )


# ── Scan pipeline ──────────────────────────────────────────────────────────────

def run_scan_pipeline(
    L:            int,
    t:            float = 1.0,
    U_values:     list  = None,
    Ut_values:    list  = None,
    mode:         str   = "uniform",
    n_electrons:  int   = None,
    n_body:       int   = 3,
    k_states:     int   = 4,
    output_dir:   str   = "outputs",
    landscape_analysis: bool = False,
    landscape_no_2d:    bool = False,
    t_prime:      float = None,
    t2:           float = None,
    t_perp:       float = None,
) -> dict:
    """
    Full parameter scan pipeline.

    Runs either a U scan or a U/t scan (or both if both lists are provided),
    then generates:
        - scan_results_U.csv  / scan_results_Ut.csv
        - ScanReport_U.pdf    / ScanReport_Ut.pdf   (5-page each)

    Parameters
    ----------
    L                  : Number of sites.
    t                  : Hopping amplitude (fixed throughout scan).
    U_values           : List of U values  (x-axis = U).
    Ut_values          : List of U/t ratios (x-axis = U/t).
    mode               : 'uniform' | 'dimerized' | 'heisenberg'
    n_electrons        : Defaults to half-filling.
    n_body             : Body order of RDM to analyse (default 3).
    k_states           : Number of eigenstates per point (default 4).
    output_dir         : Output directory for CSVs and PDFs.
    landscape_analysis : If True, generate landscape at each scan point (n=N only).
    landscape_no_2d    : If True, skip 2D surface in landscape (faster).

    Returns
    -------
    dict with keys:
        df_full_U, df_agg_U   (if U_values given)
        df_full_Ut, df_agg_Ut (if Ut_values given)
    """
    if n_electrons is None:
        n_electrons = L

    os.makedirs(output_dir, exist_ok=True)
    results = {}
    topo_kwargs = dict(t_prime=t_prime, t2=t2, t_perp=t_perp)

    print("\n" + "=" * 60)
    print("  PARAMETER SCAN PIPELINE")
    print("=" * 60)
    print(f"  L={L}  t={t}  mode={mode}  n_body={n_body}  k_states={k_states}")
    if landscape_analysis:
        print(f"  *** Landscape analysis at each scan point ENABLED ***")
    if any(v is not None for v in topo_kwargs.values()):
        print(f"  topo extras: {topo_kwargs}")
    print("-" * 60)

    # ── U scan ──
    if U_values is not None:
        print(f"\n[U scan]  {len(U_values)} points: U ∈ [{min(U_values):.2f}, {max(U_values):.2f}]")
        csv_path = os.path.join(output_dir, f"scan_results_U_L{L}_{mode}.csv")
        df_full_U, df_agg_U = run_U_scan(
            L=L, t=t, U_values=U_values, mode=mode,
            n_electrons=n_electrons, n_body=n_body,
            k_states=k_states, save_csv=csv_path, verbose=True,
            **topo_kwargs,
        )
        pdf_path = os.path.join(output_dir, f"ScanReport_U_L{L}_{mode}_n{n_body}.pdf")
        generate_scan_pdf(df_agg_U, L=L, n_body=n_body, mode=mode, output_path=pdf_path)
        results["df_full_U"] = df_full_U
        results["df_agg_U"]  = df_agg_U

    # ── U/t scan ──
    if Ut_values is not None:
        print(f"\n[U/t scan]  {len(Ut_values)} points: U/t ∈ [{min(Ut_values):.2f}, {max(Ut_values):.2f}]")
        csv_path = os.path.join(output_dir, f"scan_results_Ut_L{L}_{mode}.csv")
        df_full_Ut, df_agg_Ut = run_Ut_scan(
            L=L, Ut_values=Ut_values, t_fixed=t, mode=mode,
            n_electrons=n_electrons, n_body=n_body,
            k_states=k_states, save_csv=csv_path, verbose=True,
            **topo_kwargs,
        )
        pdf_path = os.path.join(output_dir, f"ScanReport_Ut_L{L}_{mode}_n{n_body}.pdf")
        generate_scan_pdf(df_agg_Ut, L=L, n_body=n_body, mode=mode, output_path=pdf_path)
        results["df_full_Ut"] = df_full_Ut
        results["df_agg_Ut"]  = df_agg_Ut

    print("\n" + "=" * 60)
    print("  SCAN COMPLETE")
    print("=" * 60 + "\n")
    return results