#!/usr/bin/env python3
"""
generate_ml_dataset.py
----------------------
Build a training dataset for an ML surrogate of GME measures on
Hubbard n-body RDMs.

Outputs (under --outdir, default = ./data_for_ML/):

    data_for_ML/
        manifest.csv          # scalar metadata for every record (one row each)
        records/<hash>.npz    # full record: rho, eigvals, all labels, config
        grid_summary.json     # the parameter grid used (for provenance)
        run_metadata.json     # date, package version, git SHA if available

Design choices:
  - Stratified U/t sampling (denser around the crossover U/t ~ 3-10)
  - Multiple Stiefel restarts (best-of-k as label, std as label-noise signal)
  - Diagonalize each Hamiltonian once, then loop over (state, sites)
  - Resumable: each (config, sites) is hashed; existing files are skipped
  - Saves both raw RDM (rho_real, rho_imag) AND derived features

Usage examples:
    # Quick test — sweep U at fixed t=1, small grid
    python generate_ml_dataset.py --L 4 --modes uniform --n-ut 8

    # Default full v1 dataset — sweeps U, fixed t=1
    python generate_ml_dataset.py --L 4 6 --n-ut 30

    # Sweep t at fixed U=4 instead
    python generate_ml_dataset.py --L 4 6 --scan-mode vary_t --U-fixed 4.0 --n-t 30

    # Run BOTH sweeps for full coverage (~2x data)
    python generate_ml_dataset.py --L 4 6 --scan-mode both

    # Single physics point: U=4, t=1 (matches main.py defaults)
    python generate_ml_dataset.py --L 4 --t 1.0 --n-ut 1 --ut-min 4.0 --ut-max 4.0

    # Resume an interrupted run (default; rerun the same command)
    python generate_ml_dataset.py --L 4 6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# ── Path setup so we can import hubbard_gme as in main.py ────────────────────
# This script lives in <repo>/gme_qml/; the package lives in <repo>/src/.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # <repo>/gme_qml
_REPO_ROOT = os.path.dirname(_THIS_DIR)                          # <repo>
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

# These mirror your main.py imports
from hubbard_gme.hamiltonian import (
    compute_k_eigenstates,
    double_occupancy_operator,
    classify_eigenstates_spin,
    VALID_MODES,
)
from hubbard_gme.rdm import compute_general_nRDM, generate_site_clusters
from hubbard_gme.gme import compute_C_GME_pure
from hubbard_gme.metrics import compute_entropy, compute_purity
from hubbard_gme.convex_roof import (
    convex_roof_stiefel,
    convex_roof_gcg,
    convex_roof_lower_bound_supervisor,
    convex_roof_upper_bound,
)

from openfermion.linalg import get_number_preserving_sparse_operator


# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

PURITY_PURE_THRESHOLD = 1.0 - 1e-6
RAW_TRACE_MIN = 1e-12
BRACKET_TOL = 1e-3   # tolerance when checking C_target ∈ [LB, UB]

# Manifest is appended to in chunks to survive crashes
FLUSH_EVERY_N_RECORDS = 25


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def stratified_ut_grid(n_points: int = 30,
                       ut_min: float = 0.1,
                       ut_max: float = 20.0) -> np.ndarray:
    """
    Stratified U/t grid: log-spaced in weak/strong, linear-dense in crossover.

    Roughly:  25% weak (ut_min .. 3),
              50% crossover (3 .. 10),
              25% strong (10 .. ut_max).
    """
    n_low   = max(2, n_points // 4)
    n_cross = max(4, n_points // 2)
    n_high  = max(2, n_points - n_low - n_cross)

    low   = np.geomspace(ut_min, 3.0, n_low, endpoint=False)
    cross = np.linspace(3.0, 10.0, n_cross, endpoint=False)
    high  = np.geomspace(10.0, ut_max, n_high)
    grid = np.unique(np.concatenate([low, cross, high]))
    return grid


def stratified_t_grid(n_points: int,
                      t_min: float,
                      t_max: float,
                      U_fixed: float) -> np.ndarray:
    """
    Stratified t grid at fixed U: denser where U/t falls in the crossover.

    The "interesting" t-range corresponds to U/t in [3, 10], i.e.
    t in [U_fixed/10, U_fixed/3]. We put 50% of the points there.
    """
    t_cross_lo = U_fixed / 10.0
    t_cross_hi = U_fixed / 3.0

    # Clip the crossover region to the user-requested [t_min, t_max] window
    t_cross_lo = max(t_cross_lo, t_min)
    t_cross_hi = min(t_cross_hi, t_max)

    n_low   = max(2, n_points // 4)
    n_cross = max(4, n_points // 2)
    n_high  = max(2, n_points - n_low - n_cross)

    parts = []
    if t_min < t_cross_lo:
        parts.append(np.geomspace(t_min, t_cross_lo, n_low, endpoint=False))
    if t_cross_lo < t_cross_hi:
        parts.append(np.linspace(t_cross_lo, t_cross_hi, n_cross, endpoint=False))
    if t_cross_hi < t_max:
        parts.append(np.geomspace(t_cross_hi, t_max, n_high))

    if not parts:   # fallback: just linspace
        return np.linspace(t_min, t_max, n_points)

    return np.unique(np.concatenate(parts))


def is_valid_config(L: int, mode: str, n_body: int, n_electrons: int) -> bool:
    """Reject impossible (L, mode, n_body) combinations early."""
    if mode == "ladder" and L % 2 != 0:
        return False
    if n_body > L or n_body > n_electrons:
        return False
    if n_body < 2:        # n=1 RDMs have no GME
        return False
    return True


def config_hash(cfg: Dict[str, Any]) -> str:
    """
    Stable hash for a config (uniqueness guarantee + filename suffix).

    Excludes 'sweep_tag' and 'config_hash' from the hash, so that the same
    physics point (L, t, U, mode, sites, ...) reached via different sweeps
    (vary_U vs vary_t) produces ONE record, not two.
    """
    EXCLUDE = {"sweep_tag", "config_hash", "hash"}
    serialisable = {k: v for k, v in cfg.items() if k not in EXCLUDE}
    s = json.dumps(serialisable, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:12]


def record_filename(cfg: Dict[str, Any]) -> str:
    """
    Build a human-readable .npz filename from a config dict.

    Pattern:
        L{L}_{mode}_U{U:06.3f}_t{t:06.3f}_s{state_idx}_n{n_body}_sites{a-b-c}_{hash12}.npz

    Example:
        L4_uniform_U02.500_t01.000_s0_n3_sites0-1-2_1fbd238f06e3.npz

    Why both U and t (not just U/t):
        The same U/t ratio can be reached by (U=2.5, t=1.0) or (U=10, t=4.0).
        These produce DIFFERENT physical RDMs (different absolute energy
        scales). Encoding both in the filename keeps vary_U and vary_t
        records visually distinguishable.

    Globbable:
        L4_uniform_*       → all uniform L=4 records
        *_n3_*             → all 3-body RDMs
        *_t01.000_*        → all records at t=1
        *_U04.000_*        → all records at U=4

    The hash suffix guarantees uniqueness even if other fields
    (like n_electrons) vary across runs.
    """
    L         = cfg["L"]
    mode      = cfg["mode"]
    U         = cfg["U"]
    t         = cfg["t"]
    state_idx = cfg["state_idx"]
    n_body    = cfg["n_body"]
    sites     = cfg["sites"]
    h         = cfg["config_hash"]

    sites_str = "-".join(str(s) for s in sites)
    return (
        f"L{L}_{mode}"
        f"_U{U:06.3f}_t{t:06.3f}"
        f"_s{state_idx}_n{n_body}"
        f"_sites{sites_str}_{h}.npz"
    )


def compute_spin_observables(psi: np.ndarray, L: int,
                             n_so: int, n_e: int) -> Tuple[float, float, float]:
    """
    Compute <Sz> for one wavefunction.
    For S2/S/spin_label we use the existing classify_eigenstates_spin
    on the full eigvec batch — see main loop.
    """
    from openfermion.hamiltonians import sz_operator
    Sz_sp = 0.5 * get_number_preserving_sparse_operator(
        sz_operator(L), n_so, n_e, spin_preserving=True
    )
    return float(np.vdot(psi, Sz_sp @ psi).real)


def safe_extract_value(out: Any) -> float:
    """Extract a float from a label-function output (handles dict or scalar)."""
    if isinstance(out, dict):
        for key in ("C_gcg", "C_stiefel", "C_LB", "C_UB", "C", "value", "C_GME"):
            if key in out:
                return float(out[key])
        # Fallback: first numeric value
        for v in out.values():
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
        return float("nan")
    return float(out)


def call_label_fn(fn, rho: np.ndarray, n_body: int,
                  **kwargs) -> Tuple[float, str]:
    """
    Call a label-producing function defensively.

    Tries (rho, n_body, **kwargs); if that signature fails, falls back to
    (rho, **kwargs), then (rho,). Returns (value, error_str_or_None).
    """
    last_err = None
    for args in [(rho, n_body), (rho,)]:
        try:
            return safe_extract_value(fn(*args, **kwargs)), None
        except TypeError as e:
            last_err = str(e)
            continue
        except Exception as e:
            return float("nan"), f"{type(e).__name__}: {e}"
    return float("nan"), f"signature mismatch: {last_err}"


def call_stiefel(rho: np.ndarray, n_body: int,
                 n_restarts: int = 20) -> Tuple[float, str]:
    """
    Call convex_roof_stiefel with the real signature.
    n_restarts controls Stiefel's *internal* random restarts (default 20).
    """
    try:
        out = convex_roof_stiefel(
            rho, n_body,
            n_restarts=n_restarts,
            verbose=False,        # silence per-RDM stdout in batch runs
        )
        return safe_extract_value(out), None
    except Exception as e:
        return float("nan"), f"{type(e).__name__}: {e}"


def call_gcg(rho: np.ndarray, n_body: int,
             n_restarts: int = 10,
             max_iter: int = 200,
             tol: float = 1e-9,
             k_max_extra: int = 4) -> Tuple[float, str]:
    """
    Call convex_roof_gcg — the verified-best convex-roof solver for mixed
    RDMs (same solver + defaults as the thermal pipeline's GCG settings).
    """
    try:
        out = convex_roof_gcg(
            rho, n_body,
            n_restarts=n_restarts,
            max_iter=max_iter,
            tol=tol,
            k_max_extra=k_max_extra,
            verbose=False,
        )
        return safe_extract_value(out), None
    except Exception as e:
        return float("nan"), f"{type(e).__name__}: {e}"


# ────────────────────────────────────────────────────────────────────────────
# Label computation
# ────────────────────────────────────────────────────────────────────────────

def compute_labels_for_rdm(rho: np.ndarray,
                           n_body: int,
                           n_outer_restarts: int = 1,
                           n_inner_restarts: int = 20,
                           seed_base: int = 0,
                           label_solver: str = "gcg",
                           gcg_max_iter: int = 200,
                           gcg_tol: float = 1e-9,
                           gcg_k_max_extra: int = 4) -> Dict[str, Any]:
    """
    Compute every label / constraint / auxiliary feature for a single RDM.

    label_solver selects the convex-roof solver for MIXED states:
      * "gcg"     (default) — convex_roof_gcg, the verified-best solver
                  (n_inner_restarts = its n_restarts; gcg_* = its settings)
      * "stiefel" — the legacy v1 labeling (kept for comparison)
    n_outer_restarts independent solver calls are executed; the best (lowest)
    C across them is the training label, the spread quantifies label noise.

    Returns a dict with keys:
        purity, is_pure, entropy
        C_pure                  (always computed; meaningful only if pure)
        lower_bound_super, C_UB
        C_<solver>_best/mean/std/all/n_valid/total_time for the chosen solver
        (the other solver's columns are emitted as NaN for schema stability)
        C_gcg                   (plain column = C_gcg_best; NaN for stiefel runs)
        C_target, target_source, target_in_bracket, label_solver
        plus *_error fields when a call failed
    """
    if label_solver not in ("gcg", "stiefel"):
        raise ValueError(f"label_solver must be 'gcg' or 'stiefel', got {label_solver!r}")

    purity = compute_purity(rho)
    is_pure = purity > PURITY_PURE_THRESHOLD
    out: Dict[str, Any] = {
        "purity":  float(purity),
        "is_pure": bool(is_pure),
        "entropy": float(compute_entropy(rho)),
        "label_solver": label_solver,
    }

    # ── Pure formula (always — for pure ρ this IS the convex roof) ──
    c_pure, err = call_label_fn(compute_C_GME_pure, rho, n_body)
    out["C_pure"] = c_pure
    if err: out["C_pure_error"] = err

    # ── Supervisor lower bound (key 'C_LB' in the returned dict) ────
    c_lb, err = call_label_fn(convex_roof_lower_bound_supervisor, rho, n_body)
    out["lower_bound_super"] = c_lb
    if err: out["lower_bound_super_error"] = err

    # ── Upper bound (key 'C_UB' in the returned dict) ───────────────
    c_ub, err = call_label_fn(convex_roof_upper_bound, rho, n_body)
    out["C_UB"] = c_ub
    if err: out["C_UB_error"] = err

    # ── Convex-roof solver (chosen by label_solver) ─────────────────
    nan = float("nan")
    # Emit BOTH key families so the manifest schema is identical across runs.
    for fam in ("stiefel", "gcg"):
        out.update({
            f"C_{fam}_best": nan, f"C_{fam}_mean": nan, f"C_{fam}_std": nan,
            f"C_{fam}_all": [], f"C_{fam}_n_valid": 0,
            f"C_{fam}_total_time": 0.0,
        })

    fam = label_solver
    if is_pure:
        # For a pure ρ, the convex roof IS the pure-formula value.
        out.update({
            f"C_{fam}_best":       c_pure,
            f"C_{fam}_mean":       c_pure,
            f"C_{fam}_std":        0.0,
            f"C_{fam}_all":        [c_pure] * n_outer_restarts,
            f"C_{fam}_n_valid":    n_outer_restarts,
            f"C_{fam}_total_time": 0.0,
        })
    else:
        values, errors, total_t = [], [], 0.0
        for k in range(n_outer_restarts):
            np.random.seed(seed_base + 7919 * k)   # affects solver internals
            t0 = time.time()
            if label_solver == "gcg":
                v, e = call_gcg(rho, n_body,
                                n_restarts=n_inner_restarts,
                                max_iter=gcg_max_iter,
                                tol=gcg_tol,
                                k_max_extra=gcg_k_max_extra)
            else:
                v, e = call_stiefel(rho, n_body, n_restarts=n_inner_restarts)
            total_t += time.time() - t0
            values.append(v)
            if e:
                errors.append(e)
        valid = [v for v in values if not np.isnan(v)]
        out.update({
            f"C_{fam}_all":        values,
            f"C_{fam}_best":       float(np.min(valid))  if valid else nan,
            f"C_{fam}_mean":       float(np.mean(valid)) if valid else nan,
            f"C_{fam}_std":        float(np.std(valid))  if len(valid) > 1 else 0.0,
            f"C_{fam}_n_valid":    len(valid),
            f"C_{fam}_total_time": total_t,
        })
        if errors:
            out[f"C_{fam}_errors"] = errors

    # Plain C_gcg column: lets downstream loaders prefer GCG labels directly.
    out["C_gcg"] = out["C_gcg_best"]

    # ── Training target + sanity check ──────────────────────────────
    out["C_target"]      = c_pure if is_pure else out[f"C_{fam}_best"]
    out["target_source"] = "pure_formula" if is_pure else label_solver

    in_bracket = True
    if not (np.isnan(c_lb) or np.isnan(c_ub) or np.isnan(out["C_target"])):
        in_bracket = (c_lb - BRACKET_TOL <= out["C_target"] <= c_ub + BRACKET_TOL)
    out["target_in_bracket"] = bool(in_bracket)

    return out


# ────────────────────────────────────────────────────────────────────────────
# Manifest I/O
# ────────────────────────────────────────────────────────────────────────────

# Columns NOT to keep in the manifest CSV (they're large / non-scalar)
_MANIFEST_DROP_COLS = {
    "C_stiefel_all", "C_stiefel_errors",
    "C_gcg_all", "C_gcg_errors",
    "rho_real", "rho_imag",
}


def _to_manifest_row(record: Dict[str, Any]) -> Dict[str, Any]:
    """Strip large fields, stringify list-likes for the manifest CSV."""
    row: Dict[str, Any] = {}
    for k, v in record.items():
        if k in _MANIFEST_DROP_COLS:
            continue
        if isinstance(v, (list, tuple)):
            row[k] = json.dumps(v) if k in ("sites", "rdm_eigvals") else str(v)
        elif isinstance(v, np.ndarray):
            row[k] = json.dumps(v.tolist())
        elif isinstance(v, dict):
            row[k] = json.dumps(v)
        else:
            row[k] = v
    return row


def append_manifest(rows: List[Dict[str, Any]], manifest_path: Path) -> None:
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    if manifest_path.exists():
        df_new.to_csv(manifest_path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(manifest_path, index=False)


def load_done_hashes(manifest_path: Path) -> set:
    if not manifest_path.exists():
        return set()
    try:
        df = pd.read_csv(manifest_path, usecols=["config_hash"])
        return set(df["config_hash"].astype(str))
    except Exception:
        return set()


# ────────────────────────────────────────────────────────────────────────────
# Main driver
# ────────────────────────────────────────────────────────────────────────────

def build_config_grid(args) -> List[Dict[str, Any]]:
    """
    Enumerate all (L, mode, sweep_point, state_idx, n_body, sites) configs.

    Sweep modes (--scan-mode):
        vary_U : t fixed at --t, U scanned via stratified U/t grid.
        vary_t : U fixed at --U-fixed, t scanned via stratified t grid.
        both   : runs vary_U then vary_t and concatenates (de-duplicated by hash).
    """
    configs: List[Dict[str, Any]] = []

    # ── Build the (U, t) sweep points ────────────────────────────────────
    sweep_points: List[Tuple[float, float, str]] = []   # list of (U, t, sweep_tag)

    if args.scan_mode in ("vary_U", "both"):
        ut_grid = stratified_ut_grid(args.n_ut, args.ut_min, args.ut_max)
        for ut in ut_grid:
            sweep_points.append((float(ut * args.t), float(args.t), "vary_U"))

    if args.scan_mode in ("vary_t", "both"):
        t_grid = stratified_t_grid(
            args.n_t, args.t_min, args.t_max, args.U_fixed,
        )
        for t_val in t_grid:
            sweep_points.append((float(args.U_fixed), float(t_val), "vary_t"))

    if not sweep_points:
        raise ValueError(f"No sweep points generated for --scan-mode {args.scan_mode}")

    # ── Combine with topology / size / state / cluster axes ──────────────
    for L in args.L_values:
        n_e = L if args.n_electrons is None else args.n_electrons
        for mode in args.modes:
            for n_body in args.n_body_values:
                if not is_valid_config(L, mode, n_body, n_e):
                    continue
                clusters = generate_site_clusters(L, n_body)
                for U_val, t_val, sweep_tag in sweep_points:
                    ut_ratio = U_val / t_val if abs(t_val) > 1e-14 else float("inf")
                    for state_idx in range(args.k_states):
                        for sites in clusters:
                            cfg = {
                                "L":           int(L),
                                "t":           t_val,
                                "U":           U_val,
                                "U_over_t":    float(ut_ratio),
                                "mode":        mode,
                                "n_electrons": int(n_e),
                                "state_idx":   int(state_idx),
                                "sites":       list(sites),
                                "n_body":      int(n_body),
                                "sweep_tag":   sweep_tag,
                            }
                            cfg["config_hash"] = config_hash(cfg)
                            configs.append(cfg)
    return configs


def rename_existing_records(outdir: Path) -> None:
    """
    Rename hash-only .npz files in outdir/records/ to the human-readable
    scheme, and update the manifest.csv `npz_path` column accordingly.

    Idempotent: files already in the new format are left untouched.
    """
    manifest_path = outdir / "manifest.csv"
    records_dir   = outdir / "records"

    if not manifest_path.exists():
        print(f"  [rename] no manifest at {manifest_path} — nothing to do")
        return
    if not records_dir.is_dir():
        print(f"  [rename] no records dir at {records_dir} — nothing to do")
        return

    df = pd.read_csv(manifest_path)
    n_renamed, n_skipped, n_missing = 0, 0, 0

    for i, row in df.iterrows():
        # Re-build a config dict from the manifest row
        sites_field = row["sites"]
        if isinstance(sites_field, str):
            try:
                sites = json.loads(sites_field)
            except json.JSONDecodeError:
                sites = eval(sites_field)   # fallback for str-tuple legacy rows
        else:
            sites = list(sites_field)

        cfg = {
            "L":           int(row["L"]),
            "t":           float(row["t"]),
            "U":           float(row["U"]),
            "U_over_t":    float(row["U_over_t"]),
            "mode":        str(row["mode"]),
            "n_electrons": int(row["n_electrons"]),
            "state_idx":   int(row["state_idx"]),
            "sites":       sites,
            "n_body":      int(row["n_body"]),
            "config_hash": str(row["config_hash"]),
        }

        new_name = record_filename(cfg)
        new_path = records_dir / new_name
        new_relpath = str(new_path.relative_to(outdir))

        old_relpath = str(row["npz_path"])
        old_path    = outdir / old_relpath

        if old_relpath == new_relpath and old_path.exists():
            n_skipped += 1
            continue
        if not old_path.exists():
            n_missing += 1
            continue

        old_path.rename(new_path)
        df.at[i, "npz_path"] = new_relpath
        n_renamed += 1

    df.to_csv(manifest_path, index=False)
    print(f"  [rename] renamed: {n_renamed}   already-ok: {n_skipped}   "
          f"missing: {n_missing}")
    print(f"  [rename] manifest updated: {manifest_path}")


def process_diag_group(args_tuple) -> Tuple[List[Dict], List[str]]:
    """
    Process one diagonalization group and return (manifest_rows, log_lines).

    A "group" is one (L, t, U, mode, n_electrons) Hamiltonian and ALL the
    (state_idx, sites, n_body) configs that share it.

    This function is the unit of parallelism: each worker handles one group.

    args_tuple is unpacked rather than using kwargs so multiprocessing
    serialisation stays simple. Layout:

        (diag_key, cfg_list, outdir_str, records_dir_str,
         n_inner_restarts, n_outer_restarts,
         label_solver, gcg_max_iter, gcg_tol, gcg_k_max_extra,
         verbose)

    Returns
    -------
    rows : list of manifest-row dicts (one per record, ready for CSV append)
    logs : list of human-readable log lines for the parent to print
    """
    (diag_key, cfg_list, outdir_str, records_dir_str,
     n_inner_restarts, n_outer_restarts,
     label_solver, gcg_max_iter, gcg_tol, gcg_k_max_extra,
     verbose) = args_tuple

    outdir      = Path(outdir_str)
    records_dir = Path(records_dir_str)
    L, t_val, U, mode, n_e = diag_key
    rows: List[Dict] = []
    logs: List[str]  = []

    # Need at least the largest state_idx + 1 eigenstates
    k_needed = max(c["state_idx"] for c in cfg_list) + 1

    try:
        energies, eigvecs, n_so = compute_k_eigenstates(
            L=L, t=t_val, U=U, mode=mode, n_electrons=n_e, k=k_needed,
        )
    except Exception as e:
        logs.append(f"  [skip diag] L={L} U={U:.3f} t={t_val:.3f} mode={mode}: {e}")
        return rows, logs

    # Spin / double-occ observables per eigenstate (compute once)
    spin_info_list = classify_eigenstates_spin(eigvecs, energies, L, n_e, n_so)

    D_op = double_occupancy_operator(n_so, L)
    D_sp = get_number_preserving_sparse_operator(
        D_op, n_so, n_e, spin_preserving=True
    )

    state_obs: Dict[int, Dict[str, float]] = {}
    for s_info in spin_info_list:
        s_idx = s_info["idx"]
        psi   = eigvecs[:, s_idx]
        sz    = compute_spin_observables(psi, L, n_so, n_e)
        d_occ = float(np.vdot(psi, D_sp @ psi).real)
        state_obs[s_idx] = {
            "S2":             float(s_info["S2"]),
            "S":              float(s_info["S"]),
            "multiplicity":   float(s_info["multiplicity"]),
            "spin_label":     str(s_info["spin_label"]),
            "Sz":             sz,
            "d_occ":          d_occ,
            "d_occ_per_site": d_occ / L,
            "energy":         float(s_info["energy"]),
            "gap":            float(s_info["energy"] - spin_info_list[0]["energy"]),
        }

    # Per-config RDM + labels
    for cfg in cfg_list:
        s_idx = cfg["state_idx"]
        sites = tuple(cfg["sites"])
        n_body = cfg["n_body"]
        psi = eigvecs[:, s_idx]

        try:
            rho_raw = compute_general_nRDM(list(sites), psi, n_so, n_e)
        except Exception as e:
            logs.append(f"  [skip rdm] {cfg['config_hash']}: {e}")
            continue

        raw_trace = float(np.real(np.trace(rho_raw)))
        if abs(raw_trace) < RAW_TRACE_MIN:
            if verbose:
                logs.append(
                    f"  [skip rdm] {cfg['config_hash']}: raw_trace={raw_trace:.2e}"
                )
            continue
        rho = rho_raw / raw_trace

        seed_base = int(int(cfg["config_hash"], 16) % (2**31))
        t_lab0 = time.time()
        labels = compute_labels_for_rdm(
            rho, n_body,
            n_outer_restarts=n_outer_restarts,
            n_inner_restarts=n_inner_restarts,
            seed_base=seed_base,
            label_solver=label_solver,
            gcg_max_iter=gcg_max_iter,
            gcg_tol=gcg_tol,
            gcg_k_max_extra=gcg_k_max_extra,
        )
        label_time = time.time() - t_lab0

        eigvals = np.linalg.eigvalsh(rho)
        tr_rho2 = float(np.real(np.trace(rho @ rho)))
        tr_rho3 = float(np.real(np.trace(rho @ rho @ rho)))
        rank    = int(np.sum(eigvals > 1e-10))

        record = {
            **cfg,
            "U_over_t":   U / t_val if abs(t_val) > 1e-14 else float("inf"),
            "n_so":       int(n_so),
            **state_obs[s_idx],
            **labels,
            "raw_trace":  raw_trace,
            "tr_rho2":    tr_rho2,
            "tr_rho3":    tr_rho3,
            "rank":       rank,
            "rdm_eigvals": eigvals.tolist(),
            "label_time_s": label_time,
            "seed_base":  seed_base,
        }

        npz_path = records_dir / record_filename(cfg)
        np.savez_compressed(
            npz_path,
            rho_real=rho.real.astype(np.float32),
            rho_imag=rho.imag.astype(np.float32),
            eigvals=eigvals.astype(np.float64),
            config_json=np.array(json.dumps({
                k: v for k, v in cfg.items() if k != "config_hash"
            })),
            C_target=np.float64(record["C_target"]),
            C_pure=np.float64(record["C_pure"]),
            C_stiefel_best=np.float64(record["C_stiefel_best"]),
            C_stiefel_all=np.array(labels.get("C_stiefel_all", []), dtype=np.float64),
            C_gcg=np.float64(record["C_gcg"]),
            C_gcg_all=np.array(labels.get("C_gcg_all", []), dtype=np.float64),
            label_solver=np.array(record["label_solver"]),
            lower_bound_super=np.float64(record["lower_bound_super"]),
            C_UB=np.float64(record["C_UB"]),
            purity=np.float64(record["purity"]),
            is_pure=np.bool_(record["is_pure"]),
            target_in_bracket=np.bool_(record["target_in_bracket"]),
        )

        record["npz_path"] = str(npz_path.relative_to(outdir))
        rows.append(_to_manifest_row(record))

    return rows, logs


def append_to_json_log(path: Path, entry: Dict[str, Any]) -> None:
    """
    Append `entry` to a JSON log file at `path`.

    Behavior:
      - If file does not exist:  create with [entry]
      - If file contains a list: append entry, save
      - If file contains a dict (legacy single-run format): wrap it into
        a list, then append the new entry. Original run is preserved
        and gets a synthetic 'migrated': True tag for traceability.
      - On parse error: rename the bad file with a .bad suffix and start fresh
        (never silently lose data).

    Each entry gets a 'logged_at' timestamp so multiple runs can be told apart.
    """
    entry = {**entry, "logged_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    if not path.exists():
        with open(path, "w") as f:
            json.dump([entry], f, indent=2)
        return

    # Try to read existing content
    try:
        with open(path, "r") as f:
            existing = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        # Don't lose user data — quarantine the bad file
        bad_path = path.with_suffix(path.suffix + ".bad")
        path.rename(bad_path)
        print(f"  [warn] {path.name} unreadable ({e}); quarantined as {bad_path.name}")
        with open(path, "w") as f:
            json.dump([entry], f, indent=2)
        return

    # Migrate legacy single-dict format to list-of-dicts
    if isinstance(existing, dict):
        existing = [{**existing, "migrated": True}]

    if not isinstance(existing, list):
        # Unrecognised structure — quarantine and start fresh
        bad_path = path.with_suffix(path.suffix + ".bad")
        path.rename(bad_path)
        print(f"  [warn] {path.name} not a list or dict; quarantined as {bad_path.name}")
        existing = []

    existing.append(entry)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        prog="generate_ml_dataset",
        description="Build ML training dataset for Hubbard GME surrogate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    g_grid = parser.add_argument_group("Grid (what to scan)")
    g_grid.add_argument("--L", type=int, nargs="+", default=[4, 6],
                        dest="L_values",
                        help="Lattice size(s). Accepts one or more values "
                             "(default: 4 6). Use --L 4 for a single size, "
                             "--L 4 6 8 for multiple.")
    g_grid.add_argument("--modes", type=str, nargs="+",
                        default=["uniform", "open", "nnn", "ssh", "ladder"],
                        choices=list(VALID_MODES),
                        help="Topologies to include")
    g_grid.add_argument("--n-body-values", type=int, nargs="+", default=[2, 3, 4],
                        help="Body orders (default: 2 3 4)")
    g_grid.add_argument("--k-states", type=int, default=2,
                        help="Eigenstates per (L,U,mode) (default: 2)")
    g_grid.add_argument("--scan-mode", type=str, default="vary_U",
                        choices=["vary_U", "vary_t", "both"],
                        help="Which axis to sweep:  "
                             "vary_U = vary U at fixed t (default);  "
                             "vary_t = vary t at fixed U;  "
                             "both = run both sweeps (~2x data).")
    g_grid.add_argument("--n-ut", type=int, default=30,
                        help="U/t grid size for vary_U mode (default: 30)")
    g_grid.add_argument("--ut-min", type=float, default=0.1,
                        help="U/t minimum for vary_U mode (default: 0.1)")
    g_grid.add_argument("--ut-max", type=float, default=20.0,
                        help="U/t maximum for vary_U mode (default: 20.0)")
    g_grid.add_argument("--t", type=float, default=1.0,
                        help="Fixed t for vary_U mode (default: 1.0)")

    g_grid.add_argument("--n-t", type=int, default=30,
                        help="t grid size for vary_t mode (default: 30)")
    g_grid.add_argument("--t-min", type=float, default=0.2,
                        help="t minimum for vary_t mode (default: 0.2)")
    g_grid.add_argument("--t-max", type=float, default=10.0,
                        help="t maximum for vary_t mode (default: 10.0)")
    g_grid.add_argument("--U-fixed", type=float, default=4.0,
                        help="Fixed U for vary_t mode (default: 4.0)")

    g_grid.add_argument("--n-electrons", type=int, default=None,
                        help="Override electron count (default: half-filling)")

    g_lab = parser.add_argument_group("Labeling")
    g_lab.add_argument("--label-solver", type=str, default="gcg",
                       choices=["gcg", "stiefel"],
                       help="Convex-roof solver for mixed-state labels. "
                            "Default: gcg (verified best). 'stiefel' = legacy v1.")
    g_lab.add_argument("--gcg-max-iter", type=int, default=200,
                       help="GCG iterations per restart (default 200, matches "
                            "the thermal pipeline settings)")
    g_lab.add_argument("--gcg-tol", type=float, default=1e-9,
                       help="GCG convergence tolerance (default 1e-9)")
    g_lab.add_argument("--gcg-k-max-extra", type=int, default=4,
                       help="GCG extra decomposition ranks (default 4)")
    g_lab.add_argument("--n-inner-restarts", type=int, default=20,
                       help="Stiefel internal random restarts per call "
                            "(default: 20 — matches convex_roof_stiefel default)")
    g_lab.add_argument("--n-outer-restarts", type=int, default=1,
                       help="Independent Stiefel calls per RDM. >1 gives a "
                            "label-noise estimate (C_stiefel_std). Default 1.")

    g_io = parser.add_argument_group("I/O")
    g_io.add_argument("--outdir", type=str,
                      default=os.path.join(_THIS_DIR, "data_for_ML"),
                      help="Dataset directory (default: <gme_qml>/data_for_ML)")
    g_io.add_argument("--fresh", action="store_true",
                      help="Wipe existing manifest (default: resume)")
    g_io.add_argument("--max-records", type=int, default=None,
                      help="Stop after this many NEW records (useful for testing)")
    g_io.add_argument("--rename-existing", action="store_true",
                      help="Rename existing hash-only .npz files in --outdir to "
                           "the human-readable scheme, update manifest.csv, "
                           "then exit. No new records are computed.")
    g_io.add_argument("--n-workers", type=int, default=1,
                      help="Number of parallel worker processes. "
                           "1 (default) = serial. "
                           "0 = auto (use cpu_count() - 1). "
                           "N>1 = use exactly N workers.")
    g_io.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # ── Setup output directory ────────────────────────────────────────────
    outdir       = Path(args.outdir).resolve()
    records_dir  = outdir / "records"
    manifest_path = outdir / "manifest.csv"
    records_dir.mkdir(parents=True, exist_ok=True)

    # ── --rename-existing: rename hash-only files and exit ────────────────
    if args.rename_existing:
        print(f"Renaming existing records in {outdir} ...")
        rename_existing_records(outdir)
        return

    if args.fresh and manifest_path.exists():
        manifest_path.unlink()
        for f in records_dir.glob("*.npz"):
            f.unlink()

    # ── Build run-provenance dicts (logged after we confirm there's work) ─
    grid_summary = {
        "scan_mode":          args.scan_mode,
        "L_values":           args.L_values,
        "modes":               args.modes,
        "n_body_values":       args.n_body_values,
        "k_states":            args.k_states,
        # vary_U params
        "n_ut":                args.n_ut,
        "ut_min":              args.ut_min,
        "ut_max":              args.ut_max,
        "t_fixed":             args.t,
        "ut_grid":             stratified_ut_grid(args.n_ut, args.ut_min, args.ut_max).tolist()
                                  if args.scan_mode in ("vary_U", "both") else None,
        # vary_t params
        "n_t":                 args.n_t,
        "t_min":               args.t_min,
        "t_max":               args.t_max,
        "U_fixed":             args.U_fixed,
        "t_grid":              stratified_t_grid(args.n_t, args.t_min, args.t_max,
                                                  args.U_fixed).tolist()
                                  if args.scan_mode in ("vary_t", "both") else None,
        # general
        "n_electrons":         args.n_electrons,
        "n_inner_restarts":    args.n_inner_restarts,
        "n_outer_restarts":    args.n_outer_restarts,
        # labeling solver
        "label_solver":        args.label_solver,
        "gcg_max_iter":        args.gcg_max_iter,
        "gcg_tol":             args.gcg_tol,
        "gcg_k_max_extra":     args.gcg_k_max_extra,
    }

    run_meta = {
        "started":      time.strftime("%Y-%m-%d %H:%M:%S"),
        "python":       sys.version.split()[0],
        "numpy":        np.__version__,
        "pandas":       pd.__version__,
        "outdir":       str(outdir),
        "command":      " ".join(sys.argv),
        "n_workers":    args.n_workers,
    }

    # ── Build full grid and skip already-processed configs ────────────────
    print("Building config grid ...", flush=True)
    all_configs = build_config_grid(args)
    done_hashes = load_done_hashes(manifest_path)
    todo = [c for c in all_configs if c["config_hash"] not in done_hashes]

    print(f"  Total configs:   {len(all_configs)}")
    print(f"  Already done:    {len(done_hashes)}")
    print(f"  To process now:  {len(todo)}")
    if args.max_records is not None:
        todo = todo[: args.max_records]
        print(f"  Capped at:       {len(todo)} (--max-records)")

    # ── No-op exit (don't pollute provenance logs with "did nothing" runs) ─
    if not todo:
        print("Nothing to do — all configs already processed.")
        print("(provenance logs not updated for no-op runs)")
        return

    # ── Now we know we'll do real work — log this run's provenance ────────
    grid_summary["n_configs_total"]   = len(all_configs)
    grid_summary["n_configs_already"] = len(done_hashes)
    grid_summary["n_configs_todo"]    = len(todo)
    append_to_json_log(outdir / "grid_summary.json", grid_summary)
    append_to_json_log(outdir / "run_metadata.json", run_meta)

    # ── Group by Hamiltonian so we diagonalize once per (L, t, U, mode) ──
    diag_groups: Dict[Tuple, List[Dict]] = defaultdict(list)
    for cfg in todo:
        key = (cfg["L"], cfg["t"], cfg["U"], cfg["mode"], cfg["n_electrons"])
        diag_groups[key].append(cfg)

    # ── Build per-group work tuples (same for serial & parallel) ─────────
    group_items = list(diag_groups.items())
    work_args = [
        (
            diag_key, cfg_list,
            str(outdir), str(records_dir),
            args.n_inner_restarts, args.n_outer_restarts,
            args.label_solver, args.gcg_max_iter,
            args.gcg_tol, args.gcg_k_max_extra,
            args.verbose,
        )
        for (diag_key, cfg_list) in group_items
    ]

    # ── Decide serial vs parallel ────────────────────────────────────────
    n_workers = args.n_workers
    if n_workers == 0:
        n_workers = max(1, cpu_count() - 1)   # auto: leave one core free
    n_workers = min(n_workers, len(work_args))

    new_rows: List[Dict] = []
    n_done = 0
    t_global_start = time.time()

    if n_workers <= 1:
        # ── Serial path (default, simpler logs) ──────────────────────────
        print(f"  Running SERIAL (1 worker)")
        for diag_idx, work in enumerate(work_args):
            rows, logs = process_diag_group(work)
            for line in logs:
                print(line, flush=True)
            new_rows.extend(rows)
            n_done += len(rows)

            if len(new_rows) >= FLUSH_EVERY_N_RECORDS:
                append_manifest(new_rows, manifest_path)
                new_rows = []

            diag_key = work[0]
            L, t_val, U, mode, n_e = diag_key
            elapsed = time.time() - t_global_start
            rate = n_done / max(elapsed, 1e-6)
            print(f"  [{diag_idx+1}/{len(work_args)}] L={L} U={U:>6.3f} "
                  f"t={t_val:>6.3f} mode={mode:<8s}  done={n_done:5d}  "
                  f"rate={rate:5.2f} rec/s", flush=True)
    else:
        # ── Parallel path ────────────────────────────────────────────────
        print(f"  Running PARALLEL with {n_workers} workers "
              f"(of {cpu_count()} cores available)")
        completed = 0
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(process_diag_group, w): w for w in work_args}
            for fut in as_completed(futures):
                w = futures[fut]
                diag_key = w[0]
                L, t_val, U, mode, _ = diag_key
                try:
                    rows, logs = fut.result()
                except Exception as e:
                    print(f"  [worker error] L={L} U={U:.3f} t={t_val:.3f} "
                          f"mode={mode}: {e}", flush=True)
                    completed += 1
                    continue

                for line in logs:
                    print(line, flush=True)
                new_rows.extend(rows)
                n_done += len(rows)
                completed += 1

                # Flush manifest periodically (parent only writes — no race)
                if len(new_rows) >= FLUSH_EVERY_N_RECORDS:
                    append_manifest(new_rows, manifest_path)
                    new_rows = []

                elapsed = time.time() - t_global_start
                rate = n_done / max(elapsed, 1e-6)
                print(f"  [{completed}/{len(work_args)}] L={L} U={U:>6.3f} "
                      f"t={t_val:>6.3f} mode={mode:<8s}  done={n_done:5d}  "
                      f"rate={rate:5.2f} rec/s", flush=True)

    # Final flush
    append_manifest(new_rows, manifest_path)

    print(f"\nFinished. {n_done} new records.")
    print(f"  Manifest:  {manifest_path}")
    print(f"  Records:   {records_dir}")
    print(f"  Wall time: {time.time() - t_global_start:.1f} s")


if __name__ == "__main__":
    main()