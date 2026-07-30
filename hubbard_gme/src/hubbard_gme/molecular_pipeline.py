"""
molecular_pipeline.py
---------------------
End-to-end driver for molecular GME analysis.

Public API
----------
run_molecular_analysis(mol_name, mol_data, cas, ...)
    One molecule + one CAS active space.

run_all_molecules_analysis(mol_subset=None, ...)
    Loop over molecules and active spaces.

Convex-roof methods (chosen via `cr_method`):
    'none'    — skip convex roof entirely (RDMs + metrics only)
    'expo'    — convex_roof_numerical (exp-map unitary optimisation)
    'stiefel' — convex_roof_stiefel   (Stiefel / Euler–Hurwitz Alg. B)
    'both'    — expo + stiefel   (recommended, matches the user's request)
    'full'    — full_convex_roof_analysis (all methods: paper/supervisor
                lower bounds, upper bound, expo, Cayley, Stiefel)

Output layout
-------------
By default PDFs go to:
    <output_dir>/<MolName>/RDM_<k>body_<MolName>_CAS<Ne><Norb>.pdf
    <output_dir>/<MolName>/GME_<k>body_<MolName>_CAS<Ne><Norb>.pdf

where the RDM PDF is the cheap RDM-only report and the GME PDF contains
the convex-roof analysis.
"""

from __future__ import annotations

import os
import pickle
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from hubbard_gme.rdm import get_rdm_pkl_results_dir, _basic_env_meta

from hubbard_gme.molecular import (
    build_molecular_hamiltonian,
    compute_molecular_ground_state,
    compute_molecular_k_eigenstates,
    compute_molecular_spin_docc,
    compute_all_molecular_rdms,
    compute_molecular_nrdm,
    build_metrics_dict,
)
from hubbard_gme.molecular_visualization import (
    generate_molecular_gme_pdf,
    generate_molecular_rdm_only_pdf,
)
from hubbard_gme.molecules_data import MOLECULES
from hubbard_gme.convex_roof import (
    convex_roof_numerical,
    convex_roof_stiefel,
    convex_roof_gcg,
    convex_roof_lower_bound,
    convex_roof_lower_bound_supervisor,
    convex_roof_upper_bound,
    full_convex_roof_analysis,
    C_GME_pure_dm,
)
from hubbard_gme.metrics import compute_purity


# ──────────────────────────────────────────────────────────────────────────────
# Convex-roof dispatcher
# ──────────────────────────────────────────────────────────────────────────────

_VALID_CR_METHODS = ("none", "expo", "stiefel", "gcg", "both", "all", "full")


def _nan(): return float("nan")


def run_convex_roof_on_rdm(
    rho: np.ndarray,
    sites: tuple,
    method: str = "both",
    n_restarts: int = 20,
    max_iter: int = 200,
    k_max_extra: int = 2,
    tol: float = 1e-9,
    verbose: bool = False,
) -> Optional[dict]:
    """
    Run the selected convex-roof methods on a single RDM and return a result
    dict compatible with the keys consumed by the existing visualisation
    code (hubbard_gme.visualization._write_combined_page).

    Returns None if method == 'none'.
    """
    if method == "none":
        return None
    if method not in _VALID_CR_METHODS:
        raise ValueError(
            f"Unknown cr_method '{method}'. Choose from {_VALID_CR_METHODS}."
        )

    n = len(sites)

    if method == "full":
        # Delegate to the existing unified analysis (runs everything).
        return full_convex_roof_analysis(
            rho, sites,
            n_restarts=n_restarts,
            use_de=False,
            use_cvxpy=False,
            verbose=verbose,
        )

    # For 'expo', 'stiefel', 'both': always include the cheap lower/upper
    # bounds because they are essentially free and make the summary table
    # more informative.
    lb      = convex_roof_lower_bound(rho, n)
    lb_sup  = convex_roof_lower_bound_supervisor(rho, n)
    ub      = convex_roof_upper_bound(rho, n)

    result = {
        "sites":              sites,
        "n":                  n,
        "global_purity":      float(compute_purity(rho)),
        "lower_bound":        lb,
        "lower_bound_super":  lb_sup,
        "upper_bound":        ub,
        "C_LB":               float(lb["C_LB"]),
        "C_LB_supervisor":    float(lb_sup["C_LB"]),
        "C_LB_best":          float(max(lb["C_LB"], lb_sup["C_LB"])),
        "C_UB":               float(ub["C_UB"]),
        # The method-specific entries start as NaN; filled below.
        "C_expo":             _nan(),
        "C_stiefel":          _nan(),
        "C_gcg":              _nan(),
        "C_cayl":             _nan(),
        "C_sdp":              None,
        "numerical":          {},
        "stiefel":            {},
        "gcg":                {},
        "decompositions":     {},
        "decomp":             [],
        "C_GME":              C_GME_pure_dm(rho, n),
    }

    # Exp-map unitary optimisation
    if method in ("expo", "both", "all"):
        num = convex_roof_numerical(rho, n, n_restarts=n_restarts)
        result["numerical"] = num
        result["C_expo"]    = float(num["C_numerical"])
        result["decomp"]    = num.get("decomp", [])

    # Stiefel manifold optimisation (Rothlisberger Alg. B)
    if method in ("stiefel", "both", "all"):
        stf = convex_roof_stiefel(
            rho, n, n_restarts=n_restarts, verbose=verbose,
        )
        result["stiefel"]    = stf
        result["C_stiefel"]  = float(stf["C_stiefel"])
        if not result["decomp"]:
            # Fall back to the stiefel decomposition if expo wasn't run.
            result["decomp"] = stf.get("best_record", {}).get("decomp", [])

    # GCG convex roof (Generalised Conjugate Gradient) — the primary solver,
    # returns per-k / per-restart convergence records (k_records, restart_log).
    if method in ("gcg", "all"):
        gcg = convex_roof_gcg(rho, n, n_restarts=n_restarts, max_iter=max_iter,
                              k_max_extra=k_max_extra, tol=tol, verbose=verbose)
        result["gcg"]            = gcg
        result["C_gcg"]          = float(gcg["C_gcg"])
        result["decompositions"]["gcg"] = gcg.get("k_records", [])
        if not result["decomp"]:
            result["decomp"] = gcg.get("best_record", {}).get("decomp", [])

    # Status classification (best of whichever solvers ran).
    # The verdict is based on the convex-roof value alone: GCG is the accurate
    # estimator of C_GME, whereas the analytic lower bound is a weak witness
    # that reads 0 for many genuinely entangled states — it must not downgrade
    # a positive C_gcg to "inconclusive".
    tol = 1e-5
    c_best_num = np.nanmin([
        x for x in (result["C_expo"], result["C_stiefel"], result["C_gcg"])
        if not np.isnan(x)
    ] or [_nan()])

    if np.isnan(c_best_num):
        result["status"] = "failed"
    elif c_best_num < tol:
        result["status"] = "no_GME"
    else:
        result["status"] = "GME"

    if verbose:
        print(f"    sites={sites}  "
              f"C_LB={result['C_LB_best']:.6f}  "
              f"C_gcg={result['C_gcg']:.6f}  "
              f"C_stf={result['C_stiefel']:.6f}  "
              f"C_UB={result['C_UB']:.6f}  "
              f"status={result['status']}")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# PKL persistence — save every molecular GME result like the standalone Hubbard
# gme_runs/ caches: self-contained (RDM included) + result_full + per-k/per-restart
# decompositions.
# ──────────────────────────────────────────────────────────────────────────────

_MOL_META_KEYS = (
    "formula", "basis", "charge", "multiplicity", "total_electrons",
    "ncore", "nele_cas", "norb_cas", "n_so", "hilbert_dim", "hf_energy",
    "E0", "E1", "gap", "S2", "S", "Sz", "spin_label",
    "double_occ", "double_occ_per_site", "cr_methods", "cr_restarts",
    # provenance for external-CI runs (absent/ignored for diagonalised runs)
    "wavefunction_source", "psi_source",
)
_GCG_SETTING_KEYS = (
    "k_converged", "converged", "completed_successfully", "measure",
    "n_restarts_requested", "n_restarts_effective", "k_max_extra",
    "tol", "tol_k", "max_iter", "max_iter_by_k", "elapsed_seconds",
    "elapsed_by_k", "success_by_k", "optimizer_success_by_k",
)


def _mol_gme_pkl_dir(mol_name: str) -> Path:
    d = get_rdm_pkl_results_dir() / "mol_gme_runs" / mol_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cr_settings_tag(cr_method: str, cr_restarts: int, cr_max_iter: int,
                     cr_k_max_extra: int) -> str:
    """Solver-settings fingerprint embedded in the PKL name (Hubbard convention),
    so a re-run with the SAME settings reuses the cache and a re-run with
    DIFFERENT settings recomputes automatically."""
    return (f"{cr_method}_restarts{int(cr_restarts)}_iter{int(cr_max_iter)}"
            f"_kmax{int(cr_k_max_extra)}")


def _save_molecular_gme_pkl(
    *,
    mol_name: str,
    cas_tag: str,
    sites: tuple,
    rho: np.ndarray,
    cr_result: dict,
    mol_info: dict,
    metrics_entry: dict,
    cache_version: str = "v1",
    cr_tag: str = "",
) -> str:
    """Save one (molecule, CAS, cluster) GME result to a self-contained PKL."""
    n = len(sites)
    sites_str = "-".join(str(s) for s in sites)
    _t = f"{cr_tag}_" if cr_tag else ""
    fname = (f"molgme_{cache_version}_{mol_name}_{cas_tag}_"
             f"{_t}sites-{sites_str}_n{n}.pkl")
    path = _mol_gme_pkl_dir(mol_name) / fname

    eig = np.linalg.eigvalsh(rho)
    gcg = cr_result.get("gcg") if isinstance(cr_result.get("gcg"), dict) else {}

    payload = {
        "meta": {
            "object_type": "hubbard_molecular_gme",
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "cache_version": cache_version,
            "mol_name": mol_name,
            "cas_tag": cas_tag,
            **{k: mol_info.get(k) for k in _MOL_META_KEYS},
            "sites": list(sites),
            "n": int(n),
            "environment": _basic_env_meta(),
        },
        "rho": {
            "rho_real": rho.real.astype(np.float64),
            "rho_imag": rho.imag.astype(np.float64),
            "eigvals": eig,
            "purity": float(metrics_entry.get("purity",
                            float(np.real(np.trace(rho @ rho))))),
            "entropy": float(metrics_entry.get("entropy", float("nan"))),
            "rank": int(metrics_entry.get("rank", int(np.sum(eig > 1e-10)))),
            "state_type": metrics_entry.get("state_type", ""),
        },
        "result_full": {
            "sites": tuple(sites), "n": int(n),
            "C_gcg": cr_result.get("C_gcg"),
            "C_expo": cr_result.get("C_expo"),
            "C_stiefel": cr_result.get("C_stiefel"),
            "C_GME": cr_result.get("C_GME"),
            "C_LB": cr_result.get("C_LB"),
            "C_LB_supervisor": cr_result.get("C_LB_supervisor"),
            "C_LB_best": cr_result.get("C_LB_best"),
            "C_UB": cr_result.get("C_UB"),
            "global_purity": cr_result.get("global_purity"),
            "status": cr_result.get("status"),
            **{k: gcg.get(k) for k in _GCG_SETTING_KEYS},
        },
        # per-k records, each with restart_log / decomp (same detail as gme_runs/)
        "decompositions": cr_result.get("decompositions", {}),
    }

    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return str(path)


def _load_mol_gme_result(mol_name: str, cas_tag: str, sites: tuple,
                         cache_version: str = "v1",
                         cr_tag: str = "") -> Optional[dict]:
    """Reload a previously-saved cluster GME result (its ``result_full``) so a
    re-run can reuse instead of recomputing. The ``cr_tag`` (solver settings)
    is part of the filename, so only a result computed with the SAME settings is
    reused. Returns None if not on disk. ``result_full`` carries every key the
    summary and GME PDF read (C_gcg / C_LB / C_LB_best / C_UB / C_expo /
    status)."""
    n = len(sites)
    sites_str = "-".join(str(s) for s in sites)
    _t = f"{cr_tag}_" if cr_tag else ""
    path = _mol_gme_pkl_dir(mol_name) / (
        f"molgme_{cache_version}_{mol_name}_{cas_tag}_"
        f"{_t}sites-{sites_str}_n{n}.pkl")
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        rf = payload.get("result_full")
        return dict(rf) if isinstance(rf, dict) else None
    except Exception:
        return None


def _cluster_gme_summary(cr_results: dict, norb: int) -> None:
    """Print mean/spread of C_gcg per body order, and per ring-symmetry class
    (cyclic gap pattern). Clusters in the same class are symmetry-equivalent and
    should give the SAME C_GME — a non-zero spread flags solver noise, not
    physics. (The gap pattern assumes a ring; it is a harmless grouping for
    non-ring systems.)"""
    from collections import defaultdict

    def _shape(sites):
        s = sorted(sites)
        gaps = sorted((s[(i + 1) % len(s)] - s[i]) % norb for i in range(len(s)))
        return tuple(gaps)

    by_n, by_cls = defaultdict(list), defaultdict(list)
    for sites, res in cr_results.items():
        if res is None:
            continue
        c = res.get("C_gcg")
        if c is None or (isinstance(c, float) and np.isnan(c)):
            continue
        by_n[len(sites)].append(float(c))
        by_cls[(len(sites), _shape(sites))].append(float(c))
    if not by_n:
        return

    print("\n  ── C_GME summary (GCG convex roof) ──")
    print(f"  {'n':>3} {'#':>4} {'mean':>11} {'min':>11} {'max':>11} {'std':>10}")
    for n in sorted(by_n):
        v = np.array(by_n[n])
        print(f"  {n:>3} {len(v):>4} {v.mean():>11.6f} {v.min():>11.6f} "
              f"{v.max():>11.6f} {v.std():>10.2e}")
    print("\n  per ring-symmetry class (gap pattern) — same class should match:")
    print(f"  {'n':>3} {'gaps':>14} {'#':>3} {'mean':>11} {'spread':>10}")
    for key in sorted(by_cls):
        n, sig = key
        v = np.array(by_cls[key])
        print(f"  {n:>3} {str(sig):>14} {len(v):>3} {v.mean():>11.6f} "
              f"{(v.max() - v.min()):>10.2e}")


# ──────────────────────────────────────────────────────────────────────────────
# mol_info builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_molecular_info(
    mol_name: str,
    mol_data: dict,
    cas: dict,
    basis: str,
    n_so: int,
    hilbert_dim: int,
    hf_energy: float,
    E0: float,
    E1: float,
    S2: float,
    S_val: float,
    Sz: float,
    spin_label: str,
    D_total: float,
    cr_method: str,
    cr_restarts: int,
) -> dict:
    """Assemble the mol_info dict consumed by the PDF writers."""
    norb_cas = cas["norb_cas"]
    return {
        "name":                mol_name,
        "formula":             mol_data.get("formula", mol_name),
        "basis":               basis,
        "charge":              mol_data.get("charge", 0),
        "multiplicity":        mol_data.get("multiplicity", 1),
        "total_electrons":    (mol_data.get("total_electrons")
                               or mol_data.get("Total Electrons", "?")),
        "ncore":               cas["ncore"],
        "nele_cas":            cas["nele_cas"],
        "norb_cas":            norb_cas,
        "n_so":                n_so,
        "hilbert_dim":         hilbert_dim,
        "hf_energy":           hf_energy,
        "E0":                  E0,
        "E1":                  E1,
        "gap":                 E1 - E0,
        "S2":                  S2,
        "S":                   S_val,
        "Sz":                  Sz,
        "spin_label":          spin_label,
        "double_occ":          D_total,
        "double_occ_per_site": D_total / norb_cas if norb_cas else 0.0,
        "cr_methods":          cr_method,
        "cr_restarts":         cr_restarts,
    }


# ──────────────────────────────────────────────────────────────────────────────
# External CI wavefunction loader
# Reconstruct a state vector from a Slater-determinant file (already in a
# localised basis), so run_molecular_analysis can analyse it via psi_external.
# ──────────────────────────────────────────────────────────────────────────────

# occupation character -> electrons on that spatial orbital
_OCC_ELECTRONS = {"0": 0, "a": 1, "b": 1, "2": 2}


def _occ_to_orbitals(occ: str) -> set:
    """Occupation string -> set of occupied spin-orbitals (index 2*site+spin,
    spin 0 = up, 1 = down — matching _precompute_mol_ops)."""
    orbitals: set = set()
    for k, ch in enumerate(occ):
        if ch == "a":
            orbitals.add(2 * k)
        elif ch == "b":
            orbitals.add(2 * k + 1)
        elif ch == "2":
            orbitals.add(2 * k)
            orbitals.add(2 * k + 1)
        # '0' -> nothing
    return orbitals


def load_ci_wavefunction(
    ci_file: str,
    norb: int = 6,
    ne: Optional[int] = None,
    *,
    spin_preserving: bool = True,
    s2_tol: float = 1e-4,
    verbose: bool = True,
) -> np.ndarray:
    """
    Reconstruct the state vector ``psi`` from an external Slater-determinant file.

    File format: one determinant per line, ``<index> <coefficient> <occ-string>``,
    where the occupation string has one character per spatial orbital
    (``0`` empty, ``a`` single up, ``b`` single down, ``2`` doubly occupied).

    The reconstruction discovers openfermion's basis ordering by reading the
    diagonals of the number operators (Hamiltonian-free — the basis is fixed),
    then drops each coefficient into its slot. It is validated by ``<S^2> ~ 0``
    (a singlet): a non-zero value means the file's determinant-ordering / sign
    convention does not match openfermion's, and the load is rejected rather
    than returning a silently-wrong state.

    Returns the normalised state vector (length = Sz=0 basis dimension).
    """
    from openfermion.ops import FermionOperator
    from openfermion.linalg import get_number_preserving_sparse_operator

    ne = ne if ne is not None else norb
    n_so = 2 * norb

    # ── parse: (coefficient, occupation string) per line ────────────────────
    data: List = []
    with open(ci_file, "r", encoding="utf-8") as fh:
        for line in fh:
            cols = line.split()
            if len(cols) < 3:
                continue
            data.append((float(cols[1]), cols[2]))
    if not data:
        raise ValueError(f"no (coeff, occupation) rows parsed from {ci_file!r}")

    # ── validate the raw file: electron count, Sz=0, normalisation ──────────
    for _c, occ in data:
        n_e = sum(_OCC_ELECTRONS[ch] for ch in occ)
        if n_e != ne:
            raise ValueError(
                f"determinant {occ!r} has {n_e} electrons, expected {ne}")
        if occ.count("a") != occ.count("b"):
            raise ValueError(f"determinant {occ!r} is not in the Sz=0 sector")
    norm_sq = sum(c * c for c, _ in data)

    # ── discover openfermion's slot ordering via the number operators ───────
    diags = []
    dim = None
    for p in range(n_so):
        n_p = get_number_preserving_sparse_operator(
            FermionOperator(((p, 1), (p, 0))), n_so, ne,
            spin_preserving=spin_preserving)
        diags.append(np.rint(np.real(n_p.diagonal())).astype(int))
        dim = n_p.shape[0]

    config_to_index = {}
    for i in range(dim):
        occupied = frozenset(p for p in range(n_so) if diags[p][i] == 1)
        config_to_index[occupied] = i

    # ── assemble psi: drop each coefficient into its slot ───────────────────
    psi = np.zeros(dim)
    for coeff, occ in data:
        key = frozenset(_occ_to_orbitals(occ))
        slot = config_to_index.get(key)
        if slot is None:
            raise KeyError(
                f"determinant {occ!r} (orbitals {sorted(key)}) not in the "
                f"openfermion basis — check norb/ne or the file")
        psi[slot] = coeff
    psi = psi / np.linalg.norm(psi)

    # ── validate the reconstruction: singlet, so <S^2> ~ 0 ──────────────────
    from openfermion.hamiltonians import s_squared_operator
    S2_op = get_number_preserving_sparse_operator(
        s_squared_operator(norb), n_so, ne, spin_preserving=spin_preserving)
    s2 = float(np.real(np.vdot(psi, S2_op @ psi)))

    if verbose:
        print(f"  [CI load] {len(data)} determinants, dim={dim}, "
              f"sum c^2={norm_sq:.6f}, <S^2>={s2:.2e}")
    if abs(s2) > s2_tol:
        raise ValueError(
            f"reconstructed state has <S^2>={s2:.4f} (expected ~0 for a "
            f"singlet). The external file's determinant-ordering / sign "
            f"convention does not match openfermion's — aborting.")

    return psi


# ──────────────────────────────────────────────────────────────────────────────
# Single molecule / single CAS
# ──────────────────────────────────────────────────────────────────────────────

def _cr_worker(args):
    """Top-level (picklable) worker for the parallel convex-roof loop: run the
    convex roof on ONE cluster's RDM and return (sites, result, error)."""
    rho, sites, method, n_restarts, max_iter, k_max_extra = args
    try:
        res = run_convex_roof_on_rdm(
            rho, sites, method=method, n_restarts=n_restarts,
            max_iter=max_iter, k_max_extra=k_max_extra, verbose=False)
        return sites, res, None
    except Exception as ex:                      # pragma: no cover - defensive
        return sites, None, f"{type(ex).__name__}: {ex}"


def run_molecular_analysis(
    mol_name: str,
    mol_data: dict,
    cas: dict,
    basis: str = "cc-pVDZ",
    max_n_body: int = 3,
    min_n_body: int = 1,
    output_dir: str = "results/mol_results",
    cr_method: str = "gcg",
    cr_restarts: int = 20,
    cr_max_iter: int = 200,
    cr_k_max_extra: int = 2,
    generate_pdf: bool = True,
    generate_rdm_pdf: bool = True,
    save_pkl: bool = True,
    verbose: bool = True,
    localize: Optional[str] = None,
    psi_external: Optional[np.ndarray] = None,
    external_tag: str = "extCI",
    psi_source: Optional[str] = None,
    n_workers: int = 1,
    resume: bool = True,
) -> dict:
    """
    Full molecular pipeline for ONE molecule + ONE CAS active space.

    psi_external : if given, skip the Hamiltonian build + diagonalisation and
                   use this state vector directly (e.g. an external CI
                   wavefunction loaded by ``load_ci_wavefunction``). Everything
                   downstream — RDMs, convex roof, PDFs, PKLs — is identical;
                   only the energies are unknown (NaN). Output is tagged
                   ``_<external_tag>`` so it never collides with diagonalised
                   runs.

    localize : None (canonical orbitals) or 'boys'/'pm'/'ibo' to compute the
               RDMs/GME in a LOCALIZED active-orbital basis (same ground-state
               energy; different, spatially-meaningful entanglement).

    Steps
    -----
    1. Build active-space Hamiltonian via OpenFermion / PySCF.
    2. Diagonalise for E0, E1, psi0.
    3. Compute spin observables and double occupancy.
    4. Compute all k-body RDMs for k = min_n_body ... max_n_body.
    5. Optionally run convex-roof analysis on each RDM (k >= 3 by default).
    6. Generate one PDF per n-body order.

    Parameters
    ----------
    cr_method : see module docstring
    cr_restarts : random restarts for numerical / Stiefel optimisation

    Returns
    -------
    dict with keys:
        mol_info, rdm_dict, metrics, cr_results, pdf_paths
    """
    if cr_method not in _VALID_CR_METHODS:
        raise ValueError(
            f"cr_method must be one of {_VALID_CR_METHODS}, got {cr_method!r}"
        )

    cas_tag = f"CAS{cas['nele_cas']}{cas['norb_cas']}"
    if localize:
        cas_tag += f"_loc{localize}"
    if psi_external is not None:
        cas_tag += f"_{external_tag}"

    if verbose:
        print(f"\n  {'─' * 60}")
        print(f"  {mol_name}   CAS({cas['nele_cas']},{cas['norb_cas']})  "
              f"ncore={cas['ncore']}   basis={basis}"
              + (f"   [{localize}-localized orbitals]" if localize else ""))
        print(f"  {'─' * 60}")

    if psi_external is not None:
        # ── 1-2. External wavefunction: skip Hamiltonian build + diagonalise ─
        # Use the supplied state vector directly. Only the energies are unknown
        # (NaN); norb/ne come from the (synthetic) cas, and everything below is
        # identical to the diagonalised path.
        psi0 = np.asarray(psi_external)
        norb_cas = cas["norb_cas"]; ne = cas["nele_cas"]; n_so = 2 * norb_cas
        hilbert_dim = int(psi0.shape[0])
        hf_energy = E0 = E1 = float("nan")
        if verbose:
            print(f"  external psi supplied (dim={hilbert_dim}) — "
                  f"skipping Hamiltonian build/diagonalisation")
    else:
        # ── 1. Hamiltonian ──────────────────────────────────────────────────
        if localize:
            from .molecular_localize import build_localized_molecular_hamiltonian
            _li = build_localized_molecular_hamiltonian(
                mol_data, cas, basis=basis, method=localize)
            H_sparse = _li["H_sparse"]; hf_energy = _li["hf_energy"]
            n_so = _li["n_so"]; ne = _li["ne"]; norb_cas = _li["norb"]
            hilbert_dim = H_sparse.shape[0]
        else:
            H_sparse, hf_energy, n_so, ne, norb_cas, hilbert_dim = \
                build_molecular_hamiltonian(mol_data, cas, basis=basis)

        if verbose:
            print(f"  HF energy   = {hf_energy:.6f} Ha")
            print(f"  H shape     = {H_sparse.shape}")

        # ── 2. Ground state ─────────────────────────────────────────────────
        E0, E1, psi0 = compute_molecular_ground_state(H_sparse, k=2)
        if verbose:
            print(f"  E0 = {E0:.8f}  E1 = {E1:.8f}  gap = {E1 - E0:.8f}")

    # ── 3. Spin + D ─────────────────────────────────────────────────────────
    try:
        S2, S_val, Sz, spin_label, D_total = \
            compute_molecular_spin_docc(psi0, norb_cas, n_so, ne)
    except Exception as ex:
        if verbose:
            print(f"  [warn] spin computation failed: {ex}")
        S2 = S_val = Sz = D_total = 0.0
        spin_label = f"Mult={mol_data.get('multiplicity', 1)}"

    if verbose:
        print(f"  <S2>={S2:.4f}  S={S_val:.2f}  {spin_label}  "
              f"<D>={D_total:.4f}")

    # ── 4. All RDMs ─────────────────────────────────────────────────────────
    rdm_dict = compute_all_molecular_rdms(
        psi=psi0,
        norb_cas=norb_cas,
        n_spin_orbitals=n_so,
        n_electrons=ne,
        max_n_body=max_n_body,
        min_n_body=min_n_body,
    )
    metrics = build_metrics_dict(rdm_dict)

    # Count clusters per n-body order for the title page
    n_clusters_info = {}
    for k in range(1, max_n_body + 1):
        n_clusters_info[f"n_clusters_{k}"] = sum(
            1 for s in rdm_dict if len(s) == k
        )

    if verbose:
        print(f"  RDMs computed: {len(rdm_dict)}")
        for k in range(min_n_body, max_n_body + 1):
            nk = n_clusters_info[f"n_clusters_{k}"]
            print(f"    {k}-body clusters : {nk}")

    # ── 6. mol_info for PDF title pages (built BEFORE the loop so each cluster
    #        can be persisted the instant it finishes) ──────────────────────
    mol_info = _build_molecular_info(
        mol_name=mol_name, mol_data=mol_data, cas=cas, basis=basis,
        n_so=n_so, hilbert_dim=hilbert_dim, hf_energy=hf_energy,
        E0=E0, E1=E1, S2=S2, S_val=S_val, Sz=Sz,
        spin_label=spin_label, D_total=D_total,
        cr_method=cr_method, cr_restarts=cr_restarts,
    )
    mol_info.update(n_clusters_info)
    if psi_external is not None:
        # provenance: record where the external wavefunction came from
        mol_info["wavefunction_source"] = "external_ci"
        mol_info["psi_source"] = psi_source or "<unspecified>"

    # ── 5-7. Per body order: RDM PDF, then convex roof (RESUME + INCREMENTAL
    # save), then the GME PDF — so each n is fully persisted the moment it
    # finishes and a kill / re-run resumes instead of losing work. ──────────
    out_dir = Path(output_dir) / mol_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cr_results: Dict = {}
    pkl_paths: List[str] = []
    pdf_paths: List[str] = []
    n_w = max(1, int(n_workers or 1))
    # solver-settings fingerprint in the PKL name: same settings -> reuse,
    # different settings -> recompute (matches the Hubbard cache convention)
    cr_tag = _cr_settings_tag(cr_method, cr_restarts, cr_max_iter, cr_k_max_extra)

    def _persist(sites, res):
        """Save one cluster's PKL immediately (incremental) and record it."""
        if not (save_pkl and res is not None):
            return
        try:
            pkl_paths.append(_save_molecular_gme_pkl(
                mol_name=mol_name, cas_tag=cas_tag, sites=sites,
                rho=rdm_dict[sites], cr_result=res, mol_info=mol_info,
                metrics_entry=metrics.get(sites, {}), cr_tag=cr_tag))
        except Exception as ex:
            if verbose:
                print(f"    [warn] PKL save failed for {sites}: {ex}")

    for n_body in range(min_n_body, max_n_body + 1):
        if n_clusters_info[f"n_clusters_{n_body}"] == 0:
            continue

        # RDM-only PDF (cheap; no convex roof) for this order
        if generate_rdm_pdf:
            rdm_path = out_dir / f"RDM_{n_body}body_{mol_name}_{cas_tag}.pdf"
            try:
                generate_molecular_rdm_only_pdf(
                    rdm_dict, metrics, n_body, str(rdm_path), mol_info)
                pdf_paths.append(str(rdm_path))
            except Exception as ex:
                if verbose:
                    print(f"    [warn] RDM PDF failed for n={n_body}: {ex}")

        # Convex roof only for n >= 3
        if cr_method == "none" or n_body < 3:
            continue

        targets_n = sorted(s for s in rdm_dict if len(s) == n_body)
        # RESUME: reload clusters already on disk, run only the rest
        pending = []
        for s in targets_n:
            cached = (_load_mol_gme_result(mol_name, cas_tag, s, cr_tag=cr_tag)
                      if resume else None)
            if cached is not None:
                cr_results[s] = cached
            else:
                pending.append(s)
        if verbose:
            print(f"\n  n={n_body}: {len(targets_n)} clusters "
                  f"({len(targets_n) - len(pending)} cached, "
                  f"{len(pending)} to run)  [{n_w} worker(s)]")

        if pending:
            if n_w > 1 and len(pending) > 1:
                import multiprocessing as mp
                jobs = [(rdm_dict[s], s, cr_method, cr_restarts, cr_max_iter,
                         cr_k_max_extra) for s in pending]
                with mp.Pool(processes=n_w) as pool:
                    for sites, res, err in pool.imap_unordered(_cr_worker, jobs):
                        if err is not None:
                            if verbose:
                                print(f"    [warn] convex roof failed for "
                                      f"{sites}: {err}")
                            continue
                        cr_results[sites] = res
                        _persist(sites, res)          # incremental save
                        if verbose and res is not None:
                            print(f"    sites={sites}  "
                                  f"C_gcg={res.get('C_gcg', float('nan')):.6f}  "
                                  f"status={res.get('status', '?')}")
            else:
                for sites in pending:
                    try:
                        res = run_convex_roof_on_rdm(
                            rdm_dict[sites], sites, method=cr_method,
                            n_restarts=cr_restarts, max_iter=cr_max_iter,
                            k_max_extra=cr_k_max_extra, verbose=verbose)
                        cr_results[sites] = res
                        _persist(sites, res)          # incremental save
                    except Exception as ex:
                        if verbose:
                            print(f"    [warn] convex roof failed for "
                                  f"{sites}: {ex}")

        # GME PDF for THIS order, now that it is complete
        if generate_pdf and any(len(s) == n_body for s in cr_results):
            gme_path = out_dir / f"GME_{n_body}body_{mol_name}_{cas_tag}.pdf"
            try:
                generate_molecular_gme_pdf(
                    rdm_dict, metrics, n_body, str(gme_path), mol_info,
                    cr_results=cr_results)
                pdf_paths.append(str(gme_path))
                if verbose:
                    print(f"    n={n_body} complete → {gme_path.name}")
            except Exception as ex:
                if verbose:
                    print(f"    [warn] GME PDF failed for n={n_body}: {ex}")

    if verbose and pkl_paths:
        print(f"\n  Saved {len(pkl_paths)} GME PKL(s) → "
              f"{_mol_gme_pkl_dir(mol_name)}/")

    # symmetry-class summary: mean C_gcg per order + per gap-class spread
    if verbose:
        _cluster_gme_summary(cr_results, norb_cas)

    return {
        "mol_info":   mol_info,
        "rdm_dict":   rdm_dict,
        "metrics":    metrics,
        "cr_results": cr_results,
        "pdf_paths":  pdf_paths,
        "pkl_paths":  pkl_paths,
        "psi0":       psi0,
        "E0":         E0,
        "E1":         E1,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Batch driver — all molecules / all CAS
# ──────────────────────────────────────────────────────────────────────────────

def run_all_molecules_analysis(
    mol_subset: Optional[Dict[str, dict]] = None,
    cas_index: Optional[int] = None,
    basis: str = "cc-pVDZ",
    max_n_body: int = 3,
    min_n_body: int = 1,
    output_dir: str = "results/mol_results",
    cr_method: str = "gcg",
    cr_restarts: int = 20,
    generate_pdf: bool = True,
    generate_rdm_pdf: bool = True,
    save_pkl: bool = True,
    verbose: bool = True,
    localize: Optional[str] = None,
) -> dict:
    """
    Loop `run_molecular_analysis` over a subset of molecules.

    Parameters
    ----------
    mol_subset : dict mapping name -> mol_data, or None for all MOLECULES
    cas_index  : if given, use only that index into the mol's valid_active_spaces
    """
    if mol_subset is None:
        mol_subset = MOLECULES

    print("=" * 64)
    print("  MOLECULAR GME ANALYSIS")
    print("=" * 64)
    print(f"  Molecules     : {list(mol_subset.keys())}")
    print(f"  Basis         : {basis}")
    print(f"  Max n-body    : {max_n_body}")
    print(f"  Convex roof   : {cr_method}   restarts={cr_restarts}")
    print(f"  Output root   : {output_dir}/")
    print()

    all_results: Dict[str, dict] = {}
    failed: List[str] = []

    for mol_name, mol_data in mol_subset.items():
        all_cas = mol_data["valid_active_spaces"]

        if cas_index is not None:
            if cas_index >= len(all_cas):
                print(f"  [skip] --cas {cas_index} out of range for "
                      f"{mol_name} (has {len(all_cas)})")
                continue
            cas_list = [all_cas[cas_index]]
        else:
            cas_list = all_cas

        print(f"\n{'=' * 64}")
        print(f"  {mol_name}  ({mol_data.get('formula', '?')})")
        print(f"  {len(cas_list)} active space(s) to compute")
        print(f"{'=' * 64}")

        for i, cas in enumerate(cas_list):
            cas_tag = f"CAS({cas['nele_cas']},{cas['norb_cas']})"
            print(f"\n  [{i + 1}/{len(cas_list)}]  {cas_tag}  "
                  f"ncore={cas['ncore']}")
            try:
                result = run_molecular_analysis(
                    mol_name=mol_name,
                    mol_data=mol_data,
                    cas=cas,
                    basis=basis,
                    max_n_body=max_n_body,
                    min_n_body=min_n_body,
                    output_dir=output_dir,
                    cr_method=cr_method,
                    cr_restarts=cr_restarts,
                    generate_pdf=generate_pdf,
                    generate_rdm_pdf=generate_rdm_pdf,
                    save_pkl=save_pkl,
                    verbose=verbose,
                    localize=localize,
                )
                all_results[f"{mol_name}_{cas_tag}"] = result
            except Exception as ex:
                print(f"  [ERROR] {mol_name} {cas_tag}: {ex}")
                traceback.print_exc()
                failed.append(f"{mol_name}_{cas_tag}")

    # ── Final report ────────────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print("  COMPLETED")
    print(f"{'=' * 64}")
    print(f"  Successful : {len(all_results)}")
    print(f"  Failed     : {len(failed)}")
    if failed:
        print(f"  Failed list: {failed}")
    print(f"\n  All PDFs saved under: {output_dir}/")

    return {"results": all_results, "failed": failed}


# ──────────────────────────────────────────────────────────────────────────────
# Thermal molecular GME analysis
# ──────────────────────────────────────────────────────────────────────────────

def _precompute_mol_ops(sites, n_so, ne):
    """Sparse operator table {(bra,ket): op} for a molecular orbital cluster."""
    from openfermion.ops import FermionOperator
    from openfermion.linalg import get_number_preserving_sparse_operator
    n = len(sites)
    dim = 2 ** n
    ops = {}
    for bra in range(dim):
        bra_spins = [(bra >> k) & 1 for k in range(n)]
        for ket in range(dim):
            ket_spins = [(ket >> k) & 1 for k in range(n)]
            terms = []
            for body in range(n):
                site = sites[body]
                terms += [(2 * site + bra_spins[body], 1),
                          (2 * site + ket_spins[body], 0)]
            ops[(bra, ket)] = get_number_preserving_sparse_operator(
                FermionOperator(tuple(terms)), n_so, ne, spin_preserving=True)
    return ops


def _mol_rdm_from_ops(ops, psi, n):
    dim = 2 ** n
    rho = np.zeros((dim, dim), dtype=complex)
    for (bra, ket), op in ops.items():
        rho[bra, ket] = np.vdot(psi, op @ psi)
    return rho


def run_molecular_thermal_analysis(
    mol_name: str,
    mol_data: dict,
    cas: dict,
    T_values,
    basis: str = "cc-pVDZ",
    max_n_body: int = 3,
    min_n_body: int = 3,
    k_states: int = 12,
    cr_restarts: int = 10,
    cr_max_iter: int = 120,
    cr_k_max_extra: int = 1,
    thermal_energy_cutoff: Optional[float] = None,
    output_dir: str = "results/mol_results",
    save_pkl: bool = True,
    generate_pdf: bool = True,
    verbose: bool = True,
    localize: Optional[str] = None,
) -> dict:
    """
    Thermal GME for one molecule + CAS.

    Builds the Gibbs ensemble over the k lowest eigenstates,
    Γ^(n)(T) = Σ_m p_m(T) Γ^(n)[ψ_m] with p_m ∝ exp(-(E_m-E_0)/T) (k_B=1,
    energies in Hartree), and computes C_GME(T) via GCG for every n-body
    cluster (n ≥ 3). Saves a thermal PKL and a C_GME-vs-T plot.

    localize : None (canonical orbitals) or 'boys'/'pm'/'ibo' — run the whole
        thermal analysis in the localized active-orbital basis (the supervisor
        convention is 'pm'). The cas_tag, PKL and PDF names carry a _loc<m> tag.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from itertools import combinations

    T_values = np.asarray(T_values, dtype=float)
    cas_tag = f"CAS{cas['nele_cas']}{cas['norb_cas']}"
    if localize:
        cas_tag += f"_loc{localize}"
    if verbose:
        print(f"\n  {'─'*60}\n  THERMAL  {mol_name}  CAS({cas['nele_cas']},"
              f"{cas['norb_cas']})  k_states={k_states}  "
              f"T∈[{T_values[0]:.3g},{T_values[-1]:.3g}] Ha"
              + (f"   [{localize}-localized orbitals]" if localize else "")
              + f"\n  {'─'*60}")

    if localize:
        from .molecular_localize import build_localized_molecular_hamiltonian
        _li = build_localized_molecular_hamiltonian(
            mol_data, cas, basis=basis, method=localize)
        H_sparse, hf_energy = _li["H_sparse"], _li["hf_energy"]
        n_so, ne, norb_cas = _li["n_so"], _li["ne"], _li["norb"]
        hilbert_dim = H_sparse.shape[0]
    else:
        H_sparse, hf_energy, n_so, ne, norb_cas, hilbert_dim = \
            build_molecular_hamiltonian(mol_data, cas, basis=basis)
    k_eff = min(k_states, hilbert_dim - 1)
    energies, vecs = compute_molecular_k_eigenstates(H_sparse, k=k_eff)
    E0 = float(energies[0])
    gap = float(energies[1] - energies[0]) if len(energies) > 1 else 0.0
    if verbose:
        print(f"  {len(energies)} eigenstates  E0={E0:.6f}  gap={gap:.6f} Ha")

    def _probs(T):
        if T <= 1e-14:
            w = (np.abs(energies - E0) < 1e-10).astype(float)
            return w / w.sum()
        rel = (energies - E0) / T
        if thermal_energy_cutoff is not None:
            keep = rel < thermal_energy_cutoff
            if not keep.any():
                keep = np.zeros_like(rel, bool); keep[0] = True
        else:
            keep = np.ones_like(rel, bool)
        w = np.zeros_like(rel)
        w[keep] = np.exp(-(rel[keep] - rel[keep].min()))
        return w / w.sum()

    n_body_orders = [n for n in range(max(3, min_n_body), max_n_body + 1)
                     if n <= ne and n <= norb_cas]
    clusters = {n: list(combinations(range(norb_cas), n)) for n in n_body_orders}

    thermal: Dict = {}   # sites -> {"C_gcg": arr, "purity": arr}
    for n in n_body_orders:
        for sites in clusters[n]:
            ops = _precompute_mol_ops(sites, n_so, ne)
            # raw (un-normalised) per-eigenstate RDMs: the thermal n-RDM is the
            # linear Boltzmann sum of these, normalised afterwards (the raw
            # trace carries each state's occupancy weight).
            rdms = np.stack([_mol_rdm_from_ops(ops, vecs[:, m], n)
                             for m in range(len(energies))], axis=0)
            traces = np.real(np.trace(rdms, axis1=1, axis2=2))
            if abs(traces[0]) < 1e-10:
                continue  # ground-state n-RDM ill-defined for this cluster
            C_arr = np.zeros(len(T_values)); P_arr = np.zeros(len(T_values))
            for i, T in enumerate(T_values):
                p = _probs(float(T))
                G = np.einsum("m,mij->ij", p, rdms)
                trG = np.real(np.trace(G))
                if abs(trG) < 1e-12:
                    C_arr[i] = float("nan"); P_arr[i] = float("nan"); continue
                G = G / trG
                pur = float(np.real(np.trace(G @ G)))
                if abs(pur - 1.0) < 1e-6:
                    C = float(C_GME_pure_dm(G, n))
                else:
                    C = float(convex_roof_gcg(
                        G, n, n_restarts=cr_restarts, max_iter=cr_max_iter,
                        k_max_extra=cr_k_max_extra, verbose=False)["C_gcg"])
                C_arr[i] = C; P_arr[i] = pur
            thermal[sites] = {"C_gcg": C_arr, "purity": P_arr, "n": n}
            if verbose:
                print(f"    n={n} sites={sites}  "
                      f"C(T0)={C_arr[0]:.4f}  C(Tmax)={C_arr[-1]:.4f}")

    # spin of the ground state (for the meta)
    try:
        S2, S_val, Sz, spin_label, D_total = \
            compute_molecular_spin_docc(vecs[:, 0], norb_cas, n_so, ne)
    except Exception:
        S2 = S_val = Sz = D_total = float("nan"); spin_label = "?"

    out_dir = Path(output_dir) / mol_name
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = None
    if generate_pdf and thermal:
        fig, ax = plt.subplots(figsize=(8, 5))
        for sites, r in thermal.items():
            ax.plot(T_values, r["C_gcg"], marker="o", ms=3,
                    label=f"n={r['n']} {sites}")
        ax.set_xlabel("Temperature T (Hartree, $k_B{=}1$)")
        ax.set_ylabel(r"$C_{GME}$ (GCG)")
        ax.set_title(f"Thermal GME: {mol_name} CAS({cas['nele_cas']},"
                     f"{cas['norb_cas']})  gap={gap:.4f} Ha")
        ax.axvline(gap, color="gray", ls=":", lw=1, alpha=0.6, label=f"gap={gap:.3g}")
        if len(thermal) <= 12:
            ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)
        plt.tight_layout()
        plot_path = str(out_dir / f"THERMAL_GME_{mol_name}_{cas_tag}.pdf")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight"); plt.close(fig)
        if verbose:
            print(f"  Thermal plot → {plot_path}")

    pkl_path = None
    if save_pkl and thermal:
        payload = {
            "meta": {
                "object_type": "hubbard_molecular_thermal_gme",
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "cache_version": "v1", "mol_name": mol_name, "cas_tag": cas_tag,
                "formula": mol_data.get("formula", mol_name), "basis": basis,
                "ncore": cas["ncore"], "nele_cas": cas["nele_cas"],
                "norb_cas": norb_cas, "n_so": n_so, "hilbert_dim": hilbert_dim,
                "hf_energy": hf_energy, "E0": E0, "gap": gap,
                "S2": S2, "S": S_val, "Sz": Sz, "spin_label": spin_label,
                "double_occ": D_total, "k_states": int(len(energies)),
                "energies": energies, "T_values": T_values,
                "thermal_energy_cutoff": thermal_energy_cutoff,
                "cr_restarts": cr_restarts, "environment": _basic_env_meta(),
            },
            "thermal": {str(sites): {"n": r["n"],
                                     "C_gcg": r["C_gcg"], "purity": r["purity"]}
                        for sites, r in thermal.items()},
        }
        d = get_rdm_pkl_results_dir() / "mol_thermal_runs" / mol_name
        d.mkdir(parents=True, exist_ok=True)
        pkl_path = str(d / f"molthermal_v1_{mol_name}_{cas_tag}.pkl")
        tmp = pkl_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, pkl_path)
        if verbose:
            print(f"  Thermal PKL → {pkl_path}")

    return {
        "mol_name": mol_name, "cas_tag": cas_tag, "T_values": T_values,
        "energies": energies, "gap": gap, "thermal": thermal,
        "plot_path": plot_path, "pkl_path": pkl_path,
    }