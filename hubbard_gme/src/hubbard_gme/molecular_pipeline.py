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
) -> str:
    """Save one (molecule, CAS, cluster) GME result to a self-contained PKL."""
    n = len(sites)
    sites_str = "-".join(str(s) for s in sites)
    fname = (f"molgme_{cache_version}_{mol_name}_{cas_tag}_"
             f"sites-{sites_str}_n{n}.pkl")
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
# Single molecule / single CAS
# ──────────────────────────────────────────────────────────────────────────────

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
) -> dict:
    """
    Full molecular pipeline for ONE molecule + ONE CAS active space.

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

    if verbose:
        print(f"\n  {'─' * 60}")
        print(f"  {mol_name}   CAS({cas['nele_cas']},{cas['norb_cas']})  "
              f"ncore={cas['ncore']}   basis={basis}"
              + (f"   [{localize}-localized orbitals]" if localize else ""))
        print(f"  {'─' * 60}")

    # ── 1. Hamiltonian ──────────────────────────────────────────────────────
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

    # ── 2. Ground state ─────────────────────────────────────────────────────
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

    # ── 5. Convex roof on every RDM with body order >= 3 ───────────────────
    cr_results: Dict = {}
    if cr_method != "none":
        targets = [s for s in rdm_dict if len(s) >= 3]
        if verbose:
            print(f"\n  Running convex-roof ({cr_method}, "
                  f"{cr_restarts} restarts) on {len(targets)} RDMs ...")
        for sites in targets:
            try:
                res = run_convex_roof_on_rdm(
                    rdm_dict[sites], sites,
                    method=cr_method,
                    n_restarts=cr_restarts,
                    max_iter=cr_max_iter,
                    k_max_extra=cr_k_max_extra,
                    verbose=verbose,
                )
                cr_results[sites] = res
            except Exception as ex:
                if verbose:
                    print(f"    [warn] convex roof failed for {sites}: {ex}")

    # ── 6. mol_info for PDF title pages ─────────────────────────────────────
    mol_info = _build_molecular_info(
        mol_name=mol_name, mol_data=mol_data, cas=cas, basis=basis,
        n_so=n_so, hilbert_dim=hilbert_dim, hf_energy=hf_energy,
        E0=E0, E1=E1, S2=S2, S_val=S_val, Sz=Sz,
        spin_label=spin_label, D_total=D_total,
        cr_method=cr_method, cr_restarts=cr_restarts,
    )
    mol_info.update(n_clusters_info)

    # ── 6b. Save each GME result to a self-contained PKL (like gme_runs/) ────
    pkl_paths: List[str] = []
    if save_pkl and cr_results:
        for sites, res in cr_results.items():
            if res is None:
                continue
            try:
                p = _save_molecular_gme_pkl(
                    mol_name=mol_name, cas_tag=cas_tag, sites=sites,
                    rho=rdm_dict[sites], cr_result=res, mol_info=mol_info,
                    metrics_entry=metrics.get(sites, {}),
                )
                pkl_paths.append(p)
            except Exception as ex:
                if verbose:
                    print(f"    [warn] PKL save failed for {sites}: {ex}")
        if verbose and pkl_paths:
            print(f"\n  Saved {len(pkl_paths)} GME PKL(s) → "
                  f"{_mol_gme_pkl_dir(mol_name)}/")

    # ── 7. PDFs ─────────────────────────────────────────────────────────────
    out_dir = Path(output_dir) / mol_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths: List[str] = []

    if generate_pdf or generate_rdm_pdf:
        if verbose:
            print(f"\n  Generating PDFs → {out_dir}/")

        for n_body in range(min_n_body, max_n_body + 1):
            if n_clusters_info[f"n_clusters_{n_body}"] == 0:
                continue

            # RDM-only PDF (cheap, no convex roof)
            if generate_rdm_pdf:
                rdm_path = out_dir / f"RDM_{n_body}body_{mol_name}_{cas_tag}.pdf"
                generate_molecular_rdm_only_pdf(
                    rdm_dict, metrics, n_body, str(rdm_path), mol_info,
                )
                pdf_paths.append(str(rdm_path))

            # GME PDF with convex-roof section (only if cr ran and k >= 3)
            if (generate_pdf and cr_method != "none"
                    and n_body >= 3
                    and any(len(s) == n_body for s in cr_results)):
                gme_path = out_dir / f"GME_{n_body}body_{mol_name}_{cas_tag}.pdf"
                generate_molecular_gme_pdf(
                    rdm_dict, metrics, n_body, str(gme_path), mol_info,
                    cr_results=cr_results,
                )
                pdf_paths.append(str(gme_path))

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