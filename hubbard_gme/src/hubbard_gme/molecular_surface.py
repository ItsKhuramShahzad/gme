"""
molecular_surface.py
====================

Two molecular analyses that mirror the Hubbard-model toolkit:

1. :func:`run_molecular_thermal_surface` — the molecular analogue of the Hubbard
   C_GME(t/U, T) surface. A molecule has no t/U knob, so the second axis is a
   UNIFORM BOND-SCALE factor s applied to the geometry (about its centroid):
   stretching the molecule (s > 1) weakens hopping relative to on-site
   repulsion, i.e. it plays the role of increasing U/t. For every (s, T) grid
   point the CAS Hamiltonian is rebuilt, the Gibbs ensemble over the k lowest
   eigenstates is formed, and C_GME is computed per n-orbital cluster via GCG.
   Produces a 3D surface PDF, a 2D heatmap PDF, and a self-contained PKL.

2. :func:`run_molecular_excited_singlet` — GME of the FIRST EXCITED SINGLET
   state, per cluster, side by side with the ground state (bar-chart PDF +
   PKL). States are classified by <S^2>; the first eigenstate above the ground
   manifold with S ~ 0 is taken.

Both support ``localize`` ('pm' recommended — supervisor convention): the
analysis then runs in the localized active-orbital basis, and all outputs carry
a ``_loc<method>`` tag.
"""
from __future__ import annotations

import os
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .rdm import get_rdm_pkl_results_dir, _basic_env_meta
from .molecular import (
    build_molecular_hamiltonian,
    compute_molecular_k_eigenstates,
)
from .molecular_pipeline import _precompute_mol_ops, _mol_rdm_from_ops
from .convex_roof import convex_roof_gcg, C_GME_pure_dm


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def _scaled_mol_data(mol_data: dict, s: float) -> dict:
    """mol_data with every atomic position scaled by s about the centroid."""
    geom = mol_data["geometry"]
    coords = np.asarray([list(xyz) for _, xyz in geom], dtype=float)
    center = coords.mean(axis=0)
    new = center + s * (coords - center)
    out = dict(mol_data)
    out["geometry"] = [(el, tuple(new[i])) for i, (el, _) in enumerate(geom)]
    return out


def _build_H(mol_data, cas, basis, localize):
    """(H_sparse, n_so, ne, norb) in the canonical or localized basis."""
    if localize:
        from .molecular_localize import build_localized_molecular_hamiltonian
        li = build_localized_molecular_hamiltonian(
            mol_data, cas, basis=basis, method=localize)
        return li["H_sparse"], li["n_so"], li["ne"], li["norb"]
    H, _hf, n_so, ne, norb, _dim = build_molecular_hamiltonian(
        mol_data, cas, basis=basis)
    return H, n_so, ne, norb


def _gibbs_probs(energies: np.ndarray, T: float) -> np.ndarray:
    E0 = float(energies[0])
    if T <= 1e-14:
        w = (np.abs(energies - E0) < 1e-10).astype(float)
    else:
        rel = (energies - E0) / T
        w = np.exp(-(rel - rel.min()))
    return w / w.sum()


def _cluster_C(G: np.ndarray, n: int, cr_restarts, cr_max_iter, cr_kmax) -> float:
    pur = float(np.real(np.trace(G @ G)))
    if abs(pur - 1.0) < 1e-6:
        return float(C_GME_pure_dm(G, n))
    return float(convex_roof_gcg(G, n, n_restarts=cr_restarts,
                                 max_iter=cr_max_iter, k_max_extra=cr_kmax,
                                 verbose=False)["C_gcg"])


def _save_pkl(path: Path, payload: dict) -> None:
    import pickle
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


# ──────────────────────────────────────────────────────────────────────────────
# 1. thermal surface  C_GME(bond scale s, T)
# ──────────────────────────────────────────────────────────────────────────────

def run_molecular_thermal_surface(
    mol_name: str,
    mol_data: dict,
    cas: dict,
    scales,
    T_values,
    n_body: int = 3,
    basis: str = "cc-pVDZ",
    k_states: int = 10,
    cr_restarts: int = 8,
    cr_max_iter: int = 120,
    cr_k_max_extra: int = 1,
    localize: Optional[str] = "pm",
    output_dir: str = "results/mol_results",
    save_pkl: bool = True,
    generate_pdf: bool = True,
    verbose: bool = True,
) -> dict:
    """
    C_GME(s, T) surface for one molecule: x = uniform bond-scale factor s,
    y = temperature T (Hartree, k_B=1), z = C_GME (GCG) per n_body-cluster.
    Saves the per-cluster surfaces plus the max/mean aggregates, and renders
    the 3D + 2D figures in the Hubbard style.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scales = np.asarray(scales, dtype=float)
    T_values = np.asarray(T_values, dtype=float)
    cas_tag = f"CAS{cas['nele_cas']}{cas['norb_cas']}"
    if localize:
        cas_tag += f"_loc{localize}"
    ns, nT = len(scales), len(T_values)

    if verbose:
        print(f"\n  {'─'*64}\n  (s, T) SURFACE  {mol_name}  "
              f"CAS({cas['nele_cas']},{cas['norb_cas']})  n={n_body}"
              + (f"  [{localize}-localized]" if localize else "  [canonical]"))
        print(f"  s ∈ [{scales[0]:.3g}, {scales[-1]:.3g}] ({ns} pts)   "
              f"T ∈ [{T_values[0]:.3g}, {T_values[-1]:.3g}] Ha ({nT} pts)   "
              f"k_states={k_states}\n  {'─'*64}")

    clusters: List[tuple] = []
    C_surf: Dict[tuple, np.ndarray] = {}
    E0_arr = np.zeros(ns)
    gap_arr = np.zeros(ns)

    for si, s in enumerate(scales):
        md = _scaled_mol_data(mol_data, float(s))
        H, n_so, ne, norb = _build_H(md, cas, basis, localize)
        k_eff = min(k_states, H.shape[0] - 1)
        energies, vecs = compute_molecular_k_eigenstates(H, k=k_eff)
        E0_arr[si] = float(energies[0])
        gap_arr[si] = float(energies[1] - energies[0]) if len(energies) > 1 else 0.0
        if not clusters:
            clusters = list(combinations(range(norb), n_body))
            for c in clusters:
                C_surf[c] = np.full((ns, nT), np.nan)
        if verbose:
            print(f"  [{si+1}/{ns}] s={s:.3f}  E0={energies[0]:.6f}  "
                  f"gap={gap_arr[si]:.4f} Ha", flush=True)

        for sites in clusters:
            ops = _precompute_mol_ops(sites, n_so, ne)
            rdms = np.stack([_mol_rdm_from_ops(ops, vecs[:, m], n_body)
                             for m in range(len(energies))], axis=0)
            if abs(np.real(np.trace(rdms[0]))) < 1e-10:
                continue                      # ill-defined cluster at this s
            for ti, T in enumerate(T_values):
                p = _gibbs_probs(energies, float(T))
                G = np.einsum("m,mij->ij", p, rdms)
                trG = np.real(np.trace(G))
                if abs(trG) < 1e-12:
                    continue
                C_surf[sites][si, ti] = _cluster_C(
                    G / trG, n_body, cr_restarts, cr_max_iter, cr_k_max_extra)
        if verbose:
            best = max(clusters, key=lambda c: np.nanmax(C_surf[c][si]))
            print(f"        maxC(T=0)={np.nanmax([C_surf[c][si,0] for c in clusters]):.4f}"
                  f"   maxC(T={T_values[-1]:g})="
                  f"{np.nanmax([C_surf[c][si,-1] for c in clusters]):.4f}"
                  f"   best cluster {best}", flush=True)

    stack = np.stack([C_surf[c] for c in clusters], axis=0)   # (ncl, ns, nT)
    C_max = np.nanmax(stack, axis=0)
    C_mean = np.nanmean(stack, axis=0)

    # ── figures (Hubbard style: 3D surface + 2D heatmap; max & mean panels) ──
    out: Dict[str, Optional[str]] = {"surface_pdf": None, "heatmap_pdf": None,
                                     "pkl_path": None}
    mol_dir = os.path.join(output_dir, mol_name)
    os.makedirs(mol_dir, exist_ok=True)
    SS, TT = np.meshgrid(scales, T_values, indexing="ij")
    panels = [(C_max, f"max over {len(clusters)} clusters"),
              (C_mean, "mean over clusters")]

    if generate_pdf:
        from matplotlib.ticker import MaxNLocator
        # 3D
        fig = plt.figure(figsize=(13.6, 6.0))
        for k, (Z, lab) in enumerate(panels):
            ax = fig.add_subplot(1, 2, k + 1, projection="3d")
            surf = ax.plot_surface(SS, TT, Z, cmap="viridis", edgecolor="none",
                                   antialiased=True, rstride=1, cstride=1)
            ax.set_xlabel("bond scale $s$", fontsize=12, labelpad=6)
            ax.set_ylabel("$T$ (Ha)", fontsize=12, labelpad=6)
            ax.set_zlabel(r"$C_{\mathrm{GME}}$", fontsize=12, labelpad=4)
            ax.set_title(f"$n={n_body}$ — {lab}", fontsize=12, y=1.02)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.zaxis.set_major_locator(MaxNLocator(5))
            ax.set_zlim(bottom=0)
            ax.view_init(elev=26, azim=-118)
            fig.colorbar(surf, ax=ax, shrink=0.55, aspect=12, pad=0.12)
        fig.suptitle(
            f"Thermal GME surface  $C_{{\\mathrm{{GME}}}}(s, T)$ — {mol_name} "
            f"{cas_tag}  (stretch $s$ ↔ raising $U/t$)", fontsize=13)
        fig.subplots_adjust(left=0.06, right=0.96, top=0.86, bottom=0.10,
                            wspace=0.25)
        p3 = os.path.join(mol_dir, f"SURFACE_GME_3D_{mol_name}_{cas_tag}_n{n_body}.pdf")
        fig.savefig(p3, dpi=300)          # no tight bbox: clips 3D labels
        plt.close(fig)
        out["surface_pdf"] = p3

        # 2D heatmap
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
        for ax, (Z, lab) in zip(axes, panels):
            im = ax.pcolormesh(scales, T_values, Z.T, cmap="viridis",
                               shading="gouraud", vmin=0.0)
            ax.set_xlabel("bond scale $s$", fontsize=12)
            ax.set_ylabel("$T$ (Ha)", fontsize=12)
            ax.set_title(f"$n={n_body}$ — {lab}", fontsize=12)
            fig.colorbar(im, ax=ax, shrink=0.92, pad=0.03,
                         label=r"$C_{\mathrm{GME}}$")
        fig.suptitle(
            f"Thermal GME heatmap  $C_{{\\mathrm{{GME}}}}(s, T)$ — {mol_name} "
            f"{cas_tag}", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        p2 = os.path.join(mol_dir, f"SURFACE_GME_2D_{mol_name}_{cas_tag}_n{n_body}.pdf")
        fig.savefig(p2, dpi=300, bbox_inches="tight")
        plt.close(fig)
        out["heatmap_pdf"] = p2
        if verbose:
            print(f"  3D surface → {p3}\n  2D heatmap → {p2}")

    if save_pkl:
        pkl = (get_rdm_pkl_results_dir() / "mol_surface_runs" / mol_name /
               f"molsurface_v1_{mol_name}_{cas_tag}_n{n_body}.pkl")
        _save_pkl(pkl, {
            "meta": {
                "object_type": "molecular_thermal_surface",
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "mol_name": mol_name, "cas_tag": cas_tag, "cas": dict(cas),
                "basis": basis, "localize": localize, "n_body": n_body,
                "scales": scales, "T_values": T_values, "k_states": k_states,
                "cr_restarts": cr_restarts, "cr_max_iter": cr_max_iter,
                "cr_k_max_extra": cr_k_max_extra,
                "E0_per_scale": E0_arr, "gap_per_scale": gap_arr,
                "environment": _basic_env_meta(),
            },
            "C_surface": {str(c): C_surf[c] for c in clusters},
            "C_max": C_max, "C_mean": C_mean,
        })
        out["pkl_path"] = str(pkl)
        if verbose:
            print(f"  surface PKL → {pkl}")

    out.update({"C_max": C_max, "C_mean": C_mean, "scales": scales,
                "T_values": T_values, "clusters": clusters})
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 2. first excited singlet
# ──────────────────────────────────────────────────────────────────────────────

def run_molecular_excited_singlet(
    mol_name: str,
    mol_data: dict,
    cas: dict,
    basis: str = "cc-pVDZ",
    max_n_body: int = 3,
    k_states: int = 12,
    cr_restarts: int = 8,
    cr_max_iter: int = 120,
    cr_k_max_extra: int = 1,
    localize: Optional[str] = "pm",
    output_dir: str = "results/mol_results",
    save_pkl: bool = True,
    generate_pdf: bool = True,
    verbose: bool = True,
) -> dict:
    """
    GME of the FIRST EXCITED SINGLET state, per n-orbital cluster (n = 3..max),
    compared with the ground state. States are classified via <S^2>; the first
    eigenstate above the ground manifold with S ~ 0 is selected.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from openfermion.hamiltonians import s_squared_operator
    from openfermion.linalg import get_number_preserving_sparse_operator

    cas_tag = f"CAS{cas['nele_cas']}{cas['norb_cas']}"
    if localize:
        cas_tag += f"_loc{localize}"

    H, n_so, ne, norb = _build_H(mol_data, cas, basis, localize)
    k_eff = min(k_states, H.shape[0] - 1)
    energies, vecs = compute_molecular_k_eigenstates(H, k=k_eff)
    E0 = float(energies[0])

    S2 = get_number_preserving_sparse_operator(
        s_squared_operator(norb), n_so, ne, spin_preserving=True)
    s2 = np.array([float(np.real(np.vdot(vecs[:, m], S2 @ vecs[:, m])))
                   for m in range(len(energies))])

    idx_es = None
    for m in range(1, len(energies)):
        if energies[m] > E0 + 1e-9 and s2[m] < 0.01:
            idx_es = m
            break
    if idx_es is None:
        raise RuntimeError(
            f"{mol_name}: no excited singlet among the lowest {k_eff} states "
            f"(<S^2> = {np.round(s2, 3)}) — raise --k-states.")

    if verbose:
        print(f"\n  {'─'*60}\n  EXCITED SINGLET  {mol_name}  {cas_tag}"
              + (f"  [{localize}-localized]" if localize else ""))
        print(f"  ground E0={E0:.6f}   first excited singlet: state {idx_es}, "
              f"E={energies[idx_es]:.6f}  (ΔE={energies[idx_es]-E0:.5f} Ha, "
              f"<S²>={s2[idx_es]:.4f})\n  {'─'*60}")

    n_orders = [n for n in range(3, max_n_body + 1) if n <= ne and n <= norb]
    records: Dict[tuple, dict] = {}
    for n in n_orders:
        for sites in combinations(range(norb), n):
            ops = _precompute_mol_ops(sites, n_so, ne)
            row = {}
            for tag, m in (("ground", 0), ("excited", idx_es)):
                rho = _mol_rdm_from_ops(ops, vecs[:, m], n)
                tr = np.real(np.trace(rho))
                if abs(tr) < 1e-10:
                    row[f"C_{tag}"] = float("nan")
                    continue
                row[f"C_{tag}"] = _cluster_C(rho / tr, n, cr_restarts,
                                             cr_max_iter, cr_k_max_extra)
            row["n"] = n
            records[sites] = row
            if verbose:
                print(f"    n={n} sites={sites}  C_ground={row['C_ground']:.4f}  "
                      f"C_excited={row['C_excited']:.4f}", flush=True)

    out: Dict[str, Optional[str]] = {"pdf_path": None, "pkl_path": None}
    mol_dir = os.path.join(output_dir, mol_name)
    os.makedirs(mol_dir, exist_ok=True)

    if generate_pdf and records:
        keys = sorted(records, key=lambda s: (records[s]["n"], s))
        labels = [",".join(map(str, s)) for s in keys]
        g = [max(0.0, records[s]["C_ground"]) for s in keys]
        e = [max(0.0, records[s]["C_excited"]) for s in keys]
        x = np.arange(len(keys))
        fig, ax = plt.subplots(figsize=(max(8.0, 0.42 * len(keys)), 4.8))
        ax.bar(x - 0.2, g, width=0.4, label="ground state", color="#1f77b4")
        ax.bar(x + 0.2, e, width=0.4,
               label=f"1st excited singlet (state {idx_es})", color="#d62728")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=60, fontsize=7)
        ax.set_ylabel(r"$C_{\mathrm{GME}}$ (GCG)")
        ax.set_xlabel("orbital cluster")
        ax.set_title(f"Ground vs first-excited-singlet GME — {mol_name} "
                     f"{cas_tag}  (ΔE={energies[idx_es]-E0:.4f} Ha)")
        ax.legend(); ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        pdf_path = os.path.join(
            mol_dir, f"EXCITED_SINGLET_GME_{mol_name}_{cas_tag}.pdf")
        fig.savefig(pdf_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        out["pdf_path"] = pdf_path
        if verbose:
            print(f"  plot → {pdf_path}")

    if save_pkl and records:
        pkl = (get_rdm_pkl_results_dir() / "mol_excited_runs" / mol_name /
               f"molexcited_v1_{mol_name}_{cas_tag}.pkl")
        _save_pkl(pkl, {
            "meta": {
                "object_type": "molecular_excited_singlet_gme",
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "mol_name": mol_name, "cas_tag": cas_tag, "cas": dict(cas),
                "basis": basis, "localize": localize,
                "energies": energies, "S2": s2, "excited_index": int(idx_es),
                "E0": E0, "E_excited": float(energies[idx_es]),
                "k_states": k_states, "cr_restarts": cr_restarts,
                "cr_max_iter": cr_max_iter, "cr_k_max_extra": cr_k_max_extra,
                "environment": _basic_env_meta(),
            },
            "records": {str(s): r for s, r in records.items()},
        })
        out["pkl_path"] = str(pkl)
        if verbose:
            print(f"  PKL → {pkl}")

    out.update({"excited_index": idx_es, "energies": energies, "S2": s2,
                "records": records})
    return out
