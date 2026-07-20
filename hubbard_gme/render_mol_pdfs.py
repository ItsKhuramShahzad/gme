#!/usr/bin/env python3
"""
render_mol_pdfs.py
------------------
Render molecular GME PDFs directly from the saved PKLs (no recomputation,
no PySCF, low memory). Produces, per molecule, under results/mol_results/<Mol>/:

  GME_T0_<Mol>_<cas>.pdf       bar chart of ground-state C_gcg per 3-orbital cluster
  THERMAL_GME_<Mol>_<cas>.pdf  C_GME(T) curves per cluster (+ purity)

Usage:  python render_mol_pdfs.py
"""
import glob, os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from hubbard_gme.molecular import compute_rdm_properties
from hubbard_gme.molecular_visualization import (
    generate_molecular_gme_pdf, generate_molecular_rdm_only_pdf,
)

ROOT = "results/rdm_pkl_results"
OUT = "results/mol_results"


def render_pipeline_pdfs():
    """
    Reconstruct rdm_dict / metrics / mol_info / cr_results from the saved
    per-cluster PKLs and call the SAME writers the pipeline uses, which reuse
    the Hubbard RDM-heatmap + combined-GME pages (visualization.py, untouched).
    Produces the RDM-only PDF and the full GME PDF per molecule.
    """
    by_mol = {}
    for p in glob.glob(f"{ROOT}/mol_gme_runs/*/molgme_*.pkl"):
        d = pickle.load(open(p, "rb"))
        by_mol.setdefault(d["meta"]["mol_name"], []).append(d)
    for mol, recs in by_mol.items():
        meta0 = recs[0]["meta"]; n = int(meta0["n"]); cas = meta0["cas_tag"]
        rdm_dict, metrics, cr_results = {}, {}, {}
        for d in recs:
            sites = tuple(d["meta"]["sites"])
            rho = d["rho"]["rho_real"] + 1j * d["rho"]["rho_imag"]
            rdm_dict[sites] = rho
            metrics[sites] = compute_rdm_properties(rho)
            cr = dict(d["result_full"]); cr["sites"] = sites; cr["n"] = n
            # attach the best decomposition for the combined page, if present
            kr = d.get("decompositions", {}).get("gcg", [])
            if kr:
                cr.setdefault("decomp", kr[-1].get("decomp", []))
            cr.setdefault("numerical", {}); cr.setdefault("stiefel", {})
            cr_results[sites] = cr
        mol_info = dict(meta0)
        mol_info["name"] = meta0["mol_name"]
        mol_info.setdefault("charge", meta0.get("charge", 0))
        mol_info.setdefault("multiplicity", meta0.get("multiplicity", 1))
        mol_info.setdefault("total_electrons", meta0.get("total_electrons", "?"))
        mol_info[f"n_clusters_{n}"] = len(rdm_dict)
        dd = f"{OUT}/{mol}"; os.makedirs(dd, exist_ok=True)
        try:
            generate_molecular_rdm_only_pdf(
                rdm_dict, metrics, n, f"{dd}/RDM_{n}body_{mol}_{cas}.pdf", mol_info)
            generate_molecular_gme_pdf(
                rdm_dict, metrics, n, f"{dd}/GME_{n}body_{mol}_{cas}.pdf",
                mol_info, cr_results=cr_results)
        except Exception as ex:
            print(f"  [warn] {mol}: {ex}")


def render_t0():
    by_mol = {}
    for p in glob.glob(f"{ROOT}/mol_gme_runs/*/molgme_*.pkl"):
        d = pickle.load(open(p, "rb"))
        m = d["meta"]
        by_mol.setdefault(m["mol_name"], []).append(
            (tuple(m["sites"]), float(d["result_full"].get("C_gcg", np.nan)),
             float(d["result_full"].get("C_UB", np.nan)),
             float(d["rho"].get("purity", np.nan)), m["cas_tag"]))
    for mol, recs in by_mol.items():
        recs.sort(key=lambda r: -r[1])
        sites = [",".join(map(str, r[0])) for r in recs]
        Cg = [max(0, r[1]) for r in recs]; Cub = [r[2] for r in recs]
        cas = recs[0][4]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(sites))
        ax.bar(x, Cg, color="#1f77b4", label=r"$C_{GME}$ (GCG)")
        ax.plot(x, Cub, "k^", ms=5, label=r"$C_{UB}$ (upper bound)")
        ax.set_xticks(x); ax.set_xticklabels(sites, rotation=60, fontsize=7)
        ax.set_ylabel(r"$C_{GME}$"); ax.set_xlabel("3-orbital cluster")
        ax.set_title(f"Ground-state tripartite GME: {mol}  {cas}")
        ax.legend(); ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        d = f"{OUT}/{mol}"; os.makedirs(d, exist_ok=True)
        f = f"{d}/GME_T0_{mol}_{cas}.pdf"
        fig.savefig(f, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  {f}")


def render_thermal():
    for p in glob.glob(f"{ROOT}/mol_thermal_runs/*/molthermal_*.pkl"):
        d = pickle.load(open(p, "rb"))
        m = d["meta"]; T = np.asarray(m["T_values"]); th = d["thermal"]
        mol, cas, gap = m["mol_name"], m["cas_tag"], m["gap"]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
        # color each cluster consistently across both panels; label by sites
        items = sorted(th.items(), key=lambda kv: -np.max(kv[1]["C_gcg"]))
        cmap = plt.get_cmap("tab20")
        for i, (sites, r) in enumerate(items):
            c = cmap(i % 20)
            # thermal PKL stores cluster keys as strings like "(2, 3, 4)"
            lbl = str(sites).replace(" ", "").strip("()")
            a1.plot(T, np.clip(r["C_gcg"], 0, None), marker="o", ms=3, lw=1,
                    color=c, label=lbl)
            a2.plot(T, r["purity"], marker="s", ms=3, lw=1, color=c, label=lbl)
        a1.set_xlabel("Temperature T (Hartree, $k_B{=}1$)")
        a1.set_ylabel(r"$C_{GME}$ (GCG)")
        a1.set_title(f"Thermal GME: {mol} {cas}  gap={gap:.3f} Ha")
        a1.axvline(gap, color="gray", ls=":", lw=1, alpha=.6, label=f"gap={gap:.3g}")
        a1.grid(True, alpha=.3); a1.set_ylim(bottom=0)
        a1.legend(title="cluster (orbitals)", fontsize=7, ncol=2, loc="upper right")
        a2.set_xlabel("Temperature T (Hartree)"); a2.set_ylabel(r"purity Tr$(\rho^2)$")
        a2.set_title("RDM purity vs T"); a2.grid(True, alpha=.3)
        a2.axhline(1/8, color="r", ls="--", lw=1, alpha=.5, label="$2^{-n}$ (n=3)")
        a2.legend(title="cluster (orbitals)", fontsize=7, ncol=2, loc="upper right")
        plt.tight_layout()
        dd = f"{OUT}/{mol}"; os.makedirs(dd, exist_ok=True)
        f = f"{dd}/THERMAL_GME_{mol}_{cas}.pdf"
        fig.savefig(f, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  {f}")


if __name__ == "__main__":
    print("RDM heatmaps + GME pages (Hubbard-style, from PKLs):"); render_pipeline_pdfs()
    print("T=0 GME bar charts:"); render_t0()
    print("Thermal GME curves:"); render_thermal()
    print("done.")
