#!/usr/bin/env python3
"""
run_molecules_batch.py
----------------------
Batch driver: for every molecule, pick a feasible CAS active space and run
  (1) T=0 GME analysis (GCG convex roof, per-cluster PKLs), and
  (2) thermal GME analysis (Gibbs ensemble over eigenstates, C_GME(T)).

Everything is saved:
  results/rdm_pkl_results/mol_gme_runs/<Mol>/     per-cluster T=0 GME PKLs
  results/rdm_pkl_results/mol_thermal_runs/<Mol>/ thermal C_GME(T) PKLs
  results/mol_results/<Mol>/                       PDFs (RDM, GME table, thermal)

A one-row-per-(molecule,cluster) summary is written to
  results/mol_results/molecular_summary.csv

Usage:
  python run_molecules_batch.py                 # all molecules
  python run_molecules_batch.py --mols Methylene Ethylene NH3+
  python run_molecules_batch.py --no-thermal    # T=0 only
"""
from __future__ import annotations
import argparse, os, sys, time, csv, importlib.util
from math import comb
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from hubbard_gme.molecular_pipeline import (
    run_molecular_analysis, run_molecular_thermal_analysis,
)

# The latest curated molecule dataset used by the VQE benchmark.
DEFAULT_MOLECULES_PY = ("/home/anjum/PhDWork/BenchMark For Ground State/"
                        "molecules_data.py")


def load_molecules(path: str) -> dict:
    """Load the `molecules` (or `MOLECULES`) dict from a molecules_data.py file."""
    spec = importlib.util.spec_from_file_location("molecules_data_ext", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for attr in ("molecules", "MOLECULES"):
        if hasattr(mod, attr):
            return getattr(mod, attr)
    raise AttributeError(f"{path} defines neither `molecules` nor `MOLECULES`")

# Thermal grid in Hartree (k_B = 1); spans the ~0.02 Ha gap scale.
T_GRID = np.array([0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1])


def pick_cas(cas_list, dim_cap=150):
    """
    Largest norb_cas active space with EVEN nele_cas>=4 and dim<=cap.
    Returns (None, None) if the molecule has no even-electron CAS (e.g. an
    open-shell cation like NH3+ — the Sz=0 convex roof needs n_up=n_down).
    """
    best = None
    for c in cas_list:
        ne, no = c["nele_cas"], c["norb_cas"]
        if ne < 4 or ne % 2 or no < 4:
            continue
        dim = comb(no, ne // 2) ** 2
        if dim > dim_cap:
            continue
        if best is None or no > best[0]["norb_cas"]:
            best = (c, dim)
    return best if best is not None else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mols", nargs="+", default=None, help="subset (default: all)")
    ap.add_argument("--molecules-py", default=DEFAULT_MOLECULES_PY,
                    help="path to molecules_data.py (default: BenchMark latest)")
    ap.add_argument("--max-n", type=int, default=3)
    ap.add_argument("--cr-restarts", type=int, default=8)
    ap.add_argument("--thermal-restarts", type=int, default=5)
    ap.add_argument("--cr-max-iter", type=int, default=80)
    ap.add_argument("--cr-kmax", type=int, default=1)
    ap.add_argument("--k-states", type=int, default=10)
    ap.add_argument("--no-thermal", action="store_true")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()

    MOLECULES = load_molecules(args.molecules_py)
    print(f"Loaded {len(MOLECULES)} molecules from {args.molecules_py}")
    mols = args.mols or list(MOLECULES.keys())
    rows = []
    t_start = time.time()

    for mi, name in enumerate(mols, 1):
        if name not in MOLECULES:
            print(f"[skip] unknown molecule {name}"); continue
        mol = MOLECULES[name]
        cas, dim = pick_cas(mol["valid_active_spaces"])
        if cas is None:
            print(f"\n[{mi}/{len(mols)}] {name}  — SKIP (no even-electron "
                  f"feasible CAS; open-shell/charged)", flush=True)
            continue
        print(f"\n{'='*70}\n[{mi}/{len(mols)}] {name}  "
              f"CAS({cas['nele_cas']},{cas['norb_cas']}) dim={dim}\n{'='*70}", flush=True)

        # ── T=0 GME ──────────────────────────────────────────────────────────
        try:
            res = run_molecular_analysis(
                mol_name=name, mol_data=mol, cas=cas,
                max_n_body=args.max_n, min_n_body=1,
                cr_method="gcg", cr_restarts=args.cr_restarts,
                cr_max_iter=args.cr_max_iter, cr_k_max_extra=args.cr_kmax,
                generate_pdf=not args.no_pdf, generate_rdm_pdf=not args.no_pdf,
                save_pkl=True, verbose=False,
            )
            info = res["mol_info"]
            for sites, cr in (res["cr_results"] or {}).items():
                if cr is None: continue
                rows.append({
                    "molecule": name, "cas": f"({cas['nele_cas']},{cas['norb_cas']})",
                    "n_body": len(sites), "sites": "-".join(map(str, sites)),
                    "spin": info.get("spin_label"), "gap": info.get("gap"),
                    "purity": cr.get("global_purity"),
                    "C_gcg": cr.get("C_gcg"), "C_LB_best": cr.get("C_LB_best"),
                    "C_UB": cr.get("C_UB"), "status": cr.get("status"),
                })
            print(f"  [T=0] {len(res.get('pkl_paths',[]))} GME PKLs, "
                  f"{len(res.get('pdf_paths',[]))} PDFs", flush=True)
        except Exception as ex:
            print(f"  [ERROR T=0] {name}: {ex}", flush=True)

        # ── Thermal ──────────────────────────────────────────────────────────
        if not args.no_thermal:
            try:
                th = run_molecular_thermal_analysis(
                    mol_name=name, mol_data=mol, cas=cas, T_values=T_GRID,
                    max_n_body=args.max_n, min_n_body=3, k_states=args.k_states,
                    cr_restarts=args.thermal_restarts,
                    cr_max_iter=args.cr_max_iter, cr_k_max_extra=args.cr_kmax,
                    generate_pdf=not args.no_pdf, save_pkl=True, verbose=False,
                )
                print(f"  [thermal] {len(th['thermal'])} clusters, gap={th['gap']:.4f} Ha "
                      f"→ {os.path.basename(th['pkl_path']) if th['pkl_path'] else 'none'}",
                      flush=True)
            except Exception as ex:
                print(f"  [ERROR thermal] {name}: {ex}", flush=True)

    # ── Summary CSV ──────────────────────────────────────────────────────────
    os.makedirs("results/mol_results", exist_ok=True)
    csv_path = "results/mol_results/molecular_summary.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    print(f"\n{'='*70}\nDONE in {time.time()-t_start:.0f}s. "
          f"{len(rows)} (molecule,cluster) rows → {csv_path}")


if __name__ == "__main__":
    main()
