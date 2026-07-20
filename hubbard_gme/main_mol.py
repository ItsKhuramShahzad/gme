#!/usr/bin/env python3
"""
main_mol.py — Molecular GME analysis CLI (unified entry point)

The full molecular GME workflow with the GCG convex-roof solver, saving
everything like the standalone Hubbard GME caches:

  results/rdm_pkl_results/mol_gme_runs/<Mol>/      per-cluster T=0 GME PKLs
  results/rdm_pkl_results/mol_thermal_runs/<Mol>/  thermal C_GME(T) PKLs
  results/mol_results/<Mol>/                        PDFs (RDM heatmaps, GME, thermal)

MODES  (pick one; default = T=0 GME)
------
  (default)      T=0 GME analysis: RDMs + GCG convex roof + PKLs + PDFs
  --thermal      Thermal GME: Gibbs ensemble over eigenstates, C_GME(T)
  --audit        Audit the saved molecular GME + thermal results (PASS/WARN/FAIL)
  --render-pdfs  (Re)render ALL molecular PDFs from the saved PKLs (no recompute)

EXAMPLES
--------
# 1) T=0 GME, one molecule, auto-pick a feasible even-electron CAS (GCG)
python main_mol.py --mol Benzene --auto-cas

# 2) T=0 GME for the whole BenchMark dataset (auto CAS per molecule)
python main_mol.py \
     --molecules-py "/home/anjum/PhDWork/BenchMark For Ground State/molecules_data.py" \
     --auto-cas

# 3) Thermal GME for one molecule (T in Hartree; extend T-max for large gaps)
python main_mol.py --mol NH2- --auto-cas --thermal --T-max 0.3 --T-steps 16

# 4) One molecule, explicit CAS index, fast solver, no PDFs
python main_mol.py --mol Methylene --cas 3 --cr-max-iter 60 --cr-kmax 1 --no-pdf

# 5) Audit everything that has been saved
python main_mol.py --audit

# 6) Render every PDF from the saved PKLs (RDM heatmaps + GME pages + thermal)
python main_mol.py --render-pdfs

NOTE  Large molecules (nucleobases, PAHs) may exceed workstation memory in the
      full-system SCF; run those on an HPC node. The pipeline is identical.
"""

import argparse
import os
import sys

import numpy as np

# Make the package importable when running from the project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from hubbard_gme.molecular_pipeline import (
    run_all_molecules_analysis,
    run_molecular_analysis,
    run_molecular_thermal_analysis,
)
from hubbard_gme.molecules_data import MOLECULES as DEFAULT_MOLECULES, list_molecules
from run_molecules_batch import pick_cas, load_molecules, DEFAULT_MOLECULES_PY


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main_mol.py",
        description="Molecular GME analysis (GCG convex roof; T=0, thermal, "
                    "audit, and PDF rendering).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Mode ──────────────────────────────────────────────────────────────────
    m = p.add_argument_group("Mode (default: T=0 GME)")
    m.add_argument("--thermal", action="store_true",
                   help="Run thermal GME (Gibbs ensemble, C_GME vs T).")
    m.add_argument("--excited-singlet", action="store_true", dest="excited_singlet",
                   help="GME of the FIRST EXCITED SINGLET state (vs ground state).")
    m.add_argument("--thermal-surface", action="store_true", dest="thermal_surface",
                   help="3D+2D GME surface C_GME(bond-scale s, T): the molecular "
                        "analogue of the Hubbard C_GME(t/U, T) surface. Uniformly "
                        "stretching the geometry (s>1) increases correlation "
                        "(like increasing U/t).")
    m.add_argument("--audit", action="store_true",
                   help="Audit saved molecular results and exit.")
    m.add_argument("--render-pdfs", action="store_true", dest="render_pdfs",
                   help="Render all PDFs from saved PKLs (no recompute) and exit.")

    # ── Molecule / active space ──────────────────────────────────────────────
    s = p.add_argument_group("Molecule and active space")
    s.add_argument("--mol", type=str, default=None, metavar="NAME",
                   help="Molecule name (default: all).")
    s.add_argument("--molecules-py", default=None, metavar="PATH",
                   help="Use an external molecules_data.py (e.g. the BenchMark "
                        f"dataset: {DEFAULT_MOLECULES_PY}). Default: built-in.")
    s.add_argument("--cas", type=int, default=None, metavar="INT",
                   help="Active-space index (default: all valid CAS).")
    s.add_argument("--auto-cas", action="store_true", dest="auto_cas",
                   help="Auto-pick the largest feasible even-electron CAS "
                        "(dim<=150) per molecule.")
    s.add_argument("--basis", type=str, default="cc-pVDZ", metavar="NAME",
                   help="Basis set (default: cc-pVDZ).")
    s.add_argument("--localize", choices=["boys", "pm", "ibo", "canonical"],
                   default=None,
                   help="Orbital basis for RDMs/GME (pyscf.lo: boys/pm/ibo, or "
                        "'canonical'). Same ground-state energy; localized = "
                        "spatially-meaningful entanglement; output tagged "
                        "CAS..._loc<method>. T=0 default: canonical. THERMAL-type "
                        "modes (--thermal/--excited-singlet/--thermal-surface) "
                        "default to PM per the supervisor convention; pass "
                        "'canonical' to opt out.")
    s.add_argument("--max-n", type=int, default=3, dest="max_n", metavar="INT",
                   help="Max n-body RDM order (default: 3).")
    s.add_argument("--min-n", type=int, default=1, dest="min_n", metavar="INT",
                   help="Min n-body RDM order for T=0 RDMs (default: 1).")

    # ── Solver ────────────────────────────────────────────────────────────────
    c = p.add_argument_group("Convex-roof solver")
    c.add_argument("--cr-method", type=str, default="gcg",
                   choices=["none", "expo", "stiefel", "gcg", "both", "all", "full"],
                   dest="cr_method", help="Convex-roof method (default: gcg).")
    c.add_argument("--cr-restarts", type=int, default=8, dest="cr_restarts",
                   metavar="INT", help="Random restarts (default: 8).")
    c.add_argument("--cr-max-iter", type=int, default=200, dest="cr_max_iter",
                   metavar="INT", help="GCG max iterations (default: 200; "
                   "lower = faster, e.g. 60).")
    c.add_argument("--cr-kmax", type=int, default=2, dest="cr_kmax", metavar="INT",
                   help="GCG extra decomposition ranks (default: 2).")

    # ── Thermal grid ─────────────────────────────────────────────────────────
    t = p.add_argument_group("Thermal grid (with --thermal; T in Hartree, k_B=1)")
    t.add_argument("--T-min", type=float, default=0.0, dest="T_min", metavar="FLOAT")
    t.add_argument("--T-max", type=float, default=0.1, dest="T_max", metavar="FLOAT",
                   help="Extend for large-gap molecules (default: 0.1).")
    t.add_argument("--T-steps", type=int, default=8, dest="T_steps", metavar="INT",
                   help="Number of temperature points (default: 8).")
    t.add_argument("--k-states", type=int, default=10, dest="k_states", metavar="INT",
                   help="Eigenstates in the Gibbs ensemble (default: 10).")

    # ── Surface grid (with --thermal-surface) ────────────────────────────────
    u = p.add_argument_group("Surface grid (with --thermal-surface)")
    u.add_argument("--scale-min", type=float, default=1.0, dest="scale_min",
                   metavar="FLOAT", help="Smallest bond-scale factor (default 1.0).")
    u.add_argument("--scale-max", type=float, default=1.6, dest="scale_max",
                   metavar="FLOAT", help="Largest bond-scale factor (default 1.6; "
                   "stretching raises correlation, like raising U/t).")
    u.add_argument("--scale-steps", type=int, default=6, dest="scale_steps",
                   metavar="INT", help="Scale-grid points (default 6).")
    u.add_argument("--surface-n", type=int, default=3, dest="surface_n",
                   metavar="INT", help="Body order n for the surface (default 3).")

    # ── Output ────────────────────────────────────────────────────────────────
    o = p.add_argument_group("Output")
    o.add_argument("--output-dir", type=str, default="results/mol_results",
                   dest="output_dir", metavar="PATH")
    o.add_argument("--no-pdf", action="store_true",
                   help="Do not generate PDFs (compute only).")
    o.add_argument("--no-rdm-pdf", action="store_true", dest="no_rdm_pdf",
                   help="Skip the RDM-only PDFs (T=0 mode).")
    o.add_argument("--no-save-pkl", action="store_false", dest="save_pkl",
                   help="Do not save per-cluster/thermal PKLs.")
    o.add_argument("--quiet", action="store_true", help="Less verbose stdout.")
    return p


def _iter_targets(mol_subset, cas_index, auto_cas):
    """Yield (name, mol_data, cas) honoring --cas / --auto-cas."""
    for name, mol in mol_subset.items():
        cas_list = mol["valid_active_spaces"]
        if auto_cas:
            cas, _ = pick_cas(cas_list)
            if cas is None:
                print(f"  [skip] {name}: no feasible even-electron CAS")
                continue
            yield name, mol, cas
        elif cas_index is not None:
            if cas_index < len(cas_list):
                yield name, mol, cas_list[cas_index]
            else:
                print(f"  [skip] {name}: --cas {cas_index} out of range")
        else:
            for cas in cas_list:
                yield name, mol, cas


def main() -> None:
    args = build_parser().parse_args()

    # ── Standalone modes ─────────────────────────────────────────────────────
    if args.audit:
        from hubbard_gme.molecular_audit import audit
        from pathlib import Path
        sys.exit(0 if audit(Path("results/rdm_pkl_results").resolve()) != "FAIL" else 1)

    if args.render_pdfs:
        import render_mol_pdfs as R
        print("RDM heatmaps + GME pages (from PKLs):"); R.render_pipeline_pdfs()
        print("T=0 GME bar charts:"); R.render_t0()
        print("Thermal GME curves:"); R.render_thermal()
        print("done."); return

    # ── Resolve molecule set ─────────────────────────────────────────────────
    MOLS = load_molecules(args.molecules_py) if args.molecules_py else DEFAULT_MOLECULES
    if args.mol:
        if args.mol not in MOLS:
            print(f"ERROR: molecule '{args.mol}' not found. Available: {list(MOLS)}")
            sys.exit(1)
        mol_subset = {args.mol: MOLS[args.mol]}
    else:
        mol_subset = MOLS

    # ── Orbital basis: thermal-type modes default to PM (supervisor rule) ────
    thermal_like = args.thermal or args.excited_singlet or args.thermal_surface
    if args.localize == "canonical":
        loc = None
    elif args.localize:
        loc = args.localize
    elif thermal_like:
        loc = "pm"
        print("[localize] thermal-type mode with no --localize: defaulting to "
              "Pipek-Mezey localized orbitals ('pm'); use --localize canonical "
              "to opt out.")
    else:
        loc = None

    # ── Thermal mode ─────────────────────────────────────────────────────────
    if args.thermal:
        T_vals = np.linspace(args.T_min, args.T_max, args.T_steps)
        print(f"Thermal GME: T in [{args.T_min}, {args.T_max}] Ha, "
              f"{args.T_steps} points, k_states={args.k_states}")
        for name, mol, cas in _iter_targets(mol_subset, args.cas, args.auto_cas):
            try:
                run_molecular_thermal_analysis(
                    mol_name=name, mol_data=mol, cas=cas, T_values=T_vals,
                    basis=args.basis, max_n_body=args.max_n, min_n_body=3,
                    k_states=args.k_states, cr_restarts=args.cr_restarts,
                    cr_max_iter=args.cr_max_iter, cr_k_max_extra=args.cr_kmax,
                    output_dir=args.output_dir,
                    generate_pdf=not args.no_pdf, save_pkl=args.save_pkl,
                    verbose=not args.quiet,
                    localize=loc,
                )
            except Exception as ex:
                print(f"  [ERROR thermal] {name}: {ex}")
        return

    # ── First-excited-singlet mode ───────────────────────────────────────────
    if args.excited_singlet:
        from hubbard_gme.molecular_surface import run_molecular_excited_singlet
        for name, mol, cas in _iter_targets(mol_subset, args.cas, args.auto_cas):
            try:
                run_molecular_excited_singlet(
                    mol_name=name, mol_data=mol, cas=cas, basis=args.basis,
                    max_n_body=args.max_n, k_states=args.k_states,
                    cr_restarts=args.cr_restarts, cr_max_iter=args.cr_max_iter,
                    cr_k_max_extra=args.cr_kmax, localize=loc,
                    output_dir=args.output_dir, save_pkl=args.save_pkl,
                    generate_pdf=not args.no_pdf, verbose=not args.quiet,
                )
            except Exception as ex:
                print(f"  [ERROR excited-singlet] {name}: {ex}")
        return

    # ── Thermal-surface mode:  C_GME(bond scale s, T)  3D + 2D ──────────────
    if args.thermal_surface:
        from hubbard_gme.molecular_surface import run_molecular_thermal_surface
        scales = np.linspace(args.scale_min, args.scale_max, args.scale_steps)
        T_vals = np.linspace(args.T_min, args.T_max, args.T_steps)
        for name, mol, cas in _iter_targets(mol_subset, args.cas, args.auto_cas):
            try:
                run_molecular_thermal_surface(
                    mol_name=name, mol_data=mol, cas=cas,
                    scales=scales, T_values=T_vals, n_body=args.surface_n,
                    basis=args.basis, k_states=args.k_states,
                    cr_restarts=args.cr_restarts, cr_max_iter=args.cr_max_iter,
                    cr_k_max_extra=args.cr_kmax, localize=loc,
                    output_dir=args.output_dir, save_pkl=args.save_pkl,
                    generate_pdf=not args.no_pdf, verbose=not args.quiet,
                )
            except Exception as ex:
                print(f"  [ERROR thermal-surface] {name}: {ex}")
        return

    # ── T=0 GME mode (default) ───────────────────────────────────────────────
    if args.auto_cas:
        for name, mol, cas in _iter_targets(mol_subset, None, True):
            try:
                run_molecular_analysis(
                    mol_name=name, mol_data=mol, cas=cas, basis=args.basis,
                    max_n_body=args.max_n, min_n_body=args.min_n,
                    output_dir=args.output_dir, cr_method=args.cr_method,
                    cr_restarts=args.cr_restarts, cr_max_iter=args.cr_max_iter,
                    cr_k_max_extra=args.cr_kmax,
                    generate_pdf=not args.no_pdf,
                    generate_rdm_pdf=(not args.no_pdf) and (not args.no_rdm_pdf),
                    save_pkl=args.save_pkl, verbose=not args.quiet,
                    localize=loc,
                )
            except Exception as ex:
                print(f"  [ERROR] {name}: {ex}")
    else:
        run_all_molecules_analysis(
            mol_subset=mol_subset, cas_index=args.cas, basis=args.basis,
            max_n_body=args.max_n, min_n_body=args.min_n,
            output_dir=args.output_dir, cr_method=args.cr_method,
            cr_restarts=args.cr_restarts,
            generate_pdf=not args.no_pdf,
            generate_rdm_pdf=(not args.no_pdf) and (not args.no_rdm_pdf),
            save_pkl=args.save_pkl, verbose=not args.quiet,
            localize=loc,
        )


if __name__ == "__main__":
    main()
