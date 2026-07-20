#!/usr/bin/env python3
"""
visualize_mo.py — visualize the molecular orbitals of a dataset molecule.

Runs a PySCF SCF, optionally LOCALIZES the chosen orbitals (PySCF's built-in
pyscf.lo — Boys / Pipek-Mezey / IBO), writes Gaussian ``.cube`` files, and an
interactive ``py3Dmol`` HTML viewer. By default it renders the ACTIVE-SPACE
orbitals — the ones your n-RDM / GME analysis is built on.

Usage
-----
  # canonical (delocalised) active-space orbitals of CAS index 2
  python visualize_mo.py --mol Benzene --cas 2

  # LOCALISED active-space orbitals (Pipek-Mezey) — one ~pz per atom for benzene
  python visualize_mo.py --mol Benzene --cas 2 --localize pm

  # Boys-localised HOMO/LUMO window
  python visualize_mo.py --mol Ethylene --homo-lumo 3 --localize boys

Outputs -> results/mo_visualization/<Mol>/[canonical|<method>]/
  orbital_<k>.cube          one Gaussian cube per orbital (VMD / Avogadro /
                            PyMOL / ChimeraX for publication-quality isosurfaces)
  <Mol>_orbitals.html       interactive grid viewer (open in a browser)
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def _load_molecule(name, molecules_py):
    if molecules_py:
        from run_molecules_batch import load_molecules
        mols = load_molecules(molecules_py)
    else:
        from hubbard_gme.molecules_data import MOLECULES as mols
    if name not in mols:
        raise SystemExit(f"molecule '{name}' not found. Available: {list(mols)}")
    return mols[name]


def _build_scf(mol_data, basis):
    from pyscf import gto, scf
    atoms = [[el, tuple(xyz)] for (el, xyz) in mol_data["geometry"]]
    charge = int(mol_data.get("charge", 0))
    mult = int(mol_data.get("multiplicity", 1))
    mol = gto.M(atom=atoms, basis=basis, charge=charge, spin=mult - 1, verbose=0)
    mf = scf.RHF(mol) if mol.spin == 0 else scf.ROHF(mol)
    mf.kernel()
    return mol, mf


def _resolve_cas(args, mol_data):
    if args.homo_lumo is not None or args.all_orbitals or args.occupied:
        return None
    cas_list = mol_data["valid_active_spaces"]
    if args.auto_cas:
        from run_molecules_batch import pick_cas
        cas, _ = pick_cas(cas_list)
        if cas is None:
            raise SystemExit("no feasible even-electron CAS; use --cas or --homo-lumo")
        return cas
    if args.cas is not None:
        if args.cas >= len(cas_list):
            raise SystemExit(f"--cas {args.cas} out of range (0..{len(cas_list)-1})")
        return cas_list[args.cas]
    return cas_list[0]


def _select_orbitals(args, mol, mf):
    nmo = mf.mo_coeff.shape[1]
    homo = (mol.nelectron // 2) - 1
    if args.all_orbitals:
        return list(range(nmo)), "all MOs"
    if args.occupied:
        nocc = mol.nelectron // 2
        return list(range(nocc)), "occupied MOs (bonds)"
    if args.homo_lumo is not None:
        w = int(args.homo_lumo)
        return list(range(max(0, homo - w + 1), min(nmo, homo + 1 + w))), \
            f"HOMO-{w-1}..LUMO+{w-1}"
    cas = args.cas_dict
    ncore, norb = int(cas["ncore"]), int(cas["norb_cas"])
    return list(range(ncore, ncore + norb)), \
        f"active space CAS({cas['nele_cas']},{cas['norb_cas']})"


_SHALF_CACHE = {}


def _s_half(mol):
    """S^{1/2} for Löwdin populations, cached per molecule."""
    key = id(mol)
    if key not in _SHALF_CACHE:
        S = mol.intor_symmetric("int1e_ovlp")
        ev, U = np.linalg.eigh(S)
        _SHALF_CACHE[key] = (U * np.sqrt(ev)) @ U.T
    return _SHALF_CACHE[key]


def _atom_weights(mol, coeff):
    """
    Löwdin per-atom weights of an orbital: non-negative and summing to 1
    (unlike Mulliken, which can exceed 1 or go negative). w[A] = fraction of the
    orbital sitting on atom A.
    """
    cl = _s_half(mol) @ coeff
    pop = np.real(cl * cl.conj())          # |Löwdin coeff|^2 >= 0, sums to 1
    w = np.zeros(mol.natm)
    for mu, lab in enumerate(mol.ao_labels(fmt=False)):
        w[lab[0]] += pop[mu]
    return w


def _loc_score(mol, coeffs):
    """
    MAXIMIN localization score = the localization fraction of the WORST-localized
    orbital (the largest single-atom weight of its least-localized orbital).

    We deliberately do NOT use the plain sum-of-charges^2 (Pipek-Mezey objective):
    that rewards a few very-localized orbitals + some delocalized ones (e.g.
    benzene's 4-localized + 2-delocalized minimum). Maximin instead rewards the
    solution where *every* orbital is localized, which is the chemically useful
    one (each pi orbital on its own atom).
    """
    worst = 1.0
    for k in range(coeffs.shape[1]):
        w = _atom_weights(mol, coeffs[:, k])
        worst = min(worst, float(w.max() / w.sum()))
    return worst


def _localize(mol, mf, indices, method, n_tries=12, seed=0,
              pm_exponent=4, pop_method="meta_lowdin"):
    """
    Localize the given MOs, returning the BEST of several restarts. Localization
    is non-convex and can land in poor local minima (e.g. benzene's symmetric pi
    space → 4 localized + 2 delocalized). The result with the highest maximin
    score (see _loc_score) is kept.

    Restart strategy (t = try index):
      t == 0 : PySCF's default 'atomic' initial guess (projected AOs — usually
               already the best for PM).
      t >= 1 : a RANDOM unitary rotation of the orbitals, passed DIRECTLY to
               kernel(C) so it is genuinely used as the starting point.
               (kernel() without an argument would rebuild the 'atomic' guess
               from the orbital SPAN — identical for every rotation — silently
               ignoring the restart!)

    Extra PM knobs (pm_* / pop_method ignored for boys/ibo):
      pm_exponent : 2 (standard) or 4 — the PM functional maximizes
               sum_A |Q_A|^p; p=4 penalizes delocalization tails harder and
               often localizes stubborn (aromatic) orbitals better.
      pop_method : 'meta_lowdin' (default), 'mulliken', 'lowdin', 'iao',
               'becke' — the atomic-population scheme PM maximizes. 'iao' and
               'becke' are the most basis-set-robust.
    """
    from pyscf import lo
    C0 = mf.mo_coeff[:, indices]
    k = C0.shape[1]

    def _make(C):
        if method == "boys":
            return lo.Boys(mol, C)
        if method == "pm":
            loc = lo.PipekMezey(mol, C, mf=mf)   # mf needed for iao/becke pops
            loc.exponent = int(pm_exponent)
            loc.pop_method = pop_method
            loc.conv_tol = 1e-9          # tighter than the 1e-6 default
            loc.max_cycle = 300
            return loc
        raise RuntimeError(f"unknown --localize method {method!r}")

    if method == "ibo":                   # IBO has its own construction
        a = lo.vec_lowdin(lo.iao.iao(mol, C0), mf.get_ovlp())
        return lo.ibo.ibo(mol, C0, iaos=a)

    rng = np.random.default_rng(seed)
    best, best_s = None, -np.inf
    for t in range(max(1, n_tries)):
        try:
            if t == 0:
                loc = _make(C0).kernel()          # PySCF 'atomic' guess
            else:
                Q = np.linalg.qr(rng.standard_normal((k, k)))[0]
                C = C0 @ Q
                loc = _make(C0).kernel(C)         # rotated C used as the guess
        except Exception:
            continue
        s = _loc_score(mol, loc)
        if s > best_s:
            best, best_s = loc, s
    if best is None:
        raise RuntimeError(f"localization ({method}) failed on all {n_tries} tries")
    return best


def _atom_of(mol, coeff):
    """Atom carrying the largest Löwdin weight of an orbital, and its % (0–100)."""
    w = _atom_weights(mol, coeff)
    a = int(np.argmax(w))
    return f"{mol.atom_symbol(a)}{a}", 100 * w[a] / w.sum()


def _write_cubes(mol, coeffs, outdir, prefix=""):
    from pyscf.tools import cubegen
    paths = []
    pre = f"{prefix}_" if prefix else ""
    for k in range(coeffs.shape[1]):
        p = os.path.join(outdir, f"{pre}orbital_{k}.cube")
        cubegen.orbital(mol, p, coeffs[:, k])
        paths.append(p)
    return paths


def _write_html(mol, coeffs, cube_paths, out_html, title, sublabels,
                isoval=0.02):
    import py3Dmol
    xyz = mol.tostring(format="xyz")
    k = coeffs.shape[1]
    ncol = min(3, k)
    nrow = math.ceil(k / ncol)
    view = py3Dmol.view(viewergrid=(nrow, ncol), width=380 * ncol, height=340 * nrow)
    for j in range(k):
        r, c = divmod(j, ncol)
        cube = open(cube_paths[j]).read()
        view.addModel(xyz, "xyz", viewer=(r, c))
        view.setStyle({"stick": {"radius": 0.1}}, viewer=(r, c))
        view.addVolumetricData(cube, "cube",
                               {"isoval": isoval, "color": "blue", "opacity": 0.85},
                               viewer=(r, c))
        view.addVolumetricData(cube, "cube",
                               {"isoval": -isoval, "color": "red", "opacity": 0.85},
                               viewer=(r, c))
        view.addLabel(sublabels[j], {"fontSize": 11, "backgroundOpacity": 0.6},
                      viewer=(r, c))
        view.zoomTo(viewer=(r, c))
    banner = (f"<div style='font-family:sans-serif;padding:8px 12px'>"
              f"<b>{title}</b> — blue = +, red = − lobe (isovalue {isoval}). "
              f"Drag to rotate, scroll to zoom.</div>")
    with open(out_html, "w") as f:
        f.write(banner + view._make_html())
    return out_html


def main():
    ap = argparse.ArgumentParser(description="Visualize molecular orbitals (cube + HTML).")
    ap.add_argument("--mol", required=True)
    ap.add_argument("--molecules-py", default=None)
    ap.add_argument("--basis", default="cc-pVDZ")
    ap.add_argument("--localize", choices=["boys", "pm", "ibo"], default=None,
                    help="localize the selected orbitals (default: canonical)")
    ap.add_argument("--loc-tries", type=int, default=12, dest="loc_tries",
                    help="localization restarts; keep the most-localized result "
                         "(default 12 — avoids bad local minima)")
    ap.add_argument("--pm-exponent", type=int, choices=[2, 4], default=4,
                    dest="pm_exponent",
                    help="PM functional power p in sum|Q_A|^p (default 4 — "
                         "benchmarked best on benzene pi; 2 = textbook PM)")
    ap.add_argument("--pop-method", default="meta_lowdin", dest="pop_method",
                    choices=["meta_lowdin", "mulliken", "lowdin", "iao", "becke"],
                    help="atomic-population scheme PM maximizes "
                         "(default meta_lowdin; iao/becke are most basis-robust)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--cas", type=int, default=None)
    g.add_argument("--auto-cas", action="store_true")
    g.add_argument("--homo-lumo", type=int, default=None, metavar="W")
    g.add_argument("--occupied", action="store_true",
                   help="render all OCCUPIED MOs (the bonds: cores + sigma + pi)")
    g.add_argument("--all-orbitals", action="store_true")
    ap.add_argument("--isoval", type=float, default=0.02,
                    help="isosurface value for the HTML viewer (default 0.02; "
                         "use ~0.05 to trim tails and see the localized core)")
    ap.add_argument("--output-dir", default="results/mo_visualization")
    args = ap.parse_args()

    mol_data = _load_molecule(args.mol, args.molecules_py)
    args.cas_dict = _resolve_cas(args, mol_data)

    print(f"[scf] {args.mol}  basis={args.basis} …")
    mol, mf = _build_scf(mol_data, args.basis)
    indices, desc = _select_orbitals(args, mol, mf)

    if args.localize:
        coeffs = _localize(mol, mf, indices, args.localize, n_tries=args.loc_tries,
                           pm_exponent=args.pm_exponent, pop_method=args.pop_method)
        kind = {"boys": "Boys", "pm": "Pipek-Mezey", "ibo": "IBO"}[args.localize]
        desc = f"{desc}  [{kind}-localised]"
        sublabels, nloc = [], 0
        for j in range(coeffs.shape[1]):
            atom, pct = _atom_of(mol, coeffs[:, j])
            if pct >= 40:                    # concentrated on one atom
                sublabels.append(f"loc {j}  on {atom} ({pct:.0f}%)")
                nloc += 1
            else:                            # spread out — don't pretend it's on one atom
                sublabels.append(f"loc {j}  delocalized (max {pct:.0f}% on {atom})")
        print(f"[loc] {kind}: kept best of {args.loc_tries} restarts — "
              f"{nloc}/{coeffs.shape[1]} orbitals well-localized (>=40% on one atom)")
        tag = args.localize
    else:
        coeffs = mf.mo_coeff[:, indices]
        homo = (mol.nelectron // 2) - 1
        sublabels = [
            f"MO {i}  {mf.mo_energy[i]:+.3f} Ha"
            + ("  HOMO" if i == homo else "  LUMO" if i == homo + 1 else "")
            for i in indices]
        tag = "canonical"

    # selection tag so different orbital sets never overwrite each other
    if args.all_orbitals:
        sel = "all"
    elif args.occupied:
        sel = "occupied"
    elif args.homo_lumo is not None:
        sel = f"homolumo{args.homo_lumo}"
    elif args.auto_cas:
        sel = "casauto"
    elif args.cas is not None:
        sel = f"cas{args.cas}"
    else:
        sel = "cas0"

    print(f"[mo ] {coeffs.shape[1]} orbitals: {desc}")
    # folder AND filenames carry both the selection and the method (pm/boys/…)
    stem = f"{sel}_{tag}"                     # e.g. cas2_pm , occupied_boys
    outdir = os.path.join(args.output_dir, args.mol, stem)
    os.makedirs(outdir, exist_ok=True)
    cubes = _write_cubes(mol, coeffs, outdir, prefix=stem)
    out_html = os.path.join(outdir, f"{args.mol}_{stem}_orbitals.html")
    _write_html(mol, coeffs, cubes, out_html, f"{args.mol} — {desc}", sublabels,
                isoval=args.isoval)

    print(f"[out] {len(cubes)} cube files -> {outdir}/orbital_*.cube")
    print(f"[out] interactive viewer -> {out_html}")
    if args.localize:
        print("      each orbital is labelled with the atom it localises on.")


if __name__ == "__main__":
    main()
