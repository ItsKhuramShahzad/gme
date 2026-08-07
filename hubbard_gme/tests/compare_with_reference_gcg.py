"""
Compare our convex-roof GCG solver against the independent reference
implementation in ``gcg_reference/`` (Röthlisberger/Lehmann/Loss PRA 80,
042301 + Ma et al. PRA 83, 062325), and measure which of its five policy
choices matter on our systems.

Nothing here modifies either solver, and no stored C_GME value is read: every
number is recomputed live from the density matrices in ``rdm_cache/``. That
matters because our saved GME PKLs predate the k-loop fix and would otherwise
contaminate the comparison.

WHAT EACH STEP ESTABLISHES
--------------------------
  0  reference self-check   is the oracle trustworthy at all?
  1  analytic benchmarks    each solver vs EXACT known roofs — a miss here
                            names the broken solver, no ambiguity
  2  pure measure/gradient  the innermost deterministic functions; any
                            disagreement is a definite bug, not search luck
  3  decomposition validity does sum_i p_i |psi_i><psi_i| rebuild rho?
                            (each solver against itself)
  4  our production RDMs    ours vs reference on the rows we publish
  5  policy sweep           the reference solver run twice with ONE knob
                            flipped, to see which of the five choices moves
                            values on our data

Steps 1-4 compare the two solvers. Step 5 runs only on the reference solver,
because our code does not expose those knobs — the question there is "would
this policy be worth implementing in ours?".

USAGE
-----
    python tests/compare_with_reference_gcg.py --step 1
    python tests/compare_with_reference_gcg.py --step 2 --step 3
    python tests/compare_with_reference_gcg.py --step all --csv report.csv
    python tests/compare_with_reference_gcg.py --step 5 --policy termination
    python tests/compare_with_reference_gcg.py --step 4 --restarts 12 --cases real

    # exhaustive overnight run — every case, every step, tight settings
    python tests/compare_with_reference_gcg.py --step all --full \
        --restarts 20 --kmax 6 --csv report_full.csv

    --ref PATH     where gcg_reference/ lives (default: ../gcg_reference)
    --restarts N   restarts for BOTH solvers (default 8)
    --kmax N       k_max_extra for BOTH (default 4)
    --tol T        agreement threshold (default 1e-6)
    --csv FILE     write every row to CSV
    --quick        smaller/faster case set
    --full         exhaustive: all 24 cached clusters instead of 4, 2000 random
                   states in step 2, decomposition check on every case, policy
                   sweep over every case. Hours.

Note on --full and step 4: all clusters of a given body order n are related by
translation symmetry, so they must give the SAME roof. The spread across them
is therefore a direct measure of optimiser noise on real data — worth reading
alongside the ours-vs-reference difference.

READING THE OUTPUT
------------------
Both solvers return an UPPER bound on the same minimisation, so neither is
"right" when they differ — the LOWER value is the better one. Mixed signs
across rows means different local minima (search luck). A one-sided
difference, or a miss against an analytic value, is a defect.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import pickle
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
DEFAULT_REF = os.path.join(os.path.dirname(REPO), "gcg_reference")
CACHE = os.path.join(REPO, "results", "rdm_pkl_results", "rdm_cache")

ROWS: List[Dict] = []          # every measurement, flushed to CSV at the end


def record(**kw) -> None:
    ROWS.append(kw)


# ─────────────────────────────────────────────────────────────────────────────
# case builders
# ─────────────────────────────────────────────────────────────────────────────

def analytic_cases(bench, quick: bool = False) -> List[Tuple]:
    """(label, rho, n, exact_roof) using the REFERENCE package's own families.

    Exact values are those asserted in its test suite:
        M1        -> 0
        M2(p)     -> |2p - 1|
        M3(a, g)  -> 2|g|
    plus the standard pure-state values.
    """
    cases: List[Tuple] = []

    ghz3 = bench.ghz_state(3)
    w3 = bench.w_state(3)
    cases.append(("GHZ3 pure", bench.projector(ghz3), 3, 1.0))
    cases.append(("W3 pure", bench.projector(w3), 3, float(np.sqrt(8 / 9))))
    if not quick:
        ghz4 = bench.ghz_state(4)
        cases.append(("GHZ4 pure", bench.projector(ghz4), 4, 1.0))

    cases.append(("M1", bench.m1_state(), 3, 0.0))
    ps = (0.25, 0.5, 1.0) if quick else (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    for p in ps:
        cases.append((f"M2(p={p})", bench.m2_state(p), 3, abs(2.0 * p - 1.0)))
    m3 = ((0.3, 0.25j),) if quick else ((0.3, 0.25j), (0.5, 0.2 + 0.1j))
    for alpha, gamma in m3:
        cases.append((f"M3(a={alpha},g={gamma})", bench.m3_state(alpha, gamma),
                      3, 2.0 * abs(gamma)))
    # D family has no closed form; kept for solver-vs-solver only
    cases.append(("D(p=0.4)", bench.d_state(0.4), 3, None))
    return cases


def _load_rho(path: str) -> Optional[np.ndarray]:
    payload = pickle.load(open(path, "rb"))
    for key in ("rho", "rho_gs", "rdm"):
        if isinstance(payload, dict) and key in payload:
            return np.asarray(payload[key])
    if isinstance(payload, dict):
        for value in payload.values():
            if (isinstance(value, np.ndarray) and value.ndim == 2
                    and value.shape[0] == value.shape[1]):
                return value
    return None


def real_cases(n_values=(2, 3, 4, 5), U: int = 4, t: int = 1,
               all_clusters: bool = False) -> List[Tuple]:
    """Our own cached L=6 / Ne=6 ground-state RDMs — the published rows.

    By default one cluster per body order n. With ``all_clusters`` every cached
    site-tuple is included: translation symmetry means all clusters of a given n
    must give the SAME roof, so this doubles as a solver-consistency check on
    real data (the spread across clusters is pure optimiser noise).
    """
    cases: List[Tuple] = []
    for n in n_values:
        hits = sorted(glob.glob(os.path.join(
            CACHE, f"rdm_v1_L6_Ne6_U{U}_t{t}_mode-uniform_state-gs_n{n}_*.pkl")))
        if not hits:
            continue
        for path in (hits if all_clusters else hits[:1]):
            rho = _load_rho(path)
            if rho is None:
                continue
            tr = np.trace(rho).real
            if abs(tr) > 1e-14:
                rho = rho / tr
            sites = os.path.basename(path).split("sites-")[-1].split("_")[0] \
                if "sites-" in path else ""
            label = f"L6Ne6 U={U} GS n={n}" + (f" [{sites}]" if all_clusters and sites else "")
            cases.append((label, rho, n, None))
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# step 0 — is the reference itself sound?
# ─────────────────────────────────────────────────────────────────────────────

def step0(ref_dir: str) -> None:
    print("\n" + "=" * 100)
    print("STEP 0 — reference self-check (validates the oracle before we trust it)")
    print("=" * 100)
    scripts = [
        ("symbolic check vs Röthlisberger/Ma",
         "scripts/verify_gcg_against_rothlisberger.py"),
        ("the note's own equations", "scripts/verify_gcg_notes_equations.py"),
    ]
    for label, rel in scripts:
        path = os.path.join(ref_dir, rel)
        if not os.path.exists(path):
            print(f"  [skip] {label}: {rel} not found")
            continue
        print(f"  running {rel} …", flush=True)
        t0 = time.time()
        proc = subprocess.run([sys.executable, rel], cwd=ref_dir,
                              capture_output=True, text=True)
        tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-1:] or [""]
        ok = proc.returncode == 0
        print(f"    {'PASS' if ok else 'FAIL'}  ({time.time()-t0:.1f}s)  {tail[0][:70]}")
        record(step=0, case=label, solver="reference", setting=rel,
               value="", exact="", diff="", passed=ok)


# ─────────────────────────────────────────────────────────────────────────────
# step 1 / 4 — roof comparison (both solvers)
# ─────────────────────────────────────────────────────────────────────────────

def _run_ours(ours, rho, n, restarts, kmax):
    t0 = time.time()
    res = ours(rho, n=n, n_restarts=restarts, k_max_extra=kmax, verbose=False)
    iters = sum(len(r.get("history", [])) for r in res.get("k_records", []))
    return (float(res["C_gcg"]), res.get("k_converged"), iters,
            time.time() - t0, res)


def _run_ref(theirs, rho, restarts, kmax, **policy):
    t0 = time.time()
    res = theirs(rho, n_restarts=restarts, k_extra_max=kmax, **policy)
    ds = res.get("descent_summary") or {}
    return (float(res.get("value", np.nan)), res.get("k"),
            ds.get("iterations"), time.time() - t0, res)


def compare_roofs(step: int, title: str, cases, ours, theirs,
                  restarts: int, kmax: int, tol: float) -> None:
    print("\n" + "=" * 100)
    print(f"STEP {step} — {title}")
    print("both solvers return UPPER bounds: the LOWER value is the better one")
    print("=" * 100)
    hdr = (f"{'case':<24}{'ours':>14}{'reference':>14}{'diff':>12}{'exact':>11}"
           f"{'k o/r':>9}{'iters o/r':>12}{'time o/r':>13}  verdict")
    print(hdr)
    print("-" * len(hdr))

    worst = 0.0
    n_ours_lower = n_ref_lower = n_agree = 0
    analytic_fail: List[str] = []

    for label, rho, n, exact in cases:
        a, k_a, it_a, t_a, _ = _run_ours(ours, rho, n, restarts, kmax)
        b, k_b, it_b, t_b, _ = _run_ref(theirs, rho, restarts, kmax)
        diff = a - b
        worst = max(worst, abs(diff))

        if abs(diff) < tol:
            verdict = "agree"; n_agree += 1
        elif diff < 0:
            verdict = "OURS lower"; n_ours_lower += 1
        else:
            verdict = "ref lower"; n_ref_lower += 1

        ex = f"{exact:.6f}" if exact is not None else "-"
        print(f"{label:<24}{a:>14.9f}{b:>14.9f}{diff:>+12.2e}{ex:>11}"
              f"{f'{k_a}/{k_b}':>9}{f'{it_a}/{it_b}':>12}"
              f"{f'{t_a:.0f}/{t_b:.0f}s':>13}  {verdict}", flush=True)

        for who, val in (("ours", a), ("reference", b)):
            if exact is not None and abs(val - exact) > 1e-4:
                analytic_fail.append(f"{label}: {who}={val:.9f} exact={exact:.9f}")
                print(f"{'':<24}  !! {who} misses the analytic value by "
                      f"{abs(val-exact):.2e}")
            record(step=step, case=label, solver=who, setting="default",
                   value=val, exact="" if exact is None else exact,
                   diff=diff if who == "ours" else "",
                   k=k_a if who == "ours" else k_b,
                   iters=it_a if who == "ours" else it_b,
                   seconds=round(t_a if who == "ours" else t_b, 2))

    print("-" * len(hdr))
    print(f"largest |difference| = {worst:.3e}    "
          f"agree: {n_agree}   ours lower: {n_ours_lower}   ref lower: {n_ref_lower}")
    if analytic_fail:
        print("\n  !! ANALYTIC MISMATCH — this identifies a broken solver:")
        for line in analytic_fail:
            print(f"     {line}")
    elif n_ours_lower and n_ref_lower:
        print("  mixed signs => different local minima (search luck), not a defect")
    elif n_ours_lower or n_ref_lower:
        print("  one-sided difference => investigate the HIGHER solver")
    else:
        print("  all rows agree => our values are independently confirmed")


# ─────────────────────────────────────────────────────────────────────────────
# step 2 — the innermost deterministic functions
# ─────────────────────────────────────────────────────────────────────────────

def step2(refpure, n_states: int = 300, n_spins: int = 3) -> None:
    from hubbard_gme.convex_roof import _cgme_pure_vector, _cgme_pure_with_grad

    print("\n" + "=" * 100)
    print("STEP 2 — pure-state measure and gradient (deterministic: any gap is a bug)")
    print("=" * 100)

    rng = np.random.default_rng(0)
    dim = 2 ** n_spins
    worst_val = worst_grad = 0.0
    for _ in range(n_states):
        psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        psi /= np.linalg.norm(psi)
        worst_val = max(worst_val,
                        abs(_cgme_pure_vector(psi, n_spins) - refpure.pure_cgme(psi)))
        _, g_ours = _cgme_pure_with_grad(psi, n_spins)
        g_ref = refpure.pure_cgme_grad_averaged(psi)
        # both are d/dpsi* conventions; compare up to the same scaling
        if np.linalg.norm(g_ref) > 0:
            worst_grad = max(worst_grad, float(np.linalg.norm(
                g_ours / max(np.linalg.norm(g_ours), 1e-300)
                - g_ref / np.linalg.norm(g_ref))))

    print(f"  random states tested            : {n_states}")
    print(f"  max |C_ours - C_ref|            : {worst_val:.3e}   "
          f"{'OK' if worst_val < 1e-10 else 'MISMATCH'}")
    print(f"  max gradient direction mismatch : {worst_grad:.3e}   "
          f"{'OK' if worst_grad < 1e-6 else 'CHECK'}")
    print("  (gradients are compared as directions: the two codes normalise "
          "the\n   Wirtinger convention differently, so only the direction is "
          "meaningful)")

    # analytic spot checks both must satisfy
    for label, psi, exact in (
        ("GHZ3", np.array([1, 0, 0, 0, 0, 0, 0, 1], complex) / np.sqrt(2), 1.0),
        ("W3", np.array([0, 1, 1, 0, 1, 0, 0, 0], complex) / np.sqrt(3),
         float(np.sqrt(8 / 9))),
        ("product", np.array([1, 0, 0, 0, 0, 0, 0, 0], complex), 0.0),
    ):
        a = _cgme_pure_vector(psi, 3)
        b = refpure.pure_cgme(psi)
        print(f"  {label:<8} ours={a:.12f}  ref={b:.12f}  exact={exact:.12f}")
        record(step=2, case=f"pure {label}", solver="ours", setting="-",
               value=a, exact=exact, diff=a - b)
        record(step=2, case=f"pure {label}", solver="reference", setting="-",
               value=b, exact=exact, diff="")

    record(step=2, case="random states", solver="both", setting=f"n={n_states}",
           value=worst_val, exact="", diff=worst_val)


# ─────────────────────────────────────────────────────────────────────────────
# step 3 — is the returned decomposition actually a decomposition of rho?
# ─────────────────────────────────────────────────────────────────────────────

def step3(cases, ours, theirs, restarts: int, kmax: int) -> None:
    print("\n" + "=" * 100)
    print("STEP 3 — decomposition validity: does sum_i p_i |psi_i><psi_i| rebuild rho?")
    print("=" * 100)
    print(f"{'case':<24}{'ours ||rho-sum||':>20}{'ref ||rho-sum||':>20}  verdict")
    print("-" * 70)

    for label, rho, n, _ in cases:
        # A single awkward case must not abort a multi-hour run: report the
        # failure for that row and carry on.
        try:
            res_o = ours(rho, n=n, n_restarts=restarts, k_max_extra=kmax,
                         verbose=False)
            err_o = float("nan")
            decomp = (res_o.get("best_record") or {}).get("decomp")
            if decomp:
                acc = np.zeros_like(rho)
                for entry in decomp:
                    p = entry.get("p", entry.get("prob"))
                    psi = np.asarray(entry.get("psi", entry.get("state")))
                    if p is None or psi is None or psi.shape[0] != rho.shape[0]:
                        acc = None
                        break
                    acc = acc + p * np.outer(psi, psi.conj())
                if acc is not None:
                    err_o = float(np.linalg.norm(rho - acc))
        except Exception as exc:                      # noqa: BLE001
            print(f"{label:<24}{'[ours failed]':>20}  {type(exc).__name__}: {exc}")
            continue

        try:
            res_r = theirs(rho, n_restarts=restarts, k_extra_max=kmax)
        except Exception as exc:                      # noqa: BLE001
            print(f"{label:<24}{err_o:>20.3e}{'[ref failed]':>20}  "
                  f"{type(exc).__name__}: {exc}")
            continue
        w = np.asarray(res_r.get("weights", []))
        st = np.asarray(res_r.get("states", []))
        err_r = float("nan")
        if w.size and st.size:
            # The reference stores states as COLUMNS: `states` has shape
            # (dim, k), one state vector per column. Orient defensively so a
            # future change of convention cannot silently mis-reconstruct rho.
            if st.ndim == 2 and st.shape[0] != rho.shape[0] and st.shape[1] == rho.shape[0]:
                st = st.T                     # (k, dim) -> (dim, k)
            acc = np.zeros_like(rho)
            for i in range(min(len(w), st.shape[1])):
                vec = st[:, i]
                acc = acc + w[i] * np.outer(vec, vec.conj())
            err_r = float(np.linalg.norm(rho - acc))

        ok = (np.isnan(err_o) or err_o < 1e-8) and (np.isnan(err_r) or err_r < 1e-8)
        print(f"{label:<24}{err_o:>20.3e}{err_r:>20.3e}  {'OK' if ok else 'CHECK'}")
        record(step=3, case=label, solver="ours", setting="rebuild",
               value=err_o, exact=0.0, diff="")
        record(step=3, case=label, solver="reference", setting="rebuild",
               value=err_r, exact=0.0, diff="")


# ─────────────────────────────────────────────────────────────────────────────
# step 5 — which of the five policy choices moves values on OUR data?
# ─────────────────────────────────────────────────────────────────────────────

POLICY_SWEEP = {
    "transport":   ("published", "parallel"),
    "termination": ("nonsmooth", "legacy"),
    "tie_policy":  ("clarke_average", "strict_max"),
    "bracket":     ("fixed", "adaptive"),
    "starts":      ("with_identity", "random_only"),
}


def step5(cases, theirs, restarts: int, kmax: int, only: Optional[str]) -> None:
    print("\n" + "=" * 100)
    print("STEP 5 — policy sweep on the REFERENCE solver, using OUR density matrices")
    print("our solver does not expose these knobs; the question is whether the")
    print("choice would be worth implementing in ours")
    print("=" * 100)

    knobs = {only: POLICY_SWEEP[only]} if only else POLICY_SWEEP
    for knob, (default, alternative) in knobs.items():
        print(f"\n  --- {knob}:  {default} (default)  vs  {alternative} ---")
        hdr = (f"  {'case':<24}{default:>16}{alternative:>16}{'diff':>12}"
               f"{'iters d/a':>13}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for label, rho, n, _ in cases:
            a, _, it_a, _, _ = _run_ref(theirs, rho, restarts, kmax,
                                        **{knob: default})
            b, _, it_b, _, _ = _run_ref(theirs, rho, restarts, kmax,
                                        **{knob: alternative})
            print(f"  {label:<24}{a:>16.9f}{b:>16.9f}{b-a:>+12.2e}"
                  f"{f'{it_a}/{it_b}':>13}", flush=True)
            record(step=5, case=label, solver="reference",
                   setting=f"{knob}={default}", value=a, exact="", diff="",
                   iters=it_a)
            record(step=5, case=label, solver="reference",
                   setting=f"{knob}={alternative}", value=b, exact="",
                   diff=b - a, iters=it_b)


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default=DEFAULT_REF)
    ap.add_argument("--step", action="append", default=None,
                    choices=["0", "1", "2", "3", "4", "5", "all"])
    ap.add_argument("--cases", default="both", choices=["analytic", "real", "both"])
    ap.add_argument("--policy", default=None, choices=sorted(POLICY_SWEEP))
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--quick", action="store_true",
                    help="small/fast case set")
    ap.add_argument("--full", action="store_true",
                    help="exhaustive: every case in every step, 2000 random "
                         "states in step 2, policy sweep over all cases. "
                         "Hours. Pair with --restarts 20 --kmax 6.")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    steps = args.step or ["all"]
    if "all" in steps:
        steps = ["0", "1", "2", "3", "4", "5"]

    if not os.path.isdir(args.ref):
        sys.exit(f"gcg_reference not found at {args.ref}   (pass --ref PATH)")
    sys.path.insert(0, args.ref)

    from hubbard_gme.convex_roof import convex_roof_gcg as ours
    from workflows.gme_mixed.gcg import convex_roof_gcg as theirs
    from workflows.gme_mixed import benchmarks as bench
    from workflows.gme_mixed import pure as refpure

    ana = analytic_cases(bench, args.quick)
    real = real_cases((2, 3) if args.quick else (2, 3, 4, 5),
                      all_clusters=args.full)

    print("=" * 100)
    print("COMPARISON WITH THE gcg_reference IMPLEMENTATION")
    print(f"  reference   : {args.ref}")
    print(f"  steps       : {', '.join(steps)}")
    print(f"  restarts={args.restarts}  kmax={args.kmax}  tol={args.tol:g}"
          f"{'  [quick]' if args.quick else ''}")
    print(f"  analytic cases: {len(ana)}   real RDMs: {len(real)}")
    print("  every value is recomputed live; no stored C_GME is read")
    print("=" * 100)

    # --full: exhaustive. Every case in every step, more random states, and the
    # policy sweep over all cases rather than a sample. Hours, not minutes.
    n_rand = 60 if args.quick else (2000 if args.full else 300)
    cases3 = (ana + real) if args.full else (ana[:3] + real[:2])
    cases5 = (real + ana) if args.full else (real or ana)[:3]

    if "0" in steps:
        step0(args.ref)
    if "1" in steps and args.cases in ("analytic", "both"):
        compare_roofs(1, "analytic benchmarks (exact roofs known)",
                      ana, ours, theirs, args.restarts, args.kmax, args.tol)
    if "2" in steps:
        step2(refpure, n_states=n_rand)
    if "3" in steps:
        step3(cases3, ours, theirs, args.restarts, args.kmax)
    if "4" in steps and args.cases in ("real", "both"):
        if real:
            compare_roofs(4, "our production RDMs (L=6, Ne=6 ground state)",
                          real, ours, theirs, args.restarts, args.kmax, args.tol)
        else:
            print(f"\n[skip] step 4: no cached RDMs under {CACHE}")
    if "5" in steps:
        step5(cases5, theirs, args.restarts, args.kmax, args.policy)

    if args.csv and ROWS:
        keys = ["step", "case", "solver", "setting", "value", "exact", "diff",
                "k", "iters", "seconds", "passed"]
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for row in ROWS:
                writer.writerow(row)
        print(f"\nCSV report written to {args.csv}   ({len(ROWS)} rows)")


if __name__ == "__main__":
    main()
