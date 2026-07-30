#!/usr/bin/env python3
"""
make_cluster_table.py — full ground-state (T=0) cluster table from STORED data.

Reads the already-computed RDMs out of an rdm_bank PKL (no diagonalization, no
GCG, nothing re-run) and reports, for every body order n:

    n | dim | purity | entropy | <S^2> | rank | C_GME | parity

purity/entropy/<S^2>/rank come from the stored ground-state RDM;
C_GME is read from the matching t-scan (or U-scan) PKL.

No total-spin S column: these cluster RDMs are MIXED states, so
<S^2> = sum_S p_S S(S+1) averages over spin sectors and inverting
S(S+1) would give a non-physical fractional S. (spin.py derives S
correctly because it acts on a pure many-body eigenstate.)

Usage
-----
  python make_cluster_table.py --bank <rdmbank_*.pkl> --scan <tscan_*.pkl>
  python make_cluster_table.py --bank ... --scan ... --tex docs/table.tex

Memory note: an rdm_bank stores RDMs for ALL eigenstates, so the L=8 banks are
~17 GB. Run this where you have enough RAM (it loads the bank once).
"""
import argparse, os, pickle
import numpy as np


# ── S^2 operator for n spins (bit k = site k, 0 = up, 1 = down) ──────────────
def s2_matrix(n: int) -> np.ndarray:
    I2 = np.eye(2)
    sz = np.diag([0.5, -0.5])
    sp = np.array([[0.0, 1.0], [0.0, 0.0]])
    sm = sp.T

    def site_op(op, k):
        mats = [I2] * n
        mats[n - 1 - k] = op          # bit k -> position n-1-k in the kron chain
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    Sz = [site_op(sz, k) for k in range(n)]
    Sp = [site_op(sp, k) for k in range(n)]
    Sm = [site_op(sm, k) for k in range(n)]
    S2 = np.zeros((2 ** n, 2 ** n))
    for i in range(n):
        for j in range(n):
            S2 += Sz[i] @ Sz[j] + 0.5 * (Sp[i] @ Sm[j] + Sm[i] @ Sp[j])
    return S2


def props(rho: np.ndarray, n: int) -> dict:
    rho = np.asarray(rho)
    tr = float(np.real(np.trace(rho)))
    if abs(tr) > 1e-14:
        rho = rho / tr
    ev = np.real(np.linalg.eigvalsh(rho))
    # clip to [0,1]: eigenvalues can land at 1+3e-16, which makes log2 positive
    # and the entropy print as "-0.0000" instead of exactly 0 for a pure state
    pos = np.clip(ev[ev > 1e-12], 0.0, 1.0)
    entropy = float(-(pos * np.log2(pos)).sum())
    # NOTE: <S^2> is reported, but NOT a derived S. These RDMs are mixed
    # states, so <S^2> = sum_S p_S S(S+1) is an average over spin sectors;
    # inverting S(S+1) would give a non-physical fractional "S". (The
    # inversion in spin.py IS valid there — it acts on a pure eigenstate.)
    s2 = float(np.real(np.trace(rho @ s2_matrix(n))))
    return dict(purity=float(np.real(np.trace(rho @ rho))),
                entropy=max(0.0, entropy),                   # exact 0 for pure
                rank=int((ev > 1e-10).sum()),
                s2=max(0.0, s2))                             # S^2 >= 0 always


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True, help="rdmbank_*.pkl (stored RDMs)")
    ap.add_argument("--scan", help="tscan_*/uscan_*.pkl for the C_GME column")
    ap.add_argument("--state", type=int, default=0,
                    help="eigenstate index (0 = ground state, default)")
    ap.add_argument("--tex", help="also write a LaTeX table here")
    args = ap.parse_args()

    print(f"loading bank: {args.bank}", flush=True)
    b = pickle.load(open(args.bank, "rb"))
    bank = b["bank"]
    E = np.asarray(b.get("energies", []))
    if len(E):
        print(f"  {len(E)} eigenstates stored, E0 = {E[0]:.6f}")

    # C_GME column from the scan PKL (index 0 = first grid point).
    # For n=2 the scan also stores the analytic reference:
    #   C_wootters_n2   = Wootters concurrence  (== the GCG convex roof for n=2)
    # The Werner form sqrt(1-sqrt(1-C^2)) is NOT a valid n=2 GME and is not reported.
    C, c_woot = {}, None
    L = U = tt = None
    if args.scan:
        s = pickle.load(open(args.scan, "rb"))
        r = s.get("result", s)
        C = {n: float(np.asarray(v)[0]) for n, v in r["C_GME"].items()}
        if "C_wootters_n2" in r:
            c_woot = float(np.asarray(r["C_wootters_n2"])[0])
        si = r.get("system_info", {}) or {}
        L = si.get("L")
        if "t_values" in r:                       # t-scan: U fixed, t varies
            U, tt = si.get("U"), float(np.asarray(r["t_values"])[0])
        else:                                     # U-scan: t fixed, U varies
            U, tt = float(np.asarray(r["U_values"])[0]), si.get("t")
        print(f"  parameters: L={L}, U={U}, t={tt}"
              + (f"  (U/t={U/tt:.4g})" if U and tt else ""))
        print(f"  C_GME read from scan at grid point 0")

    rows = []
    for n in sorted(bank):
        rho = np.asarray(bank[n])[args.state]
        p = props(rho, n)
        rows.append((n, 2 ** n, p["purity"], p["entropy"], p["s2"],
                     p["rank"], C.get(n, float("nan"))))

    print(f"\n{'n':>2} {'dim':>5} {'purity':>9} {'entropy':>9} {'<S^2>':>8} "
          f"{'rank':>5} {'C_GME':>9}  parity")
    print("-" * 70)
    for n, dim, pur, ent, s2, rk, c in rows:
        cs = f"{c:.4f}" if np.isfinite(c) else "  n/a"
        print(f"{n:>2} {dim:>5} {pur:>9.4f} {ent:>9.4f} {s2:>8.4f} "
              f"{rk:>5} {cs:>9}  {'even' if n % 2 == 0 else 'odd'}")

    # ── n = 2: the three bipartite quantities side by side ───────────────────
    if c_woot is not None:
        c_gcg = C.get(2, float("nan"))
        print(f"\n  n = 2 cross-check (bipartite):")
        print(f"    convex roof (GCG)                     : {c_gcg:.6f}")
        print(f"    Wootters concurrence  C               : {c_woot:.6f}"
              f"   |diff| = {abs(c_gcg - c_woot):.2e}  <- same quantity")

    if args.tex:
        os.makedirs(os.path.dirname(args.tex) or ".", exist_ok=True)
        with open(args.tex, "w") as fh:
            par = ""
            if L is not None and U is not None and tt is not None:
                par = (f" for the $L={L}$ Hubbard ring at $U={U}$, $t={tt}$ "
                       f"($U/t={U/tt:.0f}$)")
            fh.write("\\begin{table}[h]\\centering\n\\caption{Ground-state "
                     f"($T=0$) cluster properties{par}, full Hamiltonian. "
                     "All values from the stored RDMs; $C_{\\mathrm{GME}}$ from "
                     "the GCG convex roof.}\n")
            fh.write("\\begin{tabular}{@{}ccccccc@{}}\\toprule\n")
            fh.write("$n$ & $2^n$ & Purity & Entropy & $\\langle S^2\\rangle$ & "
                     "Rank & $C_{\\mathrm{GME}}$ \\\\\n\\midrule\n")
            for n, dim, pur, ent, s2, rk, c in rows:
                cs = f"{c:.4f}" if np.isfinite(c) else "---"
                fh.write(f"{n} & {dim} & {pur:.4f} & {ent:.4f} & {s2:.4f} & "
                         f"{rk} & {cs} \\\\\n")
            fh.write("\\bottomrule\\end{tabular}\n")
            if c_woot is not None:
                cg = C.get(2, float("nan"))
                fh.write("\\\\[4pt]\n\\small\n")
                fh.write("\\begin{tabular}{@{}lc@{}}\\toprule\n")
                fh.write("\\multicolumn{2}{@{}l}{$n=2$ (bipartite) cross-check}"
                         " \\\\\n\\midrule\n")
                fh.write(f"Convex roof (GCG) & {cg:.6f} \\\\\n")
                fh.write(f"Wootters concurrence $C$ & {c_woot:.6f} \\\\\n")
                fh.write("\\bottomrule\\end{tabular}\n")
            fh.write("\\end{table}\n")
        print(f"\nLaTeX -> {args.tex}")


if __name__ == "__main__":
    main()
