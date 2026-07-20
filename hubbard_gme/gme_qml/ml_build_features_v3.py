#!/usr/bin/env python3
"""
ml_build_features_v3.py — build the enriched V1/V2/V3 feature table (CSV + xlsx).

Implements the nested vectors from gme_ml_input_features.tex, with all the
agreed corrections (conditional block B~ = B_b/p_b, strict i<j upper triangle,
structural zeros excluded, s+/s- instead of signed-sum+L1, normalized Hamming
R_bd instead of H, block masks).

Per block slot (11 slots, b0..b10, design cap n<=10):
  V1 core (6):  m_b, p_b, q_b, C2_b, splus_b, sminus_b
  V2 +Hamming (5): R_b2, R_b4, R_b6, R_b8, R_b10      (normalized distribution)
  V3 +shape (2):  Cmax_b, ftau_b
Global (4): C2_glob, absmax, remin, remax     Size var: n
  V1 = 11*6 + 1 + 4 = 71
  V2 = 71 + 11*5     = 126
  V3 = 126 + 11*2    = 148

Usage (repo root):
  python gme_qml/ml_build_features_v3.py --version v3 --n 3 --L 6
  python gme_qml/ml_build_features_v3.py --version v3            # all n, all L
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd

MAXN = 10
TAU = 1e-6                         # documented absolute threshold on B~
HAM_D = [2, 4, 6, 8, 10]          # even distances only (fixed-Sz block)


def _blocks(n):
    dim = 2 ** n
    ndn = np.array([bin(i).count("1") for i in range(dim)])
    return dim, ndn


def block_features(rho, n, version):
    dim, ndn = _blocks(n)
    feats = {}
    for b in range(MAXN + 1):
        real = b <= n
        pre = f"b{b}_"
        if not real:
            feats[pre + "m"] = 0.0
            for k in ("p", "q", "C2", "splus", "sminus",
                      "Cmax", "ftau"):
                feats[pre + k] = 0.0
            for d in HAM_D:
                feats[pre + f"R{d}"] = 0.0
            continue

        sel = np.where(ndn == b)[0]
        B = rho[np.ix_(sel, sel)]
        p = float(np.real(np.trace(B)))
        feats[pre + "m"] = 1.0
        feats[pre + "p"] = p
        db = len(sel)

        if p > 1e-12:
            Bt = B / p
        else:
            Bt = B
        feats[pre + "q"] = float(np.real(np.trace(Bt @ Bt))) if p > 1e-12 else 0.0

        if db > 1:
            iu = np.triu_indices(db, 1)          # strict upper triangle i<j
            v = Bt[iu]
            av = np.abs(v)
            rv = np.real(v)
            feats[pre + "C2"] = float(np.sqrt(2.0 * np.sum(av ** 2)))
            feats[pre + "splus"] = float(np.clip(rv, 0, None).sum())
            feats[pre + "sminus"] = float(np.clip(-rv, 0, None).sum())
            feats[pre + "Cmax"] = float(av.max())
            feats[pre + "ftau"] = float((av > TAU).sum() / (db * (db - 1) / 2))
            # Hamming-resolved coherence (on B~), normalized distribution
            H = {d: 0.0 for d in HAM_D}
            for idx in range(len(iu[0])):
                i_, j_ = iu[0][idx], iu[1][idx]
                d = bin(int(sel[i_]) ^ int(sel[j_])).count("1")
                if d in H:
                    H[d] += float(av[idx] ** 2)
            tot = sum(H.values())
            for d in HAM_D:
                feats[pre + f"R{d}"] = float(H[d] / tot) if tot > 1e-15 else 0.0
        else:
            for k in ("C2", "splus", "sminus", "Cmax", "ftau"):
                feats[pre + k] = 0.0
            for d in HAM_D:
                feats[pre + f"R{d}"] = 0.0

    # global summaries over within-block strict-upper-triangle of unnormalised rho
    gvals = []
    for b in range(n + 1):
        sel = np.where(ndn == b)[0]
        if len(sel) > 1:
            Bb = rho[np.ix_(sel, sel)]
            iu = np.triu_indices(len(sel), 1)
            gvals.append(Bb[iu])
    if gvals:
        g = np.concatenate(gvals)
        feats["g_C2"] = float(np.sqrt(2.0 * np.sum(np.abs(g) ** 2)))
        feats["g_absmax"] = float(np.abs(g).max())
        feats["g_remin"] = float(np.real(g).min())
        feats["g_remax"] = float(np.real(g).max())
    else:
        feats.update(g_C2=0.0, g_absmax=0.0, g_remin=0.0, g_remax=0.0)

    feats["n"] = int(n)

    # trim columns to the requested version
    keep_core = ["m", "p", "q", "C2", "splus", "sminus"]
    keep_ham = [f"R{d}" for d in HAM_D]
    keep_shape = ["Cmax", "ftau"]
    per_block = keep_core[:]
    if version in ("v2", "v3"):
        per_block += keep_ham
    if version == "v3":
        per_block += keep_shape
    cols = []
    for b in range(MAXN + 1):
        cols += [f"b{b}_{k}" for k in per_block]
    cols += ["g_C2", "g_absmax", "g_remin", "g_remax", "n"]
    return {c: feats[c] for c in cols}, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v1", "v2", "v3"], default="v3")
    ap.add_argument("--n", type=int, default=None, help="filter cluster size")
    ap.add_argument("--L", type=int, default=None, help="filter ring size")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dirs", nargs="+", default=[
        "gme_qml/data_for_ML/ingested/records",
        "gme_qml/data_for_ML_v2/ingested/records"])
    args = ap.parse_args()

    rows = []
    feat_cols = None
    for d in args.dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.npz"))):
            try:
                z = np.load(p, allow_pickle=True)
                cfg = json.loads(str(z["config_json"]))
                n = int(cfg["n_body"]); L = int(cfg["L"])
                if args.n is not None and n != args.n:
                    continue
                if args.L is not None and L != args.L:
                    continue
                C = float(z["C_target"])
                if not np.isfinite(C):
                    continue
                rho = z["rho_real"] + 1j * z["rho_imag"]
            except Exception:
                continue
            feats, feat_cols = block_features(rho, n, args.version)
            meta = {"L": L, "t": cfg.get("t"), "U": cfg.get("U"),
                    "T": cfg.get("T", 0.0), "thermal": cfg.get("thermal", False),
                    "sector": cfg.get("sector", ""),
                    "C_target": C,
                    "npz": os.path.basename(p)}
            rows.append({**meta, **feats})

    if not rows:
        raise SystemExit("no records matched the filters")
    df = pd.DataFrame(rows)
    meta_cols = ["npz", "L", "t", "U", "T", "thermal", "sector", "C_target"]
    df = df[meta_cols + feat_cols]

    tag = args.version + (f"_n{args.n}" if args.n else "") + \
        (f"_L{args.L}" if args.L else "")
    out = args.out or f"gme_qml/data_for_ML_v2/features_{tag}.csv"
    df.to_csv(out, index=False)
    n_feat = len(feat_cols)
    print(f"records         : {len(df)}")
    print(f"feature columns : {n_feat}  (version {args.version})")
    print(f"+ metadata cols : {len(meta_cols)}  (npz link, L,t,U,T,sector,label)")
    print(f"total columns   : {df.shape[1]}")
    print(f"CSV             : {out}")


if __name__ == "__main__":
    main()
