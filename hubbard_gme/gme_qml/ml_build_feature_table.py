#!/usr/bin/env python3
"""
ml_build_feature_table.py
-------------------------
Build the FINAL machine-learning feature table (Excel + CSV) from a dataset of
record .npz files.

One row per record with:
  * a clickable hyperlink to the record's .npz file (which stores the full RDM),
  * the metadata (NOT model inputs): L, t, U, T, sector, thermal, source,
  * the 33 finalized model inputs, all computed from rho alone:
        n
        pop_b0..pop_b6     per-Sz-block population        (trace of block)
        coh_b0..coh_b6     per-Sz-block coherence strength (off-diag norm)
        sgn_b0..sgn_b6     per-Sz-block signed coherence   (singlet - / triplet +)
        bpur_b0..bpur_b6   per-Sz-block purity
        coh_total, coh_maxabs, coh_min_real, coh_max_real
  * the label C_target (GCG convex roof).

Usage (repo root):
    python gme_qml/ml_build_feature_table.py                       # data_for_ML_v2
    python gme_qml/ml_build_feature_table.py --data gme_qml/data_for_ML_v2 \
        --out gme_qml/data_for_ML_v2/feature_table.xlsx
Re-run any time — it rescans the records directory (safe while ingestion runs).
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

MAXN = 10                      # supports n up to 10 -> 11 Sz blocks (b0..b10);
                               # feature count = 1 + 4*(MAXN+1) + 4 = 49.
                               # Future-proofed for n=8 (L=8 full clusters) and
                               # n=10; unused block slots are zero for small n.


def sz_block_features(rho: np.ndarray, n: int) -> dict:
    """The finalized 32 rho-derived inputs (block features + global coherence)."""
    dim = 2 ** n
    ndn = np.array([bin(i).count("1") for i in range(dim)])   # block id = #down
    out = {}
    for b in range(MAXN + 1):
        if b <= n:
            sel = np.where(ndn == b)[0]
            B = rho[np.ix_(sel, sel)]
            pop = float(np.real(np.trace(B)))
            off = B - np.diag(np.diag(B))
            out[f"pop_b{b}"] = pop
            out[f"coh_b{b}"] = float(np.linalg.norm(off))
            out[f"sgn_b{b}"] = float(np.real(off).sum())
            out[f"bpur_b{b}"] = (float(np.real(np.trace(B @ B)) / pop**2)
                                 if pop > 1e-12 else 0.0)
        else:
            out[f"pop_b{b}"] = out[f"coh_b{b}"] = 0.0
            out[f"sgn_b{b}"] = out[f"bpur_b{b}"] = 0.0
    off = rho - np.diag(np.diag(rho))
    out["coh_total"] = float(np.linalg.norm(off))
    out["coh_maxabs"] = float(np.abs(off).max())
    out["coh_min_real"] = float(np.real(off).min())
    out["coh_max_real"] = float(np.real(off).max())
    return out


def build_rows(records_dir: str) -> pd.DataFrame:
    rows = []
    files = sorted(glob.glob(os.path.join(records_dir, "*.npz")))
    for p in files:
        try:
            z = np.load(p, allow_pickle=True)
            cfg = json.loads(str(z["config_json"]))
            rho = z["rho_real"] + 1j * z["rho_imag"]
            n = int(cfg.get("n_body"))
        except Exception as e:
            print(f"  [skip] {os.path.basename(p)}: {e}")
            continue
        row = {
            # link + identity
            "npz_path": os.path.abspath(p),
            "config_hash": cfg.get("config_hash", ""),
            # metadata (NOT model inputs)
            "L": cfg.get("L"), "t": cfg.get("t"), "U": cfg.get("U"),
            "T": cfg.get("T", 0.0), "thermal": cfg.get("thermal", False),
            "sector": cfg.get("sector", ""), "source": "bank",
            # label
            "C_target": float(z["C_target"]) if "C_target" in z else np.nan,
            # model inputs
            "n": n,
        }
        row.update(sz_block_features(rho, n))
        rows.append(row)
    return pd.DataFrame(rows)


def write_excel(df: pd.DataFrame, out_xlsx: str) -> None:
    """Excel with a real clickable hyperlink to each RDM npz file."""
    from openpyxl import Workbook
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    feature_cols = (["n"]
                    + [f"{k}_b{b}" for k in ("pop", "coh", "sgn", "bpur")
                       for b in range(MAXN + 1)])
    feature_cols += ["coh_total", "coh_maxabs", "coh_min_real", "coh_max_real"]
    meta_cols = ["L", "t", "U", "T", "thermal", "sector", "source",
                 "config_hash"]
    cols = ["rdm_link"] + meta_cols + ["C_target"] + feature_cols

    wb = Workbook()
    ws = wb.active
    ws.title = "features"
    ws.append(cols)
    for c in ws[1]:
        c.font = Font(bold=True)
    for _, r in df.iterrows():
        ws.append([""] + [r.get(c) for c in meta_cols]
                  + [r.get("C_target")] + [r.get(c) for c in feature_cols])
        cell = ws.cell(row=ws.max_row, column=1)
        cell.value = os.path.basename(r["npz_path"])
        cell.hyperlink = "file:///" + r["npz_path"].replace(os.sep, "/")
        cell.font = Font(color="0563C1", underline="single")
    ws.freeze_panes = "B2"
    for j, c in enumerate(cols, start=1):
        ws.column_dimensions[get_column_letter(j)].width = max(11, len(c) + 2)
    wb.save(out_xlsx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="gme_qml/data_for_ML_v2",
                    help="dataset dir (records under <data>/ingested/records)")
    ap.add_argument("--out", default=None, help="output .xlsx path")
    args = ap.parse_args()

    records = os.path.join(args.data, "ingested", "records")
    if not os.path.isdir(records):
        records = os.path.join(args.data, "records")
    df = build_rows(records)
    if df.empty:
        raise SystemExit(f"no records found under {records}")

    out_xlsx = args.out or os.path.join(args.data, "feature_table.xlsx")
    write_excel(df, out_xlsx)
    out_csv = os.path.splitext(out_xlsx)[0] + ".csv"
    df.to_csv(out_csv, index=False)
    print(f"records      : {len(df)}")
    print(f"Excel        : {out_xlsx}")
    print(f"CSV (same)   : {out_csv}")
    n_feat = 1 + 4 * (MAXN + 1) + 4
    print(f"model inputs : n + {4*(MAXN+1)} block features (b0..b{MAXN}) "
          f"+ 4 global coherence = {n_feat}")


if __name__ == "__main__":
    main()
