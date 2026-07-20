#!/usr/bin/env python3
"""
train_baselines.py
------------------
STAGE 4 of the gme_qml pipeline: classical ML baselines for

    Task A  DETECTION  — classify GME present (C_GME > eps) vs absent
    Task B  ESTIMATION — regress the value of C_GME

on Hubbard n-body RDM records produced by ml_generate_ml_dataset.py and/or
ml_ingest_pkls.py.

Design decisions (see README.md):
  * Fixed-dimension "physics" features (usable across n and for QML later):
    L, n_body, t, U, U/t, purity, entropy, tr rho^2, tr rho^3, rank,
    T + thermal flag (0/False for native records), top-K RDM eigenvalues.
    Optionally (--with-bounds) the cheap analytic bounds LB/UB as features.
  * Label = best available convex-roof value: C_gcg when present (ingested
    records), else C_target (native records; Stiefel-labeled v1 dataset).
  * GROUP SPLIT by Hamiltonian identity (L, mode, t, U, Ne): all records of
    the same physical system stay in one fold — random row splits leak.
  * Models: logistic/ridge (linear), random forest, hist gradient boosting.
  * Reports overall metrics + the "hard band" (|C - eps| < band) + per-n.

Outputs:  models/baseline_metrics_<ts>.json, models/<task>_<model>_<ts>.joblib

Usage:
    python gme_qml/train_baselines.py                 # native + ingested
    python gme_qml/train_baselines.py --native-only
    python gme_qml/train_baselines.py --eps 0.01 --test-frac 0.25 --with-bounds
    python gme_qml/train_baselines.py --holdout-L 6   # extrapolation: train L<6, test L=6
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_DATA_DIR = _THIS_DIR / "data_for_ML"
_MODELS_DIR = _THIS_DIR / "models"

N_EIG_FEATURES = 16   # top-K eigenvalues (desc, padded with 0) as features


# ────────────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────────────

def _parse_eigvals(s) -> np.ndarray:
    """Parse the rdm_eigvals manifest column (list repr) → top-K desc, padded."""
    out = np.zeros(N_EIG_FEATURES)
    try:
        vals = np.asarray(json.loads(s), dtype=float)
        vals = np.sort(np.clip(vals, 0.0, None))[::-1][:N_EIG_FEATURES]
        out[: len(vals)] = vals
    except Exception:
        pass
    return out


def load_dataset(native_only: bool = False,
                 data_dir: Path = _DATA_DIR) -> pd.DataFrame:
    frames = []
    native = data_dir / "manifest.csv"
    if native.exists():
        df = pd.read_csv(native)
        df["record_source"] = "native"
        frames.append(df)
    ingested = data_dir / "ingested" / "manifest.csv"
    if not native_only and ingested.exists():
        df = pd.read_csv(ingested)
        df["record_source"] = "ingested"
        frames.append(df)
    if not frames:
        raise SystemExit(f"No manifest found under {data_dir}")
    df = pd.concat(frames, ignore_index=True, sort=False)

    def _num_col(name: str) -> pd.Series:
        """Numeric column as a Series; all-NaN Series when column is absent."""
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
        return pd.Series(np.nan, index=df.index)

    # ── label: prefer GCG over Stiefel ────────────────────────────────────
    c_gcg = _num_col("C_gcg")
    c_tgt = _num_col("C_target")
    df["label_C"] = np.where(np.isfinite(c_gcg), c_gcg, c_tgt)
    df["label_from"] = np.where(np.isfinite(c_gcg), "gcg", "stiefel/target")
    df = df[np.isfinite(df["label_C"])].copy()
    df["label_C"] = df["label_C"].clip(lower=0.0)

    # ── thermal columns exist only in ingested records ────────────────────
    df["T_feat"] = pd.to_numeric(df.get("T", pd.Series(np.nan, index=df.index)),
                                 errors="coerce").fillna(0.0)
    if "thermal" in df.columns:
        df["thermal_feat"] = (
            df["thermal"].astype(str).str.lower().eq("true").astype(float)
        )
    else:
        df["thermal_feat"] = 0.0

    # ── group id = Hamiltonian identity (leakage control) ─────────────────
    df["group_id"] = (
        df["L"].astype(str) + "_" + df["mode"].astype(str) + "_"
        + df["t"].round(6).astype(str) + "_" + df["U"].round(6).astype(str)
        + "_" + df["n_electrons"].astype(str)
    )
    return df


def build_features(df: pd.DataFrame, with_bounds: bool = False):
    scalar_cols = [
        "L", "n_body", "t", "U", "U_over_t",
        "purity", "entropy", "tr_rho2", "tr_rho3", "rank",
        "T_feat", "thermal_feat",
    ]
    X_scalar = np.column_stack([
        pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy()
        for c in scalar_cols
    ])
    X_eig = np.vstack([_parse_eigvals(s) for s in df["rdm_eigvals"]])
    names = scalar_cols + [f"eig{i}" for i in range(N_EIG_FEATURES)]
    X = np.hstack([X_scalar, X_eig])
    if with_bounds:
        for c in ("lower_bound_super", "C_UB"):
            X = np.hstack([
                X,
                pd.to_numeric(df[c], errors="coerce").fillna(0.0)
                  .to_numpy().reshape(-1, 1),
            ])
            names.append(c)
    return X, names


# ────────────────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────────────────

def cls_metrics(y_true, y_pred, y_prob=None) -> dict:
    from sklearn.metrics import (accuracy_score, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n": int(len(y_true)),
        "pos_frac": float(np.mean(y_true)),
    }
    if y_prob is not None and 0 < np.mean(y_true) < 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return out


def reg_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import mean_absolute_error, r2_score
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--eps", type=float, default=1e-2,
                    help="detection threshold: entangled iff C > eps (default 0.01)")
    ap.add_argument("--band", type=float, default=0.05,
                    help="hard-band half-width around eps for boundary metrics")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--native-only", action="store_true")
    ap.add_argument("--data-dir", type=str, default=str(_DATA_DIR),
                    help="dataset directory (default: <gme_qml>/data_for_ML); "
                         "point this at the v2 GCG-labeled dataset")
    ap.add_argument("--with-bounds", action="store_true",
                    help="add cheap analytic LB/UB as features")
    ap.add_argument("--holdout-L", type=int, default=None,
                    help="extrapolation split: test = this L, train = the rest")
    ap.add_argument("--save-models", action="store_true")
    args = ap.parse_args()

    try:
        import sklearn  # noqa: F401
    except ImportError:
        raise SystemExit("scikit-learn is required:  pip install scikit-learn")

    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor,
                                  RandomForestClassifier, RandomForestRegressor)
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    df = load_dataset(native_only=args.native_only,
                      data_dir=Path(args.data_dir))
    X, feat_names = build_features(df, with_bounds=args.with_bounds)
    y_reg = df["label_C"].to_numpy()
    y_cls = (y_reg > args.eps).astype(int)
    groups = df["group_id"].to_numpy()

    print(f"Dataset: {len(df)} records "
          f"({(df['record_source'] == 'native').sum()} native, "
          f"{(df['record_source'] == 'ingested').sum()} ingested)  "
          f"features={X.shape[1]}")
    print(f"Labels : {(df['label_from'] == 'gcg').sum()} GCG, "
          f"{(df['label_from'] != 'gcg').sum()} Stiefel/target   "
          f"entangled frac (eps={args.eps}): {y_cls.mean():.3f}")

    # ── split ──────────────────────────────────────────────────────────────
    if args.holdout_L is not None:
        te = (df["L"].to_numpy() == args.holdout_L)
        tr = ~te
        split_desc = f"extrapolation: train L!={args.holdout_L} → test L={args.holdout_L}"
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_frac,
                                random_state=args.seed)
        tr_idx, te_idx = next(gss.split(X, y_cls, groups=groups))
        tr = np.zeros(len(df), bool); tr[tr_idx] = True
        te = np.zeros(len(df), bool); te[te_idx] = True
        split_desc = (f"group split by Hamiltonian "
                      f"({df.loc[tr, 'group_id'].nunique()} train / "
                      f"{df.loc[te, 'group_id'].nunique()} test groups)")
    print(f"Split  : {split_desc}  → {tr.sum()} train / {te.sum()} test rows\n")
    if tr.sum() == 0 or te.sum() == 0:
        raise SystemExit("Empty train or test split — check --holdout-L value.")

    hard = np.abs(y_reg - args.eps) < args.band     # boundary states
    results = {"config": vars(args), "split": split_desc,
               "n_train": int(tr.sum()), "n_test": int(te.sum()),
               "feature_names": feat_names, "detection": {}, "estimation": {}}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _MODELS_DIR.mkdir(exist_ok=True)

    # ── Task A: detection ──────────────────────────────────────────────────
    classifiers = {
        "logreg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000)),
        "random_forest": RandomForestClassifier(
            n_estimators=400, random_state=args.seed, n_jobs=-1),
        "hist_gbm": HistGradientBoostingClassifier(random_state=args.seed),
    }
    print("── Task A: DETECTION (GME present?) " + "─" * 30)
    for name, model in classifiers.items():
        if len(np.unique(y_cls[tr])) < 2:
            print("  [skip] training split has a single class"); break
        model.fit(X[tr], y_cls[tr])
        prob = (model.predict_proba(X[te])[:, 1]
                if hasattr(model, "predict_proba") else None)
        pred = model.predict(X[te])
        m = cls_metrics(y_cls[te], pred, prob)
        hb = te & hard
        if hb.sum() > 5:
            m["hard_band"] = cls_metrics(y_cls[hb], model.predict(X[hb]))
        results["detection"][name] = m
        auc = f"AUC={m.get('roc_auc', float('nan')):.4f}" if "roc_auc" in m else ""
        hb_txt = (f"  hard-band acc={m['hard_band']['accuracy']:.3f} "
                  f"(n={m['hard_band']['n']})" if "hard_band" in m else "")
        print(f"  {name:14s} acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  {auc}{hb_txt}")
        if args.save_models:
            import joblib
            joblib.dump(model, _MODELS_DIR / f"detect_{name}_{ts}.joblib")

    # ── Task B: estimation ─────────────────────────────────────────────────
    regressors = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=400, random_state=args.seed, n_jobs=-1),
        "hist_gbm": HistGradientBoostingRegressor(random_state=args.seed),
    }
    print("\n── Task B: ESTIMATION (value of C_GME) " + "─" * 27)
    # physics baseline: the cheap analytic supervisor lower bound as predictor
    lb = pd.to_numeric(df["lower_bound_super"], errors="coerce").fillna(0.0).to_numpy()
    m0 = reg_metrics(y_reg[te], lb[te])
    results["estimation"]["baseline_supervisor_LB"] = m0
    print(f"  {'supervisor_LB':14s} MAE={m0['mae']:.4f}  RMSE={m0['rmse']:.4f}  "
          f"R²={m0['r2']:.4f}   (physics baseline to beat)")
    for name, model in regressors.items():
        model.fit(X[tr], y_reg[tr])
        pred = np.clip(model.predict(X[te]), 0.0, None)
        m = reg_metrics(y_reg[te], pred)
        m["per_n"] = {
            int(n): reg_metrics(y_reg[te & (df['n_body'] == n).to_numpy()],
                                np.clip(model.predict(X[te & (df['n_body'] == n).to_numpy()]), 0, None))
            for n in sorted(df.loc[te, "n_body"].unique())
            if (te & (df['n_body'] == n).to_numpy()).sum() > 5
        }
        results["estimation"][name] = m
        print(f"  {name:14s} MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R²={m['r2']:.4f}")
        if args.save_models:
            import joblib
            joblib.dump(model, _MODELS_DIR / f"estimate_{name}_{ts}.joblib")

    out = _MODELS_DIR / f"baseline_metrics_{ts}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nMetrics → {out}")
    if args.save_models:
        print(f"Models  → {_MODELS_DIR}/*_{ts}.joblib")


if __name__ == "__main__":
    main()
