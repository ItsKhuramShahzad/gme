#!/usr/bin/env python3
"""
scripts/verify_project.py

Verification helper for the Hubbard GME project.

Modes:
  static       - required paths + Python syntax
  deps         - dependency import check
  imports      - import hubbard_gme modules
  pytest-light - run lightweight tests only
  pytest-all   - run all tests
  artifacts    - check output artifacts
  cache-rdm    - inspect RDM PKL cache
  cache-gme    - inspect GME PKL cache
  smoke        - run a small CLI smoke test
  all          - static + deps + imports + lightweight tests + cache/artifact checks

This script is intentionally conservative. It can run light checks without
requiring heavy scientific computations.
"""

from __future__ import annotations

import argparse
import importlib
import os
import pickle
import py_compile
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


REQUIRED_PATHS = [
    "main.py",
    "src/hubbard_gme",
    "src/hubbard_gme/pipeline.py",
    "src/hubbard_gme/convex_roof.py",
    "src/hubbard_gme/rdm.py",
    "src/hubbard_gme/visualization.py",
    "Plan.md",
    "Progress.md",
    "Verify.md",
]

CORE_DEPS = [
    "numpy",
    "scipy",
    "matplotlib",
    "tqdm",
    "openfermion",
    "pytest",
]

OPTIONAL_DEPS = [
    "cvxpy",
    "pyscf",
    "openfermionpyscf",
]

PROJECT_IMPORTS = [
    "hubbard_gme.pipeline",
    "hubbard_gme.convex_roof",
    "hubbard_gme.rdm",
    "hubbard_gme.visualization",
    "hubbard_gme.gme",
    "hubbard_gme.metrics",
]


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def iter_python_files() -> Iterable[Path]:
    for base in ["main.py", "src", "scripts", "tests"]:
        p = PROJECT_ROOT / base
        if p.is_file() and p.suffix == ".py":
            yield p
        elif p.is_dir():
            yield from p.rglob("*.py")


def check_required_paths() -> int:
    errors = 0
    for rel in REQUIRED_PATHS:
        p = PROJECT_ROOT / rel
        if p.exists():
            ok(f"Required path exists: {rel}")
        else:
            fail(f"Missing required path: {rel}")
            errors += 1
    return errors


def check_syntax() -> int:
    errors = 0
    for py in iter_python_files():
        try:
            py_compile.compile(str(py), doraise=True)
        except Exception as exc:
            fail(f"Syntax error in {py.relative_to(PROJECT_ROOT)}: {exc}")
            errors += 1
    if errors == 0:
        ok("Python syntax check passed")
    return errors


def mode_static(_: argparse.Namespace) -> int:
    errors = check_required_paths()
    errors += check_syntax()
    return 0 if errors == 0 else 1


def check_deps() -> int:
    errors = 0
    for name in CORE_DEPS:
        try:
            importlib.import_module(name)
            ok(f"Dependency import: {name}")
        except Exception as exc:
            fail(f"Missing/failed core dependency: {name} ({exc})")
            errors += 1

    for name in OPTIONAL_DEPS:
        try:
            importlib.import_module(name)
            ok(f"Optional dependency import: {name}")
        except Exception as exc:
            warn(f"Optional dependency missing/failed: {name} ({exc})")
    return errors


def mode_deps(_: argparse.Namespace) -> int:
    return 0 if check_deps() == 0 else 2


def add_src_to_path() -> None:
    src = PROJECT_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def mode_imports(_: argparse.Namespace) -> int:
    add_src_to_path()
    errors = 0
    for name in PROJECT_IMPORTS:
        try:
            importlib.import_module(name)
            ok(f"Project import: {name}")
        except Exception as exc:
            fail(f"Project import failed: {name} ({exc})")
            errors += 1
    return 0 if errors == 0 else 3


def run_cmd(cmd: list[str], cwd: Path | None = None) -> int:
    print("+ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd or PROJECT_ROOT))
    return proc.returncode


def mode_pytest_light(_: argparse.Namespace) -> int:
    test_file = PROJECT_ROOT / "tests" / "test_verification_contract.py"
    if test_file.exists():
        return run_cmd([sys.executable, "-m", "pytest", str(test_file), "-q"])
    warn("tests/test_verification_contract.py not found; falling back to project-structure tests")
    return run_cmd([sys.executable, "-m", "pytest", "tests/test_project_structure.py", "-q"])


def mode_pytest_all(_: argparse.Namespace) -> int:
    return run_cmd([sys.executable, "-m", "pytest", "tests", "-v"])


def read_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def check_rdm_pkl(path: Path) -> list[str]:
    problems: list[str] = []
    try:
        data = read_pickle(path)
    except Exception as exc:
        return [f"cannot read pickle: {exc}"]

    for key in ["meta", "rho"]:
        if key not in data:
            problems.append(f"missing key {key}")

    meta = data.get("meta", {})
    metrics = data.get("metrics", {})
    for key in ["trace", "purity", "entropy"]:
        if key not in meta and key not in metrics:
            problems.append(f"missing metric {key}")

    if "rank_numerical" not in meta and "rank_numerical" not in metrics:
        problems.append("missing rank_numerical")

    rho = data.get("rho")
    if rho is not None:
        try:
            shape = tuple(rho.shape)
            if len(shape) != 2 or shape[0] != shape[1]:
                problems.append(f"rho is not square: {shape}")
        except Exception:
            problems.append("rho has no shape")
    return problems


def mode_cache_rdm(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_root or PROJECT_ROOT / "results" / "rdm_pkl_results" / "rdm_cache")
    if not cache_dir.exists():
        fail(f"RDM cache directory not found: {cache_dir}")
        return 4

    files = sorted(cache_dir.glob("*.pkl"))
    if not files:
        fail(f"No RDM PKLs found in {cache_dir}")
        return 4

    errors = 0
    for path in files[: args.max_files]:
        problems = check_rdm_pkl(path)
        if problems:
            fail(f"{path.name}: " + "; ".join(problems))
            errors += 1
        else:
            ok(f"RDM PKL OK: {path.name}")

    print(f"Checked {min(len(files), args.max_files)} of {len(files)} RDM PKLs")
    return 0 if errors == 0 else 4


def nested_get(d: dict, *keys: str, default=None):
    cur = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def check_gme_pkl(path: Path) -> list[str]:
    problems: list[str] = []
    try:
        data = read_pickle(path)
    except Exception as exc:
        return [f"cannot read pickle: {exc}"]

    for key in ["meta", "result_full"]:
        if key not in data:
            problems.append(f"missing key {key}")

    result = data.get("result_full", {})
    if not isinstance(result, dict):
        problems.append("result_full is not a dict")
        return problems

    for key in ["stiefel", "gcg", "stiefel_gcg"]:
        if key not in result:
            problems.append(f"missing solver result {key}")

    for key in ["C_stiefel", "C_gcg", "C_stiefel_gcg"]:
        if key not in result:
            problems.append(f"missing scalar {key}")

    if "solver_runtime" not in result:
        problems.append("missing solver_runtime")
    if "solver_success" not in result:
        problems.append("missing solver_success")
    if "solver_cache_status" not in result:
        problems.append("missing solver_cache_status")
    if "effective_settings" not in result and "solver_effective_settings" not in result:
        problems.append("missing effective settings")

    if "integrity" not in data:
        problems.append("missing top-level integrity")
    else:
        integrity = data["integrity"]
        for key in ["rho_hash_sha256", "trace", "purity", "rank", "eigvals_nonzero"]:
            if key not in integrity:
                problems.append(f"integrity missing {key}")

    decomps = data.get("decompositions")
    if decomps is None:
        problems.append("missing decompositions")
    elif isinstance(decomps, dict):
        for solver, entries in decomps.items():
            if entries:
                first = entries[0]
                if "decomp" not in first:
                    problems.append(f"{solver} decomposition entry missing decomp")
    return problems


def mode_cache_gme(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_root or PROJECT_ROOT / "results" / "rdm_pkl_results" / "gme_runs" / "job_cache")
    if not cache_dir.exists():
        fail(f"GME job cache directory not found: {cache_dir}")
        return 5

    files = sorted(cache_dir.glob("*.pkl"))
    if not files:
        fail(f"No GME job PKLs found in {cache_dir}")
        return 5

    errors = 0
    for path in files[: args.max_files]:
        problems = check_gme_pkl(path)
        if problems:
            fail(f"{path.name}: " + "; ".join(problems))
            errors += 1
        else:
            ok(f"GME PKL OK: {path.name}")

    print(f"Checked {min(len(files), args.max_files)} of {len(files)} GME PKLs")
    return 0 if errors == 0 else 5


def mode_artifacts(args: argparse.Namespace) -> int:
    artifact_dir = Path(args.artifact_dir)
    if not artifact_dir.is_absolute():
        artifact_dir = PROJECT_ROOT / artifact_dir

    if not artifact_dir.exists():
        fail(f"Artifact directory not found: {artifact_dir}")
        return 6

    patterns = ["*.pdf", "*.pkl", "*.npz", "*.csv", "*.json"]
    found: list[Path] = []
    for pat in patterns:
        found.extend(artifact_dir.rglob(pat))

    if not found:
        fail(f"No artifacts found in {artifact_dir}")
        return 6

    errors = 0
    for path in sorted(found)[: args.max_files]:
        if path.stat().st_size <= 0:
            fail(f"Empty artifact: {path.relative_to(PROJECT_ROOT)}")
            errors += 1
        else:
            ok(f"Artifact: {path.relative_to(PROJECT_ROOT)} ({path.stat().st_size} bytes)")

    print(f"Found {len(found)} artifacts")
    return 0 if errors == 0 else 6


def mode_smoke(args: argparse.Namespace) -> int:
    if args.smoke_size == "tiny":
        cmd = [
            sys.executable, "main.py",
            "--L", "4", "--U", "4.0",
            "--min-n", "3", "--max-n", "3",
            "--no-pdf",
            "--workers", "1",
            "--n-restarts", "1",
            "--k-max-extra", "0",
            "--max-iter", "20",
        ]
    else:
        cmd = [
            sys.executable, "main.py",
            "--L", "4", "--U", "4.0",
            "--min-n", "3", "--max-n", "4",
            "--no-pdf",
            "--workers", "1",
            "--n-restarts", "1",
            "--k-max-extra", "0",
            "--max-iter", "50",
        ]

    if args.use_pkl_cache:
        cmd.append("--use-pkl-cache")
        cmd.extend(["--cache-version", args.cache_version])

    return run_cmd(cmd)


def mode_all(args: argparse.Namespace) -> int:
    modes = [mode_static, mode_deps, mode_imports, mode_pytest_light]
    for fn in modes:
        rc = fn(args)
        if rc != 0:
            return rc

    # Optional checks: warn instead of fail if caches/artifacts do not exist yet.
    for name, fn in [("cache-rdm", mode_cache_rdm), ("cache-gme", mode_cache_gme), ("artifacts", mode_artifacts)]:
        rc = fn(args)
        if rc != 0:
            warn(f"Optional check did not pass yet: {name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Verify Hubbard GME project")
    p.add_argument("--mode", required=True,
                   choices=[
                       "static", "deps", "imports",
                       "pytest-light", "pytest-all",
                       "artifacts", "cache-rdm", "cache-gme",
                       "smoke", "all",
                   ])
    p.add_argument("--artifact-dir", default="outputs")
    p.add_argument("--cache-root", default=None,
                   help="Override cache directory for cache-rdm/cache-gme")
    p.add_argument("--max-files", type=int, default=20)
    p.add_argument("--smoke-size", choices=["tiny", "small"], default="tiny")
    p.add_argument("--use-pkl-cache", action="store_true")
    p.add_argument("--cache-version", default="v1")
    return p


def main() -> int:
    args = build_parser().parse_args()

    dispatch = {
        "static": mode_static,
        "deps": mode_deps,
        "imports": mode_imports,
        "pytest-light": mode_pytest_light,
        "pytest-all": mode_pytest_all,
        "artifacts": mode_artifacts,
        "cache-rdm": mode_cache_rdm,
        "cache-gme": mode_cache_gme,
        "smoke": mode_smoke,
        "all": mode_all,
    }
    return dispatch[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
