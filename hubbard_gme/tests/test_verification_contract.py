from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]


def test_project_management_files_exist():
    for name in ["Plan.md", "Progress.md", "Verify.md"]:
        path = ROOT / name
        assert path.exists(), f"{name} is missing"
        assert path.stat().st_size > 1000, f"{name} is unexpectedly small"


def test_docs_contain_current_cache_keywords():
    text = "\n".join((ROOT / name).read_text(encoding="utf-8") for name in ["Plan.md", "Progress.md", "Verify.md"])
    required = [
        "per-method",
        "stiefel",
        "gcg",
        "stiefel_gcg",
        "RDM cache",
        "GME PKL",
        "n=N",
        "pure-state",
        "convex roof",
        "effective",
    ]
    missing = [word for word in required if word.lower() not in text.lower()]
    assert not missing, f"Missing documentation keywords: {missing}"


def test_verification_scripts_parse():
    for rel in ["scripts/verify_project.py", "scripts/inspect_gme_pkl.py"]:
        path = ROOT / rel
        assert path.exists(), f"{rel} missing"
        ast.parse(path.read_text(encoding="utf-8"))
