#!/usr/bin/env bash
# Convenience runner for the hubbard_gme test suite.
#
#   ./tests/run_tests.sh            # fast suite  (~3 min, 100 tests)  [default]
#   ./tests/run_tests.sh quick      # ~5 s, 19 tests
#   ./tests/run_tests.sh audit      # the 2026 code-review tests
#   ./tests/run_tests.sh all        # everything incl. slow integration tests
#   ./tests/run_tests.sh bench      # before/after transport benchmark (slow)
#
# Extra pytest flags are passed straight through, e.g.
#   ./tests/run_tests.sh quick -v
#
# tests/conftest.py puts src/ on sys.path, so no PYTHONPATH is required.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

MODE="${1:-fast}"
[ $# -gt 0 ] && shift

# Default to -q, but drop it if the caller asked for verbose output, otherwise
# -q and -v fight and you get file-level dots instead of one line per test.
QUIET="-q"
for a in "$@"; do
  case "$a" in -v|-vv|-vvv|--verbose) QUIET="" ;; esac
done

SLOW_IGNORES=(
  --ignore=tests/test_canonical_parameters.py
  --ignore=tests/test_convex_roof.py
  --ignore=tests/test_mixed_states.py
)

QUICK=(
  tests/test_gcg_transport.py
  tests/test_separable_boundary.py
  tests/test_meyerwallach.py
  tests/test_measure_param.py
)

AUDIT=(
  tests/test_gcg_transport.py       # #5 transport
  tests/test_separable_boundary.py  # #6 separable boundary
  tests/test_gradient.py            # gradient A-F (slowest of the group)
  tests/test_measure_param.py       # #1 measure propagation
  tests/test_meyerwallach.py        # MW value/gradient
)

case "$MODE" in
  fast)
    echo "==> fast suite (slow integration modules skipped)"
    python -m pytest tests/ $QUIET "${SLOW_IGNORES[@]}" "$@"
    ;;
  quick)
    echo "==> quick subset"
    python -m pytest "${QUICK[@]}" $QUIET "$@"
    ;;
  audit)
    echo "==> 2026 code-review tests"
    python -m pytest "${AUDIT[@]}" $QUIET "$@"
    ;;
  all)
    echo "==> FULL suite, including slow real-optimisation tests (can take hours)"
    python -m pytest tests/ $QUIET "$@"
    ;;
  bench)
    echo "==> before/after transport benchmark (not a pytest test)"
    python tests/test_gcg_transport_benchmark.py --which "${1:-real}"
    ;;
  *)
    echo "usage: $0 [fast|quick|audit|all|bench] [extra pytest flags]" >&2
    exit 2
    ;;
esac
