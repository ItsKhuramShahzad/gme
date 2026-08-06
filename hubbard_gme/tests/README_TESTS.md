# Running the test suite

All commands assume:

```bash
cd ~/PhDWork/hubbard_gme
conda activate qchem
```

`tests/conftest.py` puts `src/` on `sys.path`, so **no `PYTHONPATH` is needed**
and the package does not have to be installed. (`pip install -e .` also works
and additionally makes `import hubbard_gme` available outside pytest.)

---

## 1. Everyday run — fast suite (~3 min, 100 tests)

Skips the three slow real-optimisation integration modules.

```bash
python -m pytest tests/ -q \
  --ignore=tests/test_canonical_parameters.py \
  --ignore=tests/test_convex_roof.py \
  --ignore=tests/test_mixed_states.py
```

Or with the helper script:

```bash
./tests/run_tests.sh          # same thing
```

## 2. Quick check (~5 s, 19 tests)

```bash
python -m pytest tests/test_gcg_transport.py tests/test_separable_boundary.py \
                 tests/test_meyerwallach.py tests/test_measure_param.py -q
```

```bash
./tests/run_tests.sh quick
```

## 3. Everything, including the slow integration tests (hours)

```bash
python -m pytest tests/ -q
./tests/run_tests.sh all
```

---

## Audit-related tests (2026 code review)

| Command | Covers |
|---|---|
| `python -m pytest tests/test_gcg_transport.py -q` | #5 parallel transport: body-coordinate convention, tangency, parallel condition, generator self-transport, descent safeguard |
| `python -m pytest tests/test_separable_boundary.py -q` | #6 CGME derivative as C → 0: documents the 1/C scaling and asserts the final tangent gradient stays bounded |
| `python -m pytest tests/test_gradient.py -q` | analytic gradient (A–D), subdifferential membership at ties (E), end-to-end separable check (F) — ~3 min |
| `python -m pytest tests/test_measure_param.py -q` | #1 `measure` propagation, including the regression that `history[0]` differs between MW and CGME |
| `python -m pytest tests/test_meyerwallach.py -q` | Meyer–Wallach value/gradient + analytic values (GHZ = 1, W = 8/9, product = 0, Bell⊗0 = 2/3) |

```bash
./tests/run_tests.sh audit     # runs all of the above
```

## Before/after benchmark (not a pytest test)

Compares the current solver against a copy with **only** the transport line
patched. Slow (minutes to hours); results are recorded in the file's docstring.

```bash
python tests/test_gcg_transport_benchmark.py --which real
python tests/test_gcg_transport_benchmark.py --which synthetic
python tests/test_gcg_transport_benchmark.py --which both --restarts 15
```

---

## Watching progress — which test is running / passed

`run_tests.sh` passes any extra flags straight through to pytest, and switches
off its default `-q` when you ask for `-v`:

```bash
./tests/run_tests.sh fast -v          # one line per test, with PASSED/FAILED
./tests/run_tests.sh fast -v -s       # ALSO stream the solver's own output
                                      # (essential for slow tests — shows it is
                                      #  progressing rather than hung)
./tests/run_tests.sh all -v -s --durations=10
```

With `-v` you get:

```
tests/test_gcg_transport.py::test_gradient_is_body_coordinates PASSED  [  5%]
tests/test_gcg_transport.py::test_transport_preserves_tangency PASSED  [ 10%]
```

In quiet mode the characters mean: `.` pass, `F` fail, `E` error, `s` skipped.

Find the slow tests:

```bash
./tests/run_tests.sh fast --durations=10
```

If a run looks stuck, from a second terminal:

```bash
ps aux | grep pytest
```

## Useful pytest flags

```bash
-q                     # quiet
-v                     # one line per test
-x                     # stop at first failure
-k transport           # only tests whose name matches
--lf                   # re-run only last-failed
-s                     # show print() output
--durations=10         # 10 slowest tests
```

Example:

```bash
python -m pytest tests/ -v -k "transport or boundary"
python -m pytest tests/test_gradient.py::test_E -q -s
```

---

## Notes

* `pytest.ini` sets `filterwarnings = error::pytest.PytestReturnNotNoneWarning`,
  so a test that `return`s a bool instead of asserting is now an **error**
  (audit issue #2 — that pattern previously hid a genuine failure).
* The slow modules `test_canonical_parameters.py`, `test_convex_roof.py` and
  `test_mixed_states.py` run real convex-roof optimisations; run them
  deliberately, not in the everyday loop.
