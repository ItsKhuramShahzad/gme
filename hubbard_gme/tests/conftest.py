"""Pytest bootstrap for the test suite.

The package lives in ``src/hubbard_gme``, which is not on ``sys.path`` unless
the project has been installed (``pip install -e .``). Older test modules each
carried their own ``sys.path.insert(...)`` line; putting it here instead means
every test module — old and new — can simply ``import hubbard_gme`` and
``python -m pytest tests/`` works from a clean checkout with no PYTHONPATH.
"""
import os
import sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
