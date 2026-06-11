"""seapopym_repro — shared code for the SeapoPym v0.1 reproducibility deposit.

A small installed package so the category script folders (scripts/data, scripts/experiments,
scripts/figures) import the same paths, twin-experiment core and figure style without fragile
sys.path tricks or Path(__file__).parents[N]. Imported as e.g. `from seapopym_repro import paths`.
"""
from . import experiment, figstyle, paths  # noqa: F401
