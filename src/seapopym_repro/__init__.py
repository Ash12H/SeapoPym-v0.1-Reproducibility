"""Shared code for the SeapoPym v0.1 reproducibility deposit.

Installed as a package so that every script folder imports the same paths, experiment setup and
figure style, for example `from seapopym_repro import paths`.
"""
from . import experiment, figstyle, paths  # noqa: F401
