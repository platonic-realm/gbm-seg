"""Ablation orchestrator for sequential-pinning ablation studies.

Reads a YAML spec describing a study (a list of "cells" + a list of folds),
creates a per-cell experiment directory by deep-merging cell-specific
config overrides into the base experiment's configs.yaml, and emits the
exact training commands to submit.

See ``spec.py`` for the schema and ``runner.py`` for the materialisation
logic. Entry point: ``gbm.py ablate <spec.yaml>``.
"""
