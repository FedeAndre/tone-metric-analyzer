# Tone-Metric Analyzer

Browser-based Tone-Metric Analyzer, version 0.15.4 (dissertation-conformance audited).

This repository is intentionally minimal. Railway builds the app from `Dockerfile`, which unpacks `tone_metric_runtime_source.zip` into the container and launches the FastAPI application.

The runtime ZIP contains the audited v0.15.4 analyzer and current score renderer. The dissertation-conformance revision restores region-based pivots, removes automatic ternary inference from equal event spacing, keeps explicit tuplet support isolated, and anchors attack labels to notehead geometry for display without changing musical timing.
