# Tone-Metric Analyzer

Browser-based Tone-Metric Analyzer, version 0.15.3.

This repository is intentionally minimal. Railway builds the app from `Dockerfile`, which unpacks `tone_metric_runtime_source.zip` into the container and launches the FastAPI application.

The runtime ZIP contains the v0.15.3 triplet-aware analyzer and the current score renderer with analytical graphs above each printed system.
