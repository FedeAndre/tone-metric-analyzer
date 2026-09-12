# Tone-Metric Analyzer v0.16.0

This bundle contains the dissertation-first reconstruction of the Tone-Metric Levels engine.

## What changed in v0.16.0

- The Levels engine is a substantive rewrite based on the dissertation procedure rather than on v0.15.4 behavior.
- Binary sequence is fixed to `1, 2, 3, 5, 9, 17, 33, ...`; ternary is `1, 2, 4, 10, 28, ...`.
- Level 1 is a structural tactus layer independent of attacks.
- Each rhythmic denomination is completed before the next finer denomination.
- Chapter-4 placement uses the preceding-stage boundary snapshot. The worked-figure rule is one above the lower boundary height; same-stage labels cannot contaminate neighboring spans.
- Structural no-attack Level points are first-class data and render parenthetically.
- Meter arity is explicit. Unsupported meters fail instead of falling back to binary or assuming 4/4.
- 12/8 is binary at dotted-quarter level and ternary at eighth level; 4/4 has no generic triplet rule in the dissertation core.
- Tuplet interpretation is separated from the Levels mathematics; tuplet-voice attacks are excluded from the strict core while simultaneous ordinary voices remain.
- PDF analysis no longer silently substitutes symbolic MusicXML events when canonical OMR score-time recovery fails.
- Old v0.15.4 recursive refinement/tuplet entry points are removed from the active source.

See `VALIDATION_REPORT.md` inside `tone_metric_runtime_source.zip` for the exact conformance tests and current release gate.

## Windows PowerShell start commands

Unzip `tone_metric_runtime_source.zip`, open PowerShell in that unzipped runtime folder, and run:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000` in the browser.

For PDF analysis, Audiveris must be installed and either available on `PATH` or identified explicitly before starting the server, for example:

```powershell
$env:AUDIVERIS_CMD = "C:\path\to\Audiveris.exe"
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Use the actual Audiveris executable path on your machine. MusicXML/MXL validation does not require Audiveris.

## Local validation

From the unzipped runtime folder:

```powershell
.\.venv\Scripts\python.exe validate_release.py
```

## Railway

Railway continues to use `Dockerfile`. The image installs Audiveris, unpacks `tone_metric_runtime_source.zip`, installs Python requirements, and starts Uvicorn on `${PORT:-8080}`.
