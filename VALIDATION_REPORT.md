# Tone-Metric Analyzer v0.16.1 — reconstruction validation

## Release status

This is the dissertation-first v0.16.1 reconstruction candidate. The Levels engine is a substantive rewrite and does not use the v0.15.4 recursive refinement/fallback implementation.

The public-replacement gate is intentionally **not marked complete** until the exact BWV 661 PDF regression score is available again in the working session and the complete event-by-event grids of Dissertation Examples 4.7 and 4.9 are checked against machine-readable score events. The software does not silently declare conformance from self-generated tests.

## Dissertation checks implemented

- Binary sequence: `1, 2, 3, 5, 9, 17, 33, ...` (`y = 2x - 1`).
- Ternary sequence: `1, 2, 4, 10, 28, ...` (`y = 3x - 2`).
- Level 1 is structural/tactus-derived and does not depend on an attack.
- Quarter-note denomination is fully resolved before eighth-note denomination; eighth is fully resolved before sixteenth, etc.
- Chapter 4 boundary placement uses a snapshot of the preceding completed stage. The worked-figure invariant is one level above the lower boundary height, not one above the maximum stack.
- Parenthetical labels are represented internally as `structural_level_point=true`, `attack=false`, `parenthetical=true`.
- 12/8 is explicitly modeled as binary at the dotted-quarter level and ternary at the eighth-note level; finer ordinary subdivisions are binary.
- 4/4 has no generic triplet/ternary inference in the dissertation core.
- Tuplet notation is handled before the mathematical engine. Tuplet-only voice attacks are excluded from the strict core; ordinary simultaneous attacks in other voices remain.
- Meter changes restart Level 1.
- Pickup measures are right-aligned inside the full-measure framework before Levels are calculated.

## Worked-example regression coverage

- Examples 1.1–1.3: asserted exact Level-1/2/3 quarter-grid positions for mm. 1–3 of the WTC Prelude example.
- Example 1.4: asserted relative eighth-note layering and the Level-4 maximum.
- Example 1.5: asserted the sixteenth-note stage, Level-5 maximum, and the first `1+2+3` stack visible in the dissertation figure.
- Example 4.7: asserted all three Chapter-4 placement cases, including the worked-figure direction in which boundary heights 2 and 4 produce new Level 3.
- Example 4.9: asserted 12/8 stage order (binary dotted-quarter -> ternary eighth) and prohibition of spacing-based arity inference.
- Example 6.6 (Berg opening): asserted the structural `(1)` before the first sounding point and the first attack stack `1+2+3` in a right-aligned 3/4 pickup framework.

## Software-contract regression coverage

- Tied continuations do not create attacks.
- Grace notes do not create attacks.
- Rests do not create attacks.
- Simultaneous ordinary notes merge to one hit.
- Tuplet-only attacks are filtered before Levels; a simultaneous ordinary voice survives.
- Missing initial meter fails rather than assuming 4/4.
- Unsupported meter profiles fail rather than falling back to binary.
- PDF analysis has no silent canonical-OMR -> symbolic-MusicXML event fallback.
- Old `_refine_span`, explicit local tuplet-Level, and previous reachability entry points are absent from the Levels engine.
- Attack-label registration is exact canonical score-time -> originating attack column; the registrar has no nearest-event path.

## Automated validation

Run from the unzipped runtime directory:

```text
python -m unittest discover -s tests -v
python validate_release.py
```

The packaged snapshot was built only after both commands passed.

## v0.16.1 packaging/validator correction

v0.16.1 does not change the dissertation Levels engine from v0.16.0. It corrects two release-engineering problems discovered during Windows setup testing:

- the user-facing ZIP is now a single ready-to-run folder with `requirements.txt`, `validate_release.py`, and `app.py` immediately visible after extraction;
- `validate_release.py` audits only first-party runtime source (`app.py` and `tone_metric/*.py`), so a locally-created `.venv` and third-party `site-packages` cannot be mistaken for duplicate or legacy project code.
