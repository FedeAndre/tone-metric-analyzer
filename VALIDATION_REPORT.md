# Tone-Metric Analyzer v0.16.3 validation report

## Scope of this release

v0.16.3 changes the Audiveris/PDF driver and diagnostic handling only. The dissertation-first Levels engine introduced in v0.16.0 is not changed by this release.

The reason for the rewrite is a real Audiveris failure reported on the Berg/IMSLP02556 score. Audiveris reached its `PartwiseBuilder`/`ScoreExporter` path and failed while sorting wedge events with null `timeOffset` values. That is an export-stage failure and must not be allowed to redefine or silently replace the tone-metric event model.

## Required PDF pipeline

The v0.16.3 core PDF command is explicitly:

`-batch -step PAGE -save -output <dir> -- <score.pdf>`

The required core pass does **not** request `-transcribe` and does **not** request `-export`.

The saved native `.omr` is then checked as a ZIP project (`book.xml` plus at least one sheet XML) and passed to the canonical OMR score-time parser. The downstream parser must still recover a valid meter framework and attacks. If it cannot, analysis stops; no MusicXML or alternate event source is substituted.

If Audiveris returns a non-zero code *after* a structurally valid native `.omr` has already been persisted, v0.16.3 is permitted to continue with that native project. The non-zero return code and salvage state are reported in `audiveris_core_meta`. A missing or corrupt `.omr` remains fatal.

The optional physical-annotation pass now uses the saved `.omr` as its input. It does not retranscribe the PDF and does not request MusicXML export. Annotation failure is display-only and cannot alter score time or Levels.

## Automated validation performed

Fresh source validation completed successfully:

- 34/34 unit/conformance tests passed.
- Chapter 1 worked-example constraints passed.
- Chapter 4 three boundary-placement cases passed.
- Corrected binary and ternary sequences passed.
- 12/8 binary-then-ternary stage ordering passed.
- Berg opening fixture retains `(1)` before the first attack and Levels `1+2+3` at that first attack.
- Meter-change reset, pickup, tie, grace-note, rest, simultaneous-attack, and tuplet preprocessing contracts passed.
- Required Audiveris command test asserts `-step PAGE` and `-save`, and rejects `-transcribe` and `-export`.
- Non-zero Audiveris return + valid native `.omr` salvage behavior passed.
- Non-zero return + corrupt `.omr` fatal behavior passed.
- Annotation-from-saved-OMR contract passed.
- Duplicate top-level definitions and known legacy Levels paths were audited and rejected if present.
- `app.py` imports one current Levels engine entry point.
- Python compilation completed successfully.
- Uvicorn started successfully in the validation environment.
- `GET /` returned HTTP 200.
- `GET /api/status` returned HTTP 200 with `version=0.16.3` and `omr_driver=step-page-save`.

## End-to-end limitation

Audiveris itself is not installed in the build/validation environment used here, so the exact Berg/IMSLP02556 PDF could not be re-run through Audiveris locally. Therefore this report does **not** claim an end-to-end Berg-PDF pass. It validates the driver contract, persisted-OMR handling, canonical parsing contracts, application startup, and dissertation engine separately.

The previously used exact BWV 661 regression file (`bwv661-a4.pdf`) is also not available in the active runtime/library, so no new end-to-end BWV 661 PDF regression is claimed for v0.16.3.

## Legacy/override audit

The current first-party runtime source was checked for the known obsolete or conflicting implementations. The active source contains no old `_refine_span` engine, no old generic ternary-from-hit path, no old PDF-to-MusicXML core function, no old MusicXML/OMR reconciliation path, and no duplicate first-party top-level implementations. The current PDF core route is `pdf_to_omr -> build_measure_framework_from_omr -> build_hits_from_canonical_score -> preprocessing -> analyze`.
