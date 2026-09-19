# TMA Optical Attack Reader — clean optical branch

This branch is a clean rebuild of the score-reading layer. It contains no previous TMA reader, no Audiveris code, no HOMR code, and no legacy timing engine.

Purpose:

PDF/image -> independent low-level visual symbol masks -> notehead/stem/beam/dot/tie graph -> later exact rhythmic attack reconstruction.

The current stage intentionally stops before assigning musical onsets. It validates whether the page image can be converted into a trustworthy notation graph without inheriting timing, voice, or duration decisions from previous readers.

Low-level visual inference is provided only by Oemer's pretrained segmentation networks. Oemer's MusicXML builder, rhythm reconstruction, voice alignment, and semantic timing are not used.

Validation target in CI:
- exact two-page public-domain Buxtehude score used in the TMA regression work;
- both pages processed from PDF raster data;
- colored overlays and JSON notation graph generated;
- source-tree independence gate runs before analysis.

No production deployment is performed from this branch.

Validation branch initialized from a source-clean tree.


## Clean product pipeline

The validation branch runs a one-way pipeline: independent optical notation extraction → exact rational rhythmic-attack reconstruction → validated Tone-Metric Levels, wave, pivots, and trees. Horizontal engraving spacing is never converted proportionally into score time; unresolved timing blocks final TMA rather than invoking a legacy fallback.

<!-- targeted-regression-trigger: rational-full-grid-beam-band -->
