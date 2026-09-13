# Tone-Metric Analyzer v0.16.4 — development

This `development` branch is an isolated deployment carrier for the dissertation-first v0.16.4 runtime. The stable `main` branch is not modified.

## Why this branch contains transport chunks

To prevent an older Python source tree, cached bytecode, or an obsolete runtime ZIP from overriding the current implementation, the branch contains **no directly executable Tone-Metric Python source**. Instead, it carries one authoritative v0.16.4 runtime archive encoded as 28 ordered Base64 chunks (`runtime.part00.b64` through `runtime.part27.b64`).

During the Railway Docker build the chunks are concatenated, decoded, and checked against this required SHA-256:

`88fc9e687de8a2d576021d3202224a2166975538870cf0ce0be861a358e360a2`

Only after that exact archive is verified is it extracted into `/app`. The build then installs the runtime dependencies and runs `validate_release.py`. A checksum mismatch, missing/extra chunk, failed test, wrong app version, or missing Audiveris executable stops the build.

## v0.16.4 change

v0.16.4 corrects the **OMR pickup-measure phase handling** exposed by the Berg Op. 1 regression. It does not change the dissertation Levels mathematics. A legal short first measure can be right-aligned inside its explicit metric frame even when Audiveris does not mark it `abnormal`; a short interior measure cannot trigger this pickup rule.

For the Berg opening, the required theoretical phase is therefore a structural `(1)` before the first sounding event, followed by Levels `1+2+3` on the first sounding event.

## Validation gate

The authoritative v0.16.4 runtime currently passes **40 tests**, including:

- dissertation sequence and recursive-level conformance checks;
- structural-versus-sounding Level-point behavior;
- Berg-opening pickup-phase regression;
- short-interior-measure non-pickup regression;
- OMR/PDF driver contract checks;
- legacy-path and duplicate-source audits.

The required PDF pass remains `Audiveris -batch -step PAGE -save ...`; the required path does not request `-transcribe` or `-export`.

v0.16.4 remains a development build until the exact real Berg PDF is re-run and compared event-by-event with the dissertation reference. No new BWV 661 PDF regression is claimed without the exact test score.
