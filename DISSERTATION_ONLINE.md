# Tone-Metric Dissertation Online Analyzer

This standalone Railway entry point exposes the dissertation workflow in the order:

**Levels → Waves → Pivots → Trees**

It deliberately leaves the existing analyzer and validated ordinary Levels engine intact.

## Dissertation rules encoded

- Binary recursive sequence: `1, 2, 3, 5, 9, 17, ...` (`y = 2x - 1`).
- Ternary recursive sequence: `1, 2, 4, 10, 28, 82, ...` (`y = 3x - 2`).
- Level 1 is tactus-derived; subsequent Levels recursively fill unresolved spans, completing a denomination before moving to a finer one.
- Mixed binary/ternary meters use the appropriate local arity, e.g. 12/8 is binary at the dotted-quarter level and ternary at the eighth-note level.
- A wave is the envelope created by superimposed recursive Levels.
- A simple pivot descends one Level before a renewed ascent; a compound pivot descends at least two Levels before the renewed ascent.
- Tree nodes are sonic events at their lowest occupied Level. Branches run left-to-right, may stay on the same Level or rise from a lower to a higher Level, permit multiple outgoing branches but one incoming branch, and privilege the nearest legal predecessor.

## Explicit duplet/triplet extension

The dissertation defines binary and ternary metric organizations, but does **not** provide a general rule for MusicXML tuplet notation. The online implementation therefore marks the following as an explicit software extension:

- `actual-notes = 2`: local **binary** subdivision stage.
- `actual-notes = 3`: local **ternary** subdivision stage.
- The tuplet must carry an explicit span from symbolic notation metadata.
- Tuplet arity is never guessed from attack spacing.
- Unsupported arities, missing span metadata, and crossing tuplet spans produce warnings rather than silent guesses.

This preserves the dissertation's binary/ternary recursive logic while making notated duplets and triplets analyzable.

## Railway

Use `Dockerfile.dissertation`. The service exposes:

- `GET /` — web analyzer
- `GET /health` — deployment health
- `POST /api/analyze` — PDF, MXL, MusicXML or XML upload
