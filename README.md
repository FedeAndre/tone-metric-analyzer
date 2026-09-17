# Hit-only score marker — clean-room start

This branch intentionally contains no Tone-Metric Levels, waves, pivots, trees, metric-grid logic, canonical reconstruction, legacy registration, or previous analyzer modules.

It does one thing only: for an uploaded PDF score, identify every new sounding onset and draw one vertical strike at that score-time position.

Rules:
- a new note or chord onset is a hit;
- simultaneous notes in any voices/staves/parts are one hit;
- chord members at the same onset are one hit;
- rests are not hits;
- grace notes are not hits;
- a tied continuation is not a new hit;
- the first note of a tie is a hit;
- tuplets are ordinary sounding hits;
- if any symbolic hit cannot be matched exactly to a physical score-time slot, output fails instead of guessing.

No other analysis is performed.
