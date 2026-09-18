# Independent HOMR reader probe

Clean-room experimental score-reading service for the Tone-Metric project.

This branch intentionally contains no Audiveris integration and no Tone-Metric hit engine.
Its sole purpose is to test HOMR as an independent OMR source and expose normalized
MusicXML attack diagnostics. The Docker build downloads HOMR's models and must complete
a real OMR pass over a fixed public score image before the image can build.

Nothing in this branch is wired into the production/development Tone-Metric analyzer.
