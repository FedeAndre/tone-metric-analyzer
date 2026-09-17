# Reconciled Hit Engine

Clean-room hit-only score marker.

The application performs one task: draw one vertical strike for every new global note/chord attack.

Architecture:
- exact symbolic BEGIN onset establishes timed hit identity;
- tied continuations are excluded;
- simultaneous voices/staves at one exact onset are one hit;
- every recognized sounding semantic chord is independently audited;
- omitted semantic chords are reconciled only with independent simultaneity evidence;
- otherwise they become recovered hits;
- ambiguous evidence fails closed instead of guessing;
- geometry can position hits but cannot change an already established symbolic hit identity.

No tone-metric levels, waves, trees, or prior analyzer logic are present.
