# Hit-only clean sequence engine

This branch contains one hit pipeline only. Every semantic attacking note/chord enters exactly one staff sequence; tied-right continuations are excluded. Exact semantic BEGIN time groups simultaneous voices within a staff. Staff sequences are reconciled by one-to-one, order-preserving alignment. Recognized untimed attacks remain in the sequence and cannot silently disappear.

Every red strike is anchored to the center of an actual attacking notehead. Synthetic/fitted strike positions, isotonic correction, global x-clustering, chord-box-overlap simultaneity, displaced-voice special cases, canonical reconstruction, and legacy analyzer modules are absent and forbidden by the Docker release gate.
