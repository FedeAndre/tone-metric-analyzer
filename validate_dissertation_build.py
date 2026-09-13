from fractions import Fraction

from tone_metric.dissertation_full import analyze_full
from tone_metric.models import Hit, MeasureInfo, NoteAttack


def measure(num, den, duration):
    return MeasureInfo(0, "1", Fraction(0), Fraction(duration), Fraction(duration), Fraction(0), num, den, False)


def hit(t, *, actual=None, normal=None, group=None, start=None, end=None):
    t = Fraction(t)
    source = NoteAttack(
        onset=t,
        duration=Fraction(1, 4),
        measure_index=0,
        measure_number="1",
        offset_in_measure=t,
        part_id="P1",
        voice="1",
        staff="1",
        pitch="C4",
        tuplet_actual_notes=actual,
        tuplet_normal_notes=normal,
        tuplet_group=group,
        tuplet_span_start=Fraction(start) if start is not None else None,
        tuplet_span_end=Fraction(end) if end is not None else None,
    )
    return Hit(
        onset=t,
        duration=Fraction(1, 4),
        measure_index=0,
        measure_number="1",
        offset_in_measure=t,
        sources=[source],
        tuplet_arity=actual if actual in (2, 3) else None,
        tuplet_group=group,
        tuplet_span_start=Fraction(start) if start is not None else None,
        tuplet_span_end=Fraction(end) if end is not None else None,
    )


def levels_at(result, t):
    target = Fraction(t)
    for segment in result["levels"]["segments"]:
        for event in segment["events"]:
            if Fraction(event["onset_quarter"]) == target:
                return event["tone_metric_levels"]
    return []


triplet = analyze_full(
    [
        hit(0, actual=3, normal=2, group="triplet", start=0, end=1),
        hit(Fraction(1, 3), actual=3, normal=2, group="triplet", start=0, end=1),
        hit(Fraction(2, 3), actual=3, normal=2, group="triplet", start=0, end=1),
        hit(1), hit(2), hit(3),
    ],
    [measure(4, 4, 4)],
)
assert levels_at(triplet, Fraction(1, 3))
assert levels_at(triplet, Fraction(2, 3))
assert triplet["summary"]["triplet_spans"] == 1
assert triplet["waves"]
assert triplet["trees"]["nodes"]

end = Fraction(3, 2)
duplet = analyze_full(
    [
        hit(0, actual=2, normal=3, group="duplet", start=0, end=end),
        hit(Fraction(3, 4), actual=2, normal=3, group="duplet", start=0, end=end),
        hit(end), hit(Fraction(9, 4)),
    ],
    [measure(6, 8, 3)],
)
assert levels_at(duplet, Fraction(3, 4))
assert duplet["summary"]["duplet_spans"] == 1

print("dissertation-build-validation: PASS")
