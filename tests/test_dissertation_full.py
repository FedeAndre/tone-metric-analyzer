from fractions import Fraction

from tone_metric.dissertation_full import analyze_full
from tone_metric.models import Hit, MeasureInfo, NoteAttack


def _measure(num, den, duration):
    return MeasureInfo(
        index=0,
        number="1",
        start=Fraction(0),
        full_duration=Fraction(duration),
        actual_duration=Fraction(duration),
        pickup_shift=Fraction(0),
        numerator=num,
        denominator=den,
        implicit=False,
    )


def _hit(t, dur=Fraction(1, 4), *, actual=None, normal=None, group=None, start=None, end=None):
    t = Fraction(t)
    source = NoteAttack(
        onset=t,
        duration=Fraction(dur),
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
        duration=Fraction(dur),
        measure_index=0,
        measure_number="1",
        offset_in_measure=t,
        sources=[source],
        tuplet_arity=actual if actual in (2, 3) else None,
        tuplet_group=group,
        tuplet_span_start=Fraction(start) if start is not None else None,
        tuplet_span_end=Fraction(end) if end is not None else None,
    )


def _event_at(result, t):
    target = Fraction(t)
    for segment in result["levels"]["segments"]:
        for event in segment["events"]:
            if Fraction(event["onset_quarter"]) == target:
                return event
    raise AssertionError(f"event {target} not found")


def test_explicit_triplet_in_4_4_is_mapped():
    hits = [
        _hit(0, actual=3, normal=2, group="g3", start=0, end=1),
        _hit(Fraction(1, 3), actual=3, normal=2, group="g3", start=0, end=1),
        _hit(Fraction(2, 3), actual=3, normal=2, group="g3", start=0, end=1),
        _hit(1),
        _hit(2),
        _hit(3),
    ]
    result = analyze_full(hits, [_measure(4, 4, 4)])
    assert _event_at(result, Fraction(1, 3))["tone_metric_levels"]
    assert _event_at(result, Fraction(2, 3))["tone_metric_levels"]
    assert result["summary"]["triplet_spans"] == 1
    assert result["summary"]["duplet_spans"] == 0
    assert result["waves"]
    assert result["trees"]["nodes"]


def test_explicit_duplet_in_6_8_is_mapped():
    end = Fraction(3, 2)
    hits = [
        _hit(0, actual=2, normal=3, group="g2", start=0, end=end),
        _hit(Fraction(3, 4), actual=2, normal=3, group="g2", start=0, end=end),
        _hit(end),
        _hit(Fraction(9, 4)),
    ]
    result = analyze_full(hits, [_measure(6, 8, 3)])
    assert _event_at(result, Fraction(3, 4))["tone_metric_levels"]
    assert result["summary"]["duplet_spans"] == 1
    assert result["summary"]["triplet_spans"] == 0


def test_tuplets_are_not_inferred_from_spacing():
    hits = [_hit(0), _hit(Fraction(1, 3)), _hit(1)]
    result = analyze_full(hits, [_measure(4, 4, 4)])
    assert result["summary"]["triplet_spans"] == 0
    assert result["levels"]["tuplet_extension_policy"]["spacing_inference"] is False
