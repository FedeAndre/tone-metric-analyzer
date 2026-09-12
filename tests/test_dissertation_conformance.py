import unittest
from fractions import Fraction

from tone_metric.engine import (
    UnsupportedMeterError,
    analyze,
    chapter4_placement_case,
    geometric_sequence,
    meter_profile,
)
from tone_metric.models import Hit, MeasureInfo


def measure(index, number, start, duration, num, den, *, actual=None, shift=0):
    duration = Fraction(duration)
    return MeasureInfo(
        index=index,
        number=str(number),
        start=Fraction(start),
        full_duration=duration,
        actual_duration=Fraction(actual if actual is not None else duration),
        pickup_shift=Fraction(shift),
        numerator=num,
        denominator=den,
    )


def hit(t, mi, number, offset, duration=Fraction(1, 4)):
    return Hit(
        onset=Fraction(t),
        duration=Fraction(duration),
        measure_index=mi,
        measure_number=str(number),
        offset_in_measure=Fraction(offset),
        sources=[],
    )


def regular_hits(start, end, step, measure_duration):
    out = []
    t = Fraction(start)
    step = Fraction(step)
    measure_duration = Fraction(measure_duration)
    while t < Fraction(end):
        mi = int(t // measure_duration)
        out.append(hit(t, mi, mi + 1, t - mi * measure_duration, step))
        t += step
    return out


class DissertationSequenceTests(unittest.TestCase):
    def test_binary_sequence_includes_corrected_9_and_33(self):
        self.assertEqual(geometric_sequence(2, 33), [1, 2, 3, 5, 9, 17, 33])

    def test_ternary_sequence(self):
        self.assertEqual(geometric_sequence(3, 28), [1, 2, 4, 10, 28])

    def test_no_generic_meter_fallback(self):
        with self.assertRaises(UnsupportedMeterError):
            meter_profile(5, 4)

    def test_4_4_is_pure_binary_not_generic_triplet(self):
        p = meter_profile(4, 4)
        self.assertEqual(p.top_sequence_arity, 2)
        self.assertEqual(p.first_subdivision_arity, 2)

    def test_12_8_is_binary_then_ternary(self):
        p = meter_profile(12, 8)
        self.assertEqual(p.beat_unit, Fraction(3, 2))
        self.assertEqual(p.top_sequence_arity, 2)
        self.assertEqual(p.first_subdivision_arity, 3)


class Chapter4PlacementTests(unittest.TestCase):
    def test_case_1_equal_limits(self):
        self.assertEqual(chapter4_placement_case(3, 3), ("equal-limits", 4))

    def test_case_2_left_lower_worked_figure_direction(self):
        # The worked figure bridges a lower left boundary to a taller right stack.
        self.assertEqual(chapter4_placement_case(2, 4), ("left-lower-than-right", 3))

    def test_case_3_right_lower(self):
        self.assertEqual(chapter4_placement_case(4, 2), ("right-lower-than-left", 3))


class Chapter1WorkedExampleTests(unittest.TestCase):
    def setUp(self):
        self.measures = [measure(i, i + 1, 4 * i, 4, 4, 4) for i in range(3)]

    def _segment(self, step):
        return analyze(regular_hits(0, 12, step, 4), self.measures)["segments"][0]

    def test_examples_1_1_to_1_3_quarter_denomination(self):
        # Chapter 1 progressively shows Levels 1, 2 and 3 at the quarter-note
        # denomination.  These positions are asserted independently of finer notes.
        s = self._segment(Fraction(1))
        positions = s["level_positions_quarter"]
        self.assertEqual(positions["1"], ["0", "1", "2", "4", "8"])
        self.assertEqual(positions["2"], ["2", "3", "4", "5", "6", "8", "9", "10"])
        self.assertEqual(positions["3"], ["6", "7", "8", "10", "11"])
        self.assertEqual(s["max_level"], 3)

    def test_example_1_4_eighth_denomination_is_relative(self):
        s = self._segment(Fraction(1, 2))
        by_time = {e["onset_quarter"]: e["tone_metric_levels"] for e in s["events"]}
        # First two quarter-note spans remain at Level 2; the next unit rises to 3,
        # matching the worked figure's relative rather than absolute note-value use.
        self.assertEqual(by_time["0"], [1, 2])
        self.assertEqual(by_time["1/2"], [2])
        self.assertEqual(by_time["1"], [1, 2])
        self.assertEqual(by_time["3/2"], [2])
        self.assertEqual(by_time["2"], [1, 2, 3])
        self.assertEqual(by_time["5/2"], [3])
        self.assertEqual(s["max_level"], 4)
        self.assertEqual(s["denomination_stages"][1]["denomination_name"], "eighth note")

    def test_example_1_5_sixteenth_denomination_and_first_stack(self):
        s = self._segment(Fraction(1, 4))
        by_time = {e["onset_quarter"]: e["tone_metric_levels"] for e in s["events"]}
        # The high-resolution dissertation figure begins with the stack 1+2+3.
        self.assertEqual(by_time["0"], [1, 2, 3])
        self.assertEqual(by_time["1/4"], [3])
        self.assertEqual(by_time["1/2"], [2, 3])
        self.assertEqual(by_time["2"], [1, 2, 3, 4])
        self.assertEqual(s["max_level"], 5)
        self.assertEqual(
            [x["denomination_name"] for x in s["denomination_stages"]],
            ["quarter note", "eighth note", "sixteenth note"],
        )

    def test_stage_snapshot_prevents_same_stage_contamination(self):
        s = self._segment(Fraction(1, 2))
        audit = s["denomination_stages"][1]
        self.assertTrue(audit["boundary_snapshot_only"])
        # Both directional Chapter-4 cases actually occur in this tiny reference
        # example; their presence protects against a left-to-right mutation bug.
        self.assertGreater(audit["placement_cases"]["left-lower-than-right"], 0)
        self.assertGreater(audit["placement_cases"]["right-lower-than-left"], 0)


class MixedMeterWorkedExampleTests(unittest.TestCase):
    def test_example_4_9_12_8_stage_order(self):
        m = [measure(0, 1, 0, 6, 12, 8)]
        h = regular_hits(0, 6, Fraction(1, 2), 6)  # eighth-note attacks
        s = analyze(h, m)["segments"][0]
        self.assertEqual(s["beat_unit_name"], "dotted quarter note")
        self.assertEqual(s["denomination_stages"][0]["subdivision_arity"], 2)
        self.assertEqual(s["denomination_stages"][1]["subdivision_arity"], 3)
        self.assertEqual(s["denomination_stages"][1]["denomination_name"], "eighth note")
        self.assertFalse(s["arity_policy"]["spacing_inference_enabled"])


class BergOpeningTests(unittest.TestCase):
    def test_example_6_6_opening_parenthetical_and_first_attack_123(self):
        # The dissertation graphic treats the pickup inside a full 3/4 framework.
        # A two-quarter pickup is therefore right-aligned by one quarter.  The first
        # sounding attack occurs at t=1; finer activity in that tactus span raises it
        # to the visually verified 1+2+3 stack, while t=0 remains structural only.
        m = [measure(0, 1, 0, 3, 3, 4, actual=2, shift=1)]
        h = [
            hit(1, 0, 1, 1, Fraction(1, 2)),
            hit(Fraction(3, 2), 0, 1, Fraction(3, 2), Fraction(1, 2)),
            hit(2, 0, 1, 2, Fraction(1)),
        ]
        s = analyze(h, m)["segments"][0]
        first = s["events"][0]
        self.assertEqual(first["onset_quarter"], "1")
        self.assertEqual(first["tone_metric_levels"], [1, 2, 3])
        structural = {p["time_quarter"]: p for p in s["structural_points"]}
        self.assertEqual(structural["0"]["levels"], [1])
        self.assertFalse(structural["0"]["attack"])
        self.assertTrue(structural["0"]["parenthetical"])
        self.assertTrue(structural["1"]["attack"])
        self.assertFalse(structural["1"]["parenthetical"])


class MeterResetTests(unittest.TestCase):
    def test_meter_change_restarts_level1(self):
        measures = [
            measure(0, 1, 0, 4, 4, 4),
            measure(1, 2, 4, 6, 12, 8),
        ]
        hits = [hit(0, 0, 1, 0), hit(4, 1, 2, 0)]
        r = analyze(hits, measures)
        self.assertEqual(len(r["segments"]), 2)
        self.assertIn("0", r["segments"][0]["level1_positions_quarter"])
        self.assertIn("4", r["segments"][1]["level1_positions_quarter"])


if __name__ == "__main__":
    unittest.main()
