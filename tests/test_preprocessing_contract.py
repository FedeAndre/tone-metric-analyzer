import unittest
from fractions import Fraction

from tone_metric.models import Hit, NoteAttack
from tone_metric.preprocessing import prepare_hits_for_dissertation_core


def source(pitch, *, tuplet=False, voice="1"):
    return NoteAttack(
        onset=Fraction(0), duration=Fraction(1, 2), measure_index=0,
        measure_number="1", offset_in_measure=Fraction(0), part_id="P1",
        voice=voice, staff="1", pitch=pitch,
        tuplet_actual_notes=3 if tuplet else None,
        tuplet_normal_notes=2 if tuplet else None,
        tuplet_group="g1" if tuplet else None,
    )


class PreprocessingTests(unittest.TestCase):
    def test_tuplet_only_hit_is_excluded(self):
        h = Hit(Fraction(0), Fraction(1, 3), 0, "1", Fraction(0), [source("C4", tuplet=True)], tuplet_arity=3)
        out, audit = prepare_hits_for_dissertation_core([h])
        self.assertEqual(out, [])
        self.assertEqual(audit["ignored_tuplet_hit_count"], 1)
        self.assertFalse(audit["spacing_based_arity_inference"])

    def test_other_voice_at_same_time_survives_tuplet_filter(self):
        h = Hit(
            Fraction(0), Fraction(1, 2), 0, "1", Fraction(0),
            [source("C4", tuplet=True, voice="1"), source("G4", tuplet=False, voice="2")],
            tuplet_arity=3,
        )
        out, audit = prepare_hits_for_dissertation_core([h])
        self.assertEqual(len(out), 1)
        self.assertEqual([s.pitch for s in out[0].sources], ["G4"])
        self.assertIsNone(out[0].tuplet_arity)
        self.assertEqual(audit["retained_mixed_simultaneous_hit_count"], 1)

    def test_unmarked_hit_is_unchanged(self):
        h = Hit(Fraction(1), Fraction(1), 0, "1", Fraction(1), [source("E4")])
        out, audit = prepare_hits_for_dissertation_core([h])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].onset, Fraction(1))
        self.assertEqual(out[0].sources[0].pitch, "E4")


if __name__ == "__main__":
    unittest.main()
