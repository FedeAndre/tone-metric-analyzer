from __future__ import annotations
import inspect
import unittest
from fractions import Fraction
import core


class ConstraintHitEngineTests(unittest.TestCase):
    def test_rest_values(self):
        self.assertEqual(core._rest_duration('QUARTER_REST'), Fraction(1,4))
        self.assertEqual(core._rest_duration('ONE_16TH_REST'), Fraction(1,16))

    def test_no_x_cluster_hit_engine(self):
        src = inspect.getsource(core)
        forbidden = [
            '_cluster_chords_by_x', 'ONSET_CLUSTER_INTERLINE_FRACTION',
            'DISPLACED_MIN_DX', 'isotonic', 'PAVA', 'build_hits_from_canonical',
            'canonical_score', 'DISPLAY_MIN_GAP',
        ]
        for token in forbidden:
            self.assertNotIn(token, src)

    def test_no_score_specific_ids_or_measure_rules(self):
        src = inspect.getsource(core)
        for token in ['11248', '11251', '11292', '11295', '11298', '11314', '11445', '11447', '8927', '8928', '8931']:
            self.assertNotIn(token, src)

    def test_strike_has_measure_identity(self):
        s = core.Strike(0,0,1,10.0,1.0,2.0,100.0,200.0,False)
        self.assertEqual(s.measure_index, 1)

    def test_diagnostics_exposes_fail_closed_fields(self):
        d = core.Diagnostics(1,2,2,0,0,0,0,0,0)
        self.assertEqual(d.unresolved_attack_chords, 0)
        self.assertEqual(d.synthetic_strike_positions, 0)


if __name__ == '__main__':
    unittest.main()
