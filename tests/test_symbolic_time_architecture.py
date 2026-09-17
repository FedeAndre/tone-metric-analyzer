from __future__ import annotations

import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from tone_metric.engine import analyze
from tone_metric.models import Hit, MeasureInfo, NoteAttack
from tone_metric.musicxml import parse_musicxml
from tone_metric.score_registration import build_layer_anchors_from_symbolic_layout


class SymbolicTimeArchitectureTests(unittest.TestCase):
    def _source(self, onset: Fraction, offset: Fraction, voice: str, pitch: str) -> NoteAttack:
        return NoteAttack(
            onset=onset,
            duration=Fraction(1, 2),
            measure_index=0,
            measure_number="1",
            offset_in_measure=offset,
            part_id="P1",
            voice=voice,
            staff="1",
            pitch=pitch,
        )

    def _hit(self, onset: Fraction, sources: int = 1) -> Hit:
        src = [self._source(onset, onset, str(i + 1), f"C{i + 4}") for i in range(sources)]
        return Hit(
            onset=onset,
            duration=Fraction(1, 2),
            measure_index=0,
            measure_number="1",
            offset_in_measure=onset,
            sources=src,
        )

    def test_musicxml_merges_simultaneous_notes_despite_large_x_offset(self):
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1"><part-name>Upper</part-name></score-part>
    <score-part id="P2"><part-name>Lower</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes><divisions>2</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
      <note default-x="10"><pitch><step>C</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice><staff>1</staff></note>
      <note default-x="60"><rest/><duration>2</duration><voice>1</voice><staff>1</staff></note>
      <note default-x="100"><pitch><step>D</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice><staff>1</staff></note>
      <note default-x="140"><pitch><step>F</step><alter>1</alter><octave>4</octave></pitch><duration>2</duration><voice>1</voice><staff>1</staff><accidental>sharp</accidental></note>
    </measure>
  </part>
  <part id="P2">
    <measure number="1">
      <attributes><divisions>2</divisions></attributes>
      <note default-x="52"><pitch><step>G</step><octave>3</octave></pitch><duration>2</duration><voice>2</voice><staff>1</staff></note>
      <forward><duration>2</duration></forward>
      <note default-x="108"><pitch><step>B</step><octave>3</octave></pitch><duration>2</duration><voice>2</voice><staff>1</staff></note>
    </measure>
  </part>
</score-partwise>
"""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "fixture.musicxml"
            path.write_text(xml, encoding="utf-8")
            hits, measures, _warnings = parse_musicxml(path)

        self.assertEqual([h.onset for h in hits], [Fraction(0), Fraction(2), Fraction(3)])
        self.assertEqual(len(hits[0].sources), 2)
        self.assertEqual({s.pitch for s in hits[0].sources}, {"C4", "G3"})
        self.assertEqual({s.pitch for s in hits[-1].sources}, {"F#4"})
        self.assertEqual(len(measures), 1)

    def test_buxtehude_measure4_pattern_cannot_be_rewritten_by_geometry(self):
        measure = MeasureInfo(0, "4", Fraction(0), Fraction(4), Fraction(4), Fraction(0), 4, 4)
        hits = [
            self._hit(Fraction(0)),
            self._hit(Fraction(1)),
            self._hit(Fraction(2), sources=3),
            self._hit(Fraction(5, 2), sources=2),
            self._hit(Fraction(3)),
        ]
        result = analyze(hits, [measure])
        onsets = [e["onset_quarter"] for e in result["segments"][0]["events"]]
        self.assertEqual(onsets, ["0", "1", "2", "5/2", "3"])
        self.assertNotIn("3/2", onsets)
        self.assertEqual(len(onsets), 5)

    def test_metric_grid_keeps_legitimate_empty_structural_positions(self):
        measure = MeasureInfo(0, "5", Fraction(0), Fraction(4), Fraction(4), Fraction(0), 4, 4)
        result = analyze([self._hit(Fraction(0)), self._hit(Fraction(1, 4))], [measure])
        points = result["segments"][0]["structural_points"]
        by_time = {p["time_quarter"]: p for p in points}
        self.assertIn("1/2", by_time)
        self.assertFalse(by_time["1/2"]["attack"])
        self.assertTrue(by_time["1/2"]["levels"])
        self.assertEqual(result["analysis_position_policy"], "symbolic-attacks-plus-independent-metric-grid")

    def test_buxtehude_measure46_ten_ground_truth_onsets_remain_ten_events(self):
        measure = MeasureInfo(0, "46", Fraction(0), Fraction(4), Fraction(4), Fraction(0), 4, 4)
        times = [
            Fraction(0), Fraction(1, 2), Fraction(3, 4), Fraction(1),
            Fraction(3, 2), Fraction(7, 4), Fraction(2), Fraction(5, 2),
            Fraction(3), Fraction(7, 2),
        ]
        hits = [self._hit(t, sources=(3 if i in {1, 6, 8} else 1)) for i, t in enumerate(times)]
        result = analyze(hits, [measure])
        events = result["segments"][0]["events"]
        self.assertEqual(len(events), 10)
        self.assertEqual([Fraction(e["onset_quarter"]) for e in events], times)

    def test_registration_uses_attack_key_and_cannot_split_simultaneous_geometry(self):
        measure = MeasureInfo(0, "4", Fraction(0), Fraction(4), Fraction(4), Fraction(0), 4, 4)
        result = analyze([self._hit(Fraction(0), sources=2), self._hit(Fraction(1))], [measure])
        result["_symbolic_visual_groups"] = [
            {
                "serial": 0,
                "attack_key": "0:0",
                "analyzed": True,
                "layout_page": 0,
                "layout_system": 0,
                "layout_x_abs": 10.0,
                "layout_page_width": 200.0,
            },
            {
                "serial": 1,
                "attack_key": "0:0",
                "analyzed": True,
                "layout_page": 0,
                "layout_system": 0,
                "layout_x_abs": 52.0,
                "layout_page_width": 200.0,
            },
            {
                "serial": 2,
                "attack_key": "0:1",
                "analyzed": True,
                "layout_page": 0,
                "layout_system": 0,
                "layout_x_abs": 100.0,
                "layout_page_width": 200.0,
            },
            {
                "serial": 3,
                "attack_key": "0:3/2",
                "analyzed": False,
                "layout_page": 0,
                "layout_system": 0,
                "layout_x_abs": 75.0,
                "layout_page_width": 200.0,
            },
        ]
        anchors, _structural, stats, _warnings = build_layer_anchors_from_symbolic_layout(
            result, {0: (1000.0, 1400.0)}
        )
        mapped = [a for rows in anchors.values() for a in rows]
        self.assertEqual(len(mapped), 2)
        self.assertEqual({a["attack_key"] for a in mapped}, {"0:0", "0:1"})
        self.assertEqual(stats["registration_can_create_attacks"], False)
        self.assertEqual(stats["registration_can_retime_attacks"], False)
        self.assertEqual(stats["simultaneous_events_with_multiple_visual_positions"], 1)
        self.assertEqual(stats["max_simultaneous_visual_x_spread"], 42.0)
        self.assertNotIn("0:3/2", {a["attack_key"] for a in mapped})


if __name__ == "__main__":
    unittest.main()
