from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from tone_metric.score_registration import (
    build_layer_anchors_from_symbolic_slots,
    build_symbolic_registration_meta,
)


SHEET_XML = """<?xml version="1.0" encoding="UTF-8"?>
<sheet>
  <page>
    <system id="1">
      <stack id="1" left="100" right="500">
        <slot id="1" x-offset="100" time-offset="0"/>
        <slot id="2" x-offset="300" time-offset="5/8"/>
        <slot id="3" x-offset="250" time-offset="3/4"/>
      </stack>
      <part id="1"><measure id="1"><voice id="1"><slots>
        <entry><key>1</key><value chord="1001" status="BEGIN"/></entry>
        <entry><key>2</key><value chord="1002" status="BEGIN"/></entry>
        <entry><key>3</key><value chord="1003" status="BEGIN"/></entry>
      </slots></voice></measure></part>
      <part id="2"><measure id="2"><voice id="1"><slots>
        <entry><key>1</key><value chord="2001" status="BEGIN"/></entry>
      </slots></voice></measure></part>
      <sig>
        <head-chord id="1001"><bounds x="160" y="100" w="20" h="40"/></head-chord>
        <head-chord id="2001"><bounds x="300" y="300" w="20" h="40"/></head-chord>
        <head-chord id="1002"><bounds x="390" y="110" w="20" h="40"/></head-chord>
        <head-chord id="1003"><bounds x="340" y="120" w="20" h="40"/></head-chord>
        <head id="11"><bounds x="165" y="125" w="20" h="20"/></head>
        <head id="12"><bounds x="305" y="325" w="20" h="20"/></head>
        <head id="21"><bounds x="395" y="135" w="20" h="20"/></head>
        <head id="31"><bounds x="345" y="145" w="20" h="20"/></head>
        <relation source="1001" target="11"><containment/></relation>
        <relation source="2001" target="12"><containment/></relation>
        <relation source="1002" target="21"><containment/></relation>
        <relation source="1003" target="31"><containment/></relation>
      </sig>
    </system>
  </page>
</sheet>
"""


class SymbolicRegistrationTests(unittest.TestCase):
    def _omr(self, directory: Path) -> Path:
        path = directory / "fixture.omr"
        with ZipFile(path, "w", ZIP_DEFLATED) as zf:
            zf.writestr("sheet#1/sheet#1.xml", SHEET_XML)
        return path

    def _analysis(self, meta: dict) -> dict:
        def event(i, offset, levels):
            return {
                "event_index": i,
                "measure_index": 0,
                "measure_number": "4",
                "offset_in_measure_quarter": offset,
                "onset_quarter": offset,
                "duration_quarter": "1/2",
                "tone_metric_levels": levels,
                "attack_key": f"0:{offset}",
            }
        return {
            "analysis_position_policy": "symbolic-attacks-plus-independent-metric-grid",
            "_symbolic_registration_meta": meta,
            "segments": [{
                "events": [
                    event(0, "0", [1, 2]),
                    event(1, "5/2", [3]),
                    event(2, "3", [4]),
                ]
            }],
        }

    def test_registration_uses_exact_time_key_not_x_order(self):
        # Deliberately non-monotonic engraving: score time 5/2 has x=400, while
        # later score time 3 has x=350. Geometry must not rewrite chronology.
        with tempfile.TemporaryDirectory() as td:
            meta = build_symbolic_registration_meta(self._omr(Path(td)))
        anchors, _, stats, warnings = build_layer_anchors_from_symbolic_slots(
            self._analysis(meta), {0: (1000.0, 1000.0)}
        )
        self.assertTrue(meta["available"])
        self.assertEqual(meta["slot_count"], 3)
        self.assertEqual(stats["symbolic_events_expected"], 3)
        self.assertEqual(stats["symbolic_events_mapped"], 3)
        self.assertEqual(stats["symbolic_events_unmapped"], 0)
        self.assertFalse(stats["x_order_used_for_timing"])
        self.assertFalse(stats["x_clustering_used_for_timing"])
        self.assertEqual(warnings, [])
        rows = anchors[0]
        self.assertEqual([r["offset_in_measure_quarter"] for r in rows], ["0", "5/2", "3"])
        self.assertGreater(rows[1]["cx_norm"], rows[2]["cx_norm"])

    def test_simultaneous_far_apart_heads_remain_one_event(self):
        with tempfile.TemporaryDirectory() as td:
            meta = build_symbolic_registration_meta(self._omr(Path(td)))
        first = next(r for r in meta["slot_rows"] if r["offset_quarter"] == "0")
        self.assertEqual(set(first["begin_chord_ids"]), {"1001", "2001"})
        self.assertEqual(set(h["head_id"] for h in first["heads"]), {"11", "12"})
        anchors, _, stats, _ = build_layer_anchors_from_symbolic_slots(
            self._analysis(meta), {0: (1000.0, 1000.0)}
        )
        zero = next(r for r in anchors[0] if r["offset_in_measure_quarter"] == "0")
        self.assertEqual(len(zero["simultaneous_head_ids"]), 2)
        self.assertEqual(stats["symbolic_events_mapped"], 3)

    def test_legacy_canonical_reconstruction_is_absent(self):
        root = Path(__file__).resolve().parents[1]
        self.assertFalse((root / "tone_metric" / "canonical_score.py").exists())
        app_text = (root / "app.py").read_text(encoding="utf-8")
        reg_text = (root / "tone_metric" / "score_registration.py").read_text(encoding="utf-8")
        self.assertNotIn("from tone_metric.canonical_score", app_text)
        self.assertNotIn("build_hits_from_canonical_score(", app_text)
        self.assertNotIn("cluster_tolerance_px", reg_text)
        self.assertNotIn("recover_canonical_columns", reg_text)


if __name__ == "__main__":
    unittest.main()
