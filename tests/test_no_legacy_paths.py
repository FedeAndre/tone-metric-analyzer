import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class LegacyPathAuditTests(unittest.TestCase):
    def test_removed_old_recursive_entry_points(self):
        engine = (ROOT / "tone_metric" / "engine.py").read_text(encoding="utf-8")
        for forbidden in (
            "_refine_span(",
            "_ternary_spans_from_hits(",
            "_denominator_reachable_by_local_arity(",
            "explicit_local_ternary_span_count",
        ):
            self.assertNotIn(forbidden, engine)

    def test_no_pdf_symbolic_silent_fallback(self):
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn('result["silent_event_fallback_enabled"] = False', app)
        self.assertNotIn("symbolic MusicXML hits were used", app)
        self.assertNotIn("Preserve the previous symbolic fallback", app)

    def test_one_levels_engine_imported_by_app(self):
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertEqual(app.count("from tone_metric.engine import analyze"), 1)

    def test_attack_registration_has_no_nearest_event_path(self):
        reg = (ROOT / "tone_metric" / "score_registration.py").read_text(encoding="utf-8")
        self.assertIn("canonical-score-event-exact-origin-column", reg)
        self.assertNotIn("nearest-column", reg)


if __name__ == "__main__":
    unittest.main()
