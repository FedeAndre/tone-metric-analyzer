import asyncio
import json
import subprocess
import tempfile
import unittest
from io import BytesIO
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from tone_metric.omr import AudiverisOmrResult, omr_to_annotations, pdf_to_omr
from tone_metric.canonical_score import build_measure_framework_from_omr
from tone_metric.models import Hit, MeasureInfo


class OmrPdfContractTests(unittest.TestCase):
    def test_required_pdf_pass_uses_page_step_save_without_transcribe_or_export(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pdf = root / "score.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")
            out = root / "omr"
            captured = {}

            def fake_run(cmd, args, timeout=900):
                captured["cmd"] = cmd
                captured["args"] = list(args)
                self.assertIn("-step", args)
                self.assertEqual(args[args.index("-step") + 1], "PAGE")
                self.assertIn("-save", args)
                self.assertNotIn("-transcribe", args)
                self.assertNotIn("-export", args)
                output = Path(args[args.index("-output") + 1])
                output.mkdir(parents=True, exist_ok=True)
                with ZipFile(output / "score.omr", "w") as zf:
                    zf.writestr("book.xml", "<book/>")
                    zf.writestr("sheet#1/sheet#1.xml", "<sheet/>")
                return subprocess.CompletedProcess([cmd] + list(args), 0, "saved")

            with patch("tone_metric.omr.find_audiveris", return_value="Audiveris.exe"), patch(
                "tone_metric.omr._run", side_effect=fake_run
            ):
                result = pdf_to_omr(pdf, out)

            self.assertEqual(result.path.name, "score.omr")
            self.assertEqual(result.returncode, 0)
            self.assertFalse(result.salvaged_after_nonzero)
            self.assertNotIn("-transcribe", captured["args"])
            self.assertNotIn("-export", captured["args"])

    def test_nonzero_exit_is_accepted_only_when_native_omr_is_valid(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pdf = root / "score.pdf"; pdf.write_bytes(b"%PDF-1.4\n")
            out = root / "omr"

            def fake_run(cmd, args, timeout=900):
                output = Path(args[args.index("-output") + 1]); output.mkdir(parents=True, exist_ok=True)
                with ZipFile(output / "score.omr", "w") as zf:
                    zf.writestr("book.xml", "<book/>")
                    zf.writestr("sheet#1/sheet#1.xml", "<sheet/>")
                return subprocess.CompletedProcess([cmd] + list(args), -9, "late secondary failure")

            with patch("tone_metric.omr.find_audiveris", return_value="Audiveris.exe"), patch(
                "tone_metric.omr._run", side_effect=fake_run
            ):
                result = pdf_to_omr(pdf, out)
            self.assertEqual(result.returncode, -9)
            self.assertTrue(result.salvaged_after_nonzero)

    def test_nonzero_exit_with_missing_or_invalid_omr_is_fatal(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pdf = root / "score.pdf"; pdf.write_bytes(b"%PDF-1.4\n")
            out = root / "omr"
            def fake_run(cmd, args, timeout=900):
                output = Path(args[args.index("-output") + 1]); output.mkdir(parents=True, exist_ok=True)
                (output / "score.omr").write_bytes(b"not a zip")
                return subprocess.CompletedProcess([cmd] + list(args), -9, "failed")
            with patch("tone_metric.omr.find_audiveris", return_value="Audiveris.exe"), patch(
                "tone_metric.omr._run", side_effect=fake_run
            ):
                with self.assertRaisesRegex(RuntimeError, "unusable .omr"):
                    pdf_to_omr(pdf, out)

    def test_annotation_pass_uses_saved_omr_without_retranscription_or_export(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            omr = root / "score.omr"; omr.write_bytes(b"placeholder")
            out = root / "annotations"
            captured = {}
            def fake_run(cmd, args, timeout=900):
                captured["args"] = list(args)
                self.assertIn("-annotate", args)
                self.assertNotIn("-transcribe", args)
                self.assertNotIn("-export", args)
                self.assertEqual(Path(args[-1]), omr)
                output = Path(args[args.index("-output") + 1]); output.mkdir(parents=True, exist_ok=True)
                with ZipFile(output / "score-annotations.zip", "w") as zf:
                    zf.writestr("manifest.txt", "ok")
                return subprocess.CompletedProcess([cmd] + list(args), 0, "annotated")
            with patch("tone_metric.omr.find_audiveris", return_value="Audiveris.exe"), patch(
                "tone_metric.omr._run", side_effect=fake_run
            ):
                archive, _log = omr_to_annotations(omr, out)
            self.assertIsNotNone(archive)
            self.assertNotIn("-transcribe", captured["args"])
            self.assertNotIn("-export", captured["args"])

    @staticmethod
    def _write_omr(path: Path, xml: str):
        with ZipFile(path, "w") as zf:
            zf.writestr("sheet#1/sheet#1.xml", xml)

    def test_measure_framework_is_built_directly_from_omr_explicit_meters(self):
        xml = """<sheet>
          <page><system>
            <time-custom id="t1" time-rational="4/4"/>
            <time-custom id="t2" time-rational="3/4"/>
            <stack id="s1" expected="1/1" duration="1/4"/>
            <stack id="s2" expected="1/1" duration="1/1"/>
            <stack id="s3" expected="3/4" duration="3/4"/>
            <part>
              <measure abnormal="true"><times>t1</times></measure>
              <measure/>
              <measure><times>t2</times></measure>
            </part>
          </system></page>
        </sheet>"""
        with tempfile.TemporaryDirectory() as td:
            omr = Path(td) / "score.omr"
            self._write_omr(omr, xml)
            measures, warnings, meta = build_measure_framework_from_omr(omr)

        self.assertEqual([(m.numerator, m.denominator) for m in measures], [(4, 4), (4, 4), (3, 4)])
        self.assertTrue(measures[0].implicit)
        self.assertEqual(str(measures[0].actual_duration), "1")
        self.assertEqual(str(measures[0].pickup_shift), "3")
        self.assertEqual(str(measures[1].start), "4")
        self.assertEqual(str(measures[2].start), "8")
        self.assertEqual(meta["source"], "omr-only-explicit-meter-framework")
        self.assertFalse(meta["musicxml_required_for_pdf"])
        self.assertEqual([x["meter"] for x in meta["meter_changes"]], ["4/4", "3/4"])
        self.assertEqual(warnings, [])

    def test_meter_is_never_inferred_from_omr_duration(self):
        xml = """<sheet><page><system>
          <stack id="s1" expected="1/1" duration="1/1"/>
          <part><measure/></part>
        </system></page></sheet>"""
        with tempfile.TemporaryDirectory() as td:
            omr = Path(td) / "score.omr"
            self._write_omr(omr, xml)
            with self.assertRaisesRegex(ValueError, "will not infer meter/arity"):
                build_measure_framework_from_omr(omr)
            measures, _warnings, _meta = build_measure_framework_from_omr(omr, initial_meter_override=(2, 2))
        self.assertEqual((measures[0].numerator, measures[0].denominator), (2, 2))

    def test_later_time_signature_is_not_applied_retroactively(self):
        xml = """<sheet><page><system>
          <time-custom id="t2" time-rational="3/4"/>
          <stack id="s1" expected="1/1" duration="1/1"/>
          <stack id="s2" expected="3/4" duration="3/4"/>
          <part><measure/><measure><times>t2</times></measure></part>
        </system></page></sheet>"""
        with tempfile.TemporaryDirectory() as td:
            omr = Path(td) / "score.omr"
            self._write_omr(omr, xml)
            with self.assertRaisesRegex(ValueError, "No explicit initial meter"):
                build_measure_framework_from_omr(omr)


    def test_pdf_app_flow_does_not_require_symbolic_export(self):
        import app as app_module
        from fastapi import UploadFile

        measures = [MeasureInfo(
            index=0,
            number="1",
            start=Fraction(0),
            full_duration=Fraction(4),
            actual_duration=Fraction(4),
            pickup_shift=Fraction(0),
            numerator=4,
            denominator=4,
            implicit=False,
        )]
        hits = [Hit(
            onset=Fraction(0),
            duration=Fraction(1),
            measure_index=0,
            measure_number="1",
            offset_in_measure=Fraction(0),
            sources=[],
        )]
        upload = UploadFile(file=BytesIO(b"%PDF-1.4\n"), filename="berg.pdf")

        with patch.object(app_module, "render_pdf_pages", return_value=[]), \
             patch.object(app_module, "pdf_to_omr", return_value=AudiverisOmrResult(
                 path=Path("fake.omr"), returncode=0, salvaged_after_nonzero=False,
                 command_args=("-batch", "-step", "PAGE", "-save"), log_excerpt="")), \
             patch.object(app_module, "build_measure_framework_from_omr", return_value=(measures, [], {"source": "omr-only-explicit-meter-framework"})), \
             patch.object(app_module, "build_hits_from_canonical_score", return_value=(hits, [], {"canonical_hit_count": 1})), \
             patch.object(app_module, "omr_to_annotations", return_value=(None, "")):
            response = asyncio.run(app_module.analyze_upload(upload, "auto"))

        payload = json.loads(response.body.decode("utf-8"))
        self.assertIsNone(payload["symbolic_source"])
        self.assertEqual(payload["symbolic_hit_count"], 0)
        self.assertEqual(payload["analysis_hit_source"], "canonical-omr-score-time-step-page-save-strict")
        self.assertFalse(payload["silent_event_fallback_enabled"])
        self.assertEqual(payload["audiveris_core_meta"]["driver"], "step-page-save")
        self.assertFalse(payload["audiveris_core_meta"]["salvaged_after_nonzero"])

    def test_removed_export_dependent_pdf_paths_are_absent(self):
        root = Path(__file__).resolve().parents[1]
        app = (root / "app.py").read_text(encoding="utf-8")
        canonical = (root / "tone_metric" / "canonical_score.py").read_text(encoding="utf-8")
        omr = (root / "tone_metric" / "omr.py").read_text(encoding="utf-8")
        self.assertNotIn("pdf_to_musicxml", app)
        self.assertNotIn("reconcile_measure_framework_from_omr", canonical)
        self.assertNotIn("def pdf_to_musicxml", omr)
        self.assertIn("pdf_to_omr", app)
        self.assertIn("build_measure_framework_from_omr", app)
        self.assertNotIn('"-transcribe", "-save"', omr)
        self.assertNotIn('"-export"', omr)


if __name__ == "__main__":
    unittest.main()
