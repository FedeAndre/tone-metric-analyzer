from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from tone_metric.pdfview import render_pdf_pages
from tone_metric.omr import pdf_to_musicxml, pdf_to_annotations
from tone_metric.musicxml import parse_musicxml, extract_visual_groups
from tone_metric.canonical_score import build_hits_from_canonical_score
from tone_metric.engine import analyze
from tone_metric.omr_project import read_omr_slots, omr_slots_debug_rows
from tone_metric.physical import build_normalized_overlay


def timed(label, fn, rows):
    t0 = time.perf_counter()
    value = fn()
    dt = time.perf_counter() - t0
    rows.append((label, dt))
    print(f"PROFILE {label}: {dt:.3f}s", flush=True)
    return value


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: profile_omr_pipeline.py INPUT_PDF OUTPUT_DIR")
    pdf = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    total0 = time.perf_counter()

    pages = timed("render_pdf_pages", lambda: render_pdf_pages(pdf, out / "pages"), rows)
    symbolic, omr = timed("audiveris_transcribe_export", lambda: pdf_to_musicxml(pdf, out / "omr"), rows)
    annotations, annotation_warning = timed("audiveris_annotation_only", lambda: pdf_to_annotations(pdf, out / "annotations"), rows)
    hits, measures, parse_warnings = timed("parse_musicxml", lambda: parse_musicxml(symbolic, initial_meter_override=None), rows)
    recovered_hits, canonical_warnings, canonical_meta = timed(
        "canonical_score_recovery",
        lambda: build_hits_from_canonical_score(omr, measures, symbolic_hits=hits),
        rows,
    )
    result = timed("tone_metric_analyze", lambda: analyze(recovered_hits, measures), rows)

    def slot_dump():
        slots, meta = read_omr_slots(omr)
        (out / "omr-slots-profile.json").write_text(
            json.dumps({"meta": meta, "slots": omr_slots_debug_rows(slots)}, indent=2),
            encoding="utf-8",
        )
        return len(slots)

    slot_count = timed("read_omr_slots_and_write_debug", slot_dump, rows)

    overlay = None
    if annotations is not None:
        visual_groups, layout_known = timed(
            "extract_visual_groups",
            lambda: extract_visual_groups(symbolic, initial_meter_override=None),
            rows,
        )
        overlay = timed(
            "build_physical_overlay",
            lambda: build_normalized_overlay(
                annotations,
                visual_groups,
                layout_known,
                result,
                out / "physical",
                omr_path=omr,
            ),
            rows,
        )

    def serialize_debug():
        payload = {
            "measures": result.get("measures", []),
            "segments": result.get("segments", []),
            "physical_overlay": overlay,
            "canonical_score_meta": canonical_meta,
            "parse_warnings": list(parse_warnings) + list(canonical_warnings),
        }
        text = json.dumps(payload, ensure_ascii=False)
        (out / "profile-result.json").write_text(text, encoding="utf-8")
        return len(text)

    json_bytes = timed("serialize_and_write_result", serialize_debug, rows)
    total = time.perf_counter() - total0
    print(f"PROFILE TOTAL: {total:.3f}s", flush=True)

    summary = {
        "pdf": pdf.name,
        "pages": len(pages),
        "symbolic_hits": len(hits),
        "canonical_hits": len(recovered_hits),
        "measures": len(measures),
        "max_level": max((max(seg.get("levels_enabled", [0])) for seg in result.get("segments", []) if seg.get("levels_enabled")), default=0),
        "annotation_available": annotations is not None,
        "annotation_warning": annotation_warning,
        "physical_overlay_available": bool((overlay or {}).get("available")),
        "omr_slot_count": slot_count,
        "serialized_chars": json_bytes,
        "canonical_meta": canonical_meta,
        "timings_seconds": {k: round(v, 3) for k, v in rows},
        "total_seconds": round(total, 3),
    }
    (out / "profile-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
