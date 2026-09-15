from __future__ import annotations
import json, sys, time
from pathlib import Path
from tone_metric.omr import pdf_to_musicxml, pdf_to_annotations
from tone_metric.pdfview import render_pdf_pages
from tone_metric.musicxml import parse_musicxml, extract_visual_groups
from tone_metric.canonical_score import build_hits_from_canonical_score
from tone_metric.engine import analyze
from tone_metric.physical import build_normalized_overlay

pdf = Path(sys.argv[1]).resolve()
root = Path('profile-out').resolve()
root.mkdir(exist_ok=True)
T0 = time.perf_counter()
timings = {}

def stage(name, fn):
    t=time.perf_counter(); out=fn(); timings[name]=time.perf_counter()-t; print(f'{name}: {timings[name]:.3f}s', flush=True); return out

pages = stage('render_pdf_pages', lambda: render_pdf_pages(pdf, root/'pages'))
symbolic, omr = stage('audiveris_transcribe_export_omr', lambda: pdf_to_musicxml(pdf, root/'omr'))
annotation, annotation_warning = stage('audiveris_annotation_only', lambda: pdf_to_annotations(pdf, root/'annotations'))
hits, measures, parse_warnings = stage('parse_musicxml', lambda: parse_musicxml(symbolic, initial_meter_override=None))
recovered, canonical_warnings, canonical_meta = stage('canonical_recovery', lambda: build_hits_from_canonical_score(omr, measures, symbolic_hits=hits))
result = stage('tone_metric_analyze', lambda: analyze(recovered, measures))
if annotation is not None:
    visual_groups, layout_known = stage('extract_visual_groups', lambda: extract_visual_groups(symbolic, initial_meter_override=None))
    overlay = stage('physical_overlay', lambda: build_normalized_overlay(annotation, visual_groups, layout_known, result, root/'physical', omr_path=omr))
else:
    overlay={'available':False}
timings['total_profiled']=time.perf_counter()-T0
summary={
 'timings_seconds':timings,
 'symbolic_hits':len(hits),
 'canonical_hits':len(recovered),
 'measures':len(measures),
 'overlay_available':overlay.get('available'),
 'annotation_warning':annotation_warning,
 'canonical_meta':canonical_meta,
}
print(json.dumps(summary, indent=2), flush=True)
Path('profile-summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
