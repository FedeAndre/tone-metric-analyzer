from __future__ import annotations
import ast
import hashlib
import pathlib
import sys
from fractions import Fraction
from tempfile import TemporaryDirectory
from zipfile import ZipFile


def targeted_checks():
    from tone_metric.models import MeasureInfo, Hit
    from tone_metric.canonical_score import _reconcile_measure_framework_with_omr, CanonicalFrameworkError

    def m(i, n, num=3, den=4):
        full = Fraction(num * 4, den)
        return MeasureInfo(i, str(n), Fraction(i * 3), full, full, Fraction(0), num, den, False)

    with TemporaryDirectory() as td:
        omr = pathlib.Path(td) / 'x.omr'
        xml = '<sheet><system>' + ''.join(f'<stack id="{i}" left="0" right="10"/>' for i in range(1, 5)) + '</system></sheet>'
        with ZipFile(omr, 'w') as zf:
            zf.writestr('sheet#1/sheet#1.xml', xml)
        measures = [m(0, 1), m(1, 3), m(2, 4)]
        h = Hit(Fraction(7), Fraction(1), 1, '3', Fraction(1), [])
        meta, _ = _reconcile_measure_framework_with_omr(omr, measures, [h])
        assert [x.number for x in measures] == ['1', '2', '3', '4']
        assert [x.index for x in measures] == [0, 1, 2, 3]
        assert [x.start for x in measures] == [Fraction(0), Fraction(3), Fraction(6), Fraction(9)]
        assert h.measure_index == 2 and h.onset == Fraction(7)
        assert meta['framework_reconciled'] and meta['synthesized_measure_numbers'] == ['2']
        assert meta['framework_measure_count'] == 4

        bad = [m(0, 1, 3, 4), m(1, 3, 4, 4), m(2, 4, 4, 4)]
        try:
            _reconcile_measure_framework_with_omr(omr, bad, [])
        except CanonicalFrameworkError as exc:
            assert 'conflicting meters' in str(exc)
        else:
            raise AssertionError('conflicting-meter gap must fail closed')


def refresh_manifest():
    p = pathlib.Path('v0152_sha256.txt')
    out = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        _old, path = line.split(None, 1)
        out.append(f'{hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()}  {path}')
    p.write_text('\n'.join(out) + '\n')


def clean_audit():
    for p in [pathlib.Path('app.py'), *sorted(pathlib.Path('tone_metric').glob('*.py'))]:
        tree = ast.parse(p.read_text(), filename=str(p))
        names = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names[node.name] = names.get(node.name, 0) + 1
        dup = [n for n, count in names.items() if count > 1]
        assert not dup, f'{p}: duplicate top-level definitions {dup}'

    app = pathlib.Path('app.py').read_text()
    assert app.count('@app.post("/api/analyze")') == 1
    assert 'symbolic MusicXML hits were used' not in app
    canonical = pathlib.Path('tone_metric/canonical_score.py').read_text()
    assert canonical.count('def build_hits_from_canonical_score(') == 1
    assert canonical.count('def _reconcile_measure_framework_with_omr(') == 1
    assert canonical.count('class CanonicalFrameworkError') == 1
    omr = pathlib.Path('tone_metric/omr.py').read_text()
    assert omr.count('def pdf_to_musicxml(') == 1
    assert '"-save"' not in omr
    forbidden = [
        'validate_release.py', 'tone_metric/omr_exact.py', 'tone_metric/omr_visual_rhythm.py',
        'tone_metric/layer_registration.py', 'tone_metric/render.py',
    ]
    assert not [f for f in forbidden if pathlib.Path(f).exists()]
    assert not list(pathlib.Path('.').rglob('*.pyc'))
    assert not list(pathlib.Path('.').rglob('__pycache__'))


def check_berg(path):
    import json
    d = json.load(open(path))
    meta = d.get('canonical_score_meta') or {}
    events = [e for s in d.get('segments', []) for e in s.get('events', [])]
    pages = (d.get('physical_overlay') or {}).get('pages') or []
    by_page = {int(p.get('page_index')): p for p in pages}
    missing = [e for e in events if not e.get('tone_metric_levels')]
    summary = {
        'measures': len(d.get('measures', [])), 'symbolic': d.get('symbolic_hit_count'),
        'canonical': d.get('canonical_hit_count'), 'events': len(events),
        'missing_levels': len(missing), 'max_level': d.get('max_level'),
        'musicxml_measures': meta.get('musicxml_measure_count'),
        'omr_stacks': meta.get('omr_measure_stack_count'),
        'framework_measures': meta.get('framework_measure_count'),
        'synthesized': meta.get('synthesized_measure_count'),
        'page8_layers': len(by_page.get(7, {}).get('layer_anchors', [])),
        'page9_layers': len(by_page.get(8, {}).get('layer_anchors', [])),
        'page9_trees': len(by_page.get(8, {}).get('tree_nodes', [])),
    }
    print(summary)
    assert d.get('analysis_hit_source') == 'canonical-score-time'
    assert len(d.get('measures', [])) == 175
    assert d['measures'][0]['number'] == '1' and d['measures'][-1]['number'] == '175'
    assert meta.get('musicxml_measure_count') == 136
    assert meta.get('omr_measure_stack_count') == 175
    assert meta.get('framework_measure_count') == 175
    assert meta.get('framework_reconciled') is True
    assert meta.get('synthesized_measure_count') == 39
    assert meta.get('measure_count') == 175
    assert not any('more measure stacks than the canonical MusicXML framework' in str(w) for w in meta.get('warnings', []))
    assert len(events) > 1329
    assert not missing
    assert (d.get('physical_overlay') or {}).get('available') is True
    assert len(by_page.get(7, {}).get('layer_anchors', [])) > 16
    assert len(by_page.get(8, {}).get('layer_anchors', [])) > 0
    assert len(by_page.get(8, {}).get('tree_nodes', [])) > 0
    assert (d.get('physical_overlay') or {}).get('matching', {}).get('structural_parenthetical_labels_missing') == 0
    print('EXACT FULL-SCORE BERG: PASS')


def check_bwv(path):
    import json
    d = json.load(open(path))
    events = [e for s in d.get('segments', []) for e in s.get('events', [])]
    po = d.get('physical_overlay') or {}
    matching = po.get('matching') or {}
    meta = d.get('canonical_score_meta') or {}
    print({'measures': len(d.get('measures', [])), 'symbolic': d.get('symbolic_hit_count'), 'canonical': d.get('canonical_hit_count'), 'events': len(events), 'max': d.get('max_level')})
    assert d.get('symbolic_hit_count') == 549
    assert d.get('canonical_hit_count') == 733
    assert len(events) == 733 and all(e.get('tone_metric_levels') for e in events)
    assert d.get('levels_enabled') == list(range(1, 10)) and d.get('max_level') == 9
    assert len(d.get('measures', [])) == 92
    assert meta.get('omr_measure_stack_count') == 92
    assert meta.get('framework_reconciled') is False
    assert po.get('available') is True
    assert matching.get('wave_profile_points_mapped') == 735
    assert matching.get('pivot_profile_mapped') == 27
    assert matching.get('tree_event_nodes_mapped') == 733
    assert matching.get('tree_branches_mapped') == 732
    print('EXACT BWV 661: PASS')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'prepare':
        targeted_checks()
        refresh_manifest()
        clean_audit()
        print('TARGETED CHECKS + CLEAN AUDIT: PASS')
    elif cmd == 'berg':
        check_berg(sys.argv[2])
    elif cmd == 'bwv':
        check_bwv(sys.argv[2])
    else:
        raise SystemExit(f'unknown command: {cmd}')
