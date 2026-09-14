from __future__ import annotations

import ast
import json
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

from tone_metric.musicxml import parse_musicxml
from tone_metric.engine import analyze


def _write_xml(text: str) -> Path:
    f = tempfile.NamedTemporaryFile('w', suffix='.musicxml', delete=False, encoding='utf-8')
    f.write(text)
    f.close()
    return Path(f.name)


def synthetic() -> None:
    # Full 3/4 opening + sectional heavy-light barline (no repeat): one structural
    # pre-entry L1 is expected, while the actual opening attack keeps its normal L1.
    xml = '''<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes><divisions>4</divisions><time><beats>3</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><type>quarter</type></note>
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><type>quarter</type></note>
      <note><pitch><step>E</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><type>quarter</type></note>
      <barline location="right"><bar-style>heavy-light</bar-style></barline>
    </measure>
    <measure number="2"><note><pitch><step>F</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><type>quarter</type></note><forward><duration>8</duration></forward></measure>
  </part>
</score-partwise>'''
    p = _write_xml(xml)
    hits, measures, warnings = parse_musicxml(p)
    assert measures[0].opening_anacrusis is True
    assert measures[0].pickup_shift == 0
    r = analyze(hits, measures)
    seg = r['segments'][0]
    pre = [x for x in seg['structural_points'] if x.get('opening_anacrusis_pre_entry')]
    assert len(pre) == 1, pre
    assert pre[0]['time_quarter'] == '-1' and pre[0]['levels'] == [1]
    first = seg['events'][0]
    assert first['onset_quarter'] == '0'
    assert 1 in first['tone_metric_levels']

    # Same sectional glyph with an explicit repeat is not the dissertation cue.
    xml_repeat = xml.replace(
        '<barline location="right"><bar-style>heavy-light</bar-style></barline>',
        '<barline location="right"><bar-style>heavy-light</bar-style><repeat direction="forward"/></barline>'
    )
    p2 = _write_xml(xml_repeat)
    _, m2, _ = parse_musicxml(p2)
    assert m2[0].opening_anacrusis is False

    # Ordinary complete first measure: no extra pre-entry structure.
    xml_plain = xml.replace('<barline location="right"><bar-style>heavy-light</bar-style></barline>', '')
    p3 = _write_xml(xml_plain)
    h3, m3, _ = parse_musicxml(p3)
    assert m3[0].opening_anacrusis is False
    r3 = analyze(h3, m3)
    assert not any(x.get('opening_anacrusis_pre_entry') for x in r3['segments'][0]['structural_points'])

    # Ordinary incomplete pickup keeps the pre-existing duration-derived pickup path;
    # it must not also receive the sectional pre-entry rule.
    xml_short = '''<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0"><part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list><part id="P1">
<measure number="1" implicit="yes"><attributes><divisions>4</divisions><time><beats>3</beats><beat-type>4</beat-type></time></attributes><note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><type>quarter</type></note></measure>
<measure number="2"><note><pitch><step>D</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><type>quarter</type></note><forward><duration>8</duration></forward></measure>
</part></score-partwise>'''
    p4 = _write_xml(xml_short)
    h4, m4, _ = parse_musicxml(p4)
    assert m4[0].pickup_shift == Fraction(2)
    assert m4[0].opening_anacrusis is False
    r4 = analyze(h4, m4)
    assert not any(x.get('opening_anacrusis_pre_entry') for x in r4['segments'][0]['structural_points'])
    print('SYNTHETIC OPENING-ANACRUSIS RULE: PASS')


def berg(path: str) -> None:
    d = json.load(open(path, encoding='utf-8'))
    events = [e for s in d.get('segments', []) for e in s.get('events', [])]
    assert len(d.get('measures', [])) == 175
    assert d.get('symbolic_hit_count') == 859
    assert d.get('canonical_hit_count') == 1724
    assert len(events) == 1724 and all(e.get('tone_metric_levels') for e in events)
    assert d.get('max_level') == 16
    m0 = d['measures'][0]
    assert m0.get('number') == '1'
    assert m0.get('pickup_shift_quarter') == '0'
    assert m0.get('opening_anacrusis') is True
    seg0 = d['segments'][0]
    assert seg0.get('opening_anacrusis') is True
    assert seg0.get('opening_anacrusis_pre_entry_quarter') == '-1'
    pre = [p for p in seg0.get('structural_points', []) if p.get('opening_anacrusis_pre_entry')]
    assert len(pre) == 1 and pre[0].get('time_quarter') == '-1' and pre[0].get('levels') == [1]
    first = events[0]
    assert first.get('measure_index') == 0 and first.get('onset_quarter') == '0'
    assert first.get('tone_metric_levels') == [1, 2]

    po = d.get('physical_overlay') or {}
    assert po.get('available') is True
    matching = po.get('matching') or {}
    assert matching.get('structural_parenthetical_labels_missing') == 0
    reasons = matching.get('structural_reasons') or {}
    assert reasons.get('opening-anacrusis-pre-entry-metric-position') == 1
    assert matching.get('structural_parenthetical_positions') == 257
    assert matching.get('structural_parenthetical_labels_expected') == 656
    assert matching.get('structural_parenthetical_labels_mapped') == 656
    # The pre-entry articulation is structural setup only; the already validated
    # wave/pivot/tree profiles must not be rephased or inflated.
    assert matching.get('wave_profile_points_mapped') == 1980
    assert matching.get('pivot_profile_mapped') == 143
    assert matching.get('tree_event_nodes_mapped') == 1724
    assert matching.get('tree_branches_mapped') == 1723

    pages = {int(p.get('page_index')): p for p in po.get('pages', [])}
    page0 = pages[0]
    preanchors = [a for a in page0.get('structural_anchors', []) if a.get('structural_reason') == 'opening-anacrusis-pre-entry-metric-position']
    assert len(preanchors) == 1
    first_attack = min(
        (a for a in page0.get('layer_anchors', []) if int(a.get('measure_index', -1)) == 0),
        key=lambda a: float(a.get('recovered_x_abs', 1e18)),
    )
    assert float(preanchors[0]['recovered_x_abs']) < float(first_attack['recovered_x_abs'])
    print(json.dumps({
        'measures': len(d['measures']),
        'events': len(events),
        'opening_pre_entry_time': pre[0]['time_quarter'],
        'first_event_levels': first['tone_metric_levels'],
        'parenthetical_positions': matching.get('structural_parenthetical_positions'),
        'parenthetical_labels': matching.get('structural_parenthetical_labels_mapped'),
        'wave_points': matching.get('wave_profile_points_mapped'),
        'pivots': matching.get('pivot_profile_mapped'),
        'tree_nodes': matching.get('tree_event_nodes_mapped'),
        'tree_branches': matching.get('tree_branches_mapped'),
    }, indent=2))
    print('EXACT BERG OPENING + FULL SCORE: PASS')


def bwv(path: str) -> None:
    d = json.load(open(path, encoding='utf-8'))
    events = [e for s in d.get('segments', []) for e in s.get('events', [])]
    po = d.get('physical_overlay') or {}; m = po.get('matching') or {}
    assert len(d.get('measures', [])) == 92
    assert d.get('symbolic_hit_count') == 549
    assert d.get('canonical_hit_count') == 733
    assert len(events) == 733 and all(e.get('tone_metric_levels') for e in events)
    assert d.get('levels_enabled') == list(range(1, 10)) and d.get('max_level') == 9
    assert d['measures'][0].get('opening_anacrusis') is False
    assert not any(p.get('opening_anacrusis_pre_entry') for s in d.get('segments', []) for p in s.get('structural_points', []))
    assert po.get('available') is True
    assert m.get('wave_profile_points_mapped') == 735
    assert m.get('pivot_profile_mapped') == 27
    assert m.get('tree_event_nodes_mapped') == 733
    assert m.get('tree_branches_mapped') == 732
    print('EXACT BWV 661 UNCHANGED: PASS')


def clean() -> None:
    forbidden = [
        'validate_release.py',
        'tone_metric/omr_exact.py',
        'tone_metric/omr_visual_rhythm.py',
        'tone_metric/layer_registration.py',
        'tone_metric/render.py',
    ]
    for p in forbidden:
        assert not Path(p).exists(), p
    # One authoritative top-level definition per name in every runtime module.
    for p in [Path('app.py'), *sorted(Path('tone_metric').glob('*.py'))]:
        tree = ast.parse(p.read_text(encoding='utf-8'))
        names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        dup = sorted({x for x in names if names.count(x) > 1})
        assert not dup, f'{p}: duplicate top-level definitions {dup}'
    app = Path('app.py').read_text(encoding='utf-8')
    assert app.count('@app.post("/api/analyze")') == 1
    canon = Path('tone_metric/canonical_score.py').read_text(encoding='utf-8')
    assert canon.count('def build_hits_from_canonical_score') == 1
    eng = Path('tone_metric/engine.py').read_text(encoding='utf-8')
    assert eng.count('def analyze_segment') == 1
    reg = Path('tone_metric/score_registration.py').read_text(encoding='utf-8')
    assert reg.count('def build_layer_anchors_from_canonical_score') == 1
    # Runtime logic must stay score-agnostic.
    runtime = '\n'.join(Path(p).read_text(encoding='utf-8') for p in [
        'tone_metric/models.py','tone_metric/musicxml.py','tone_metric/engine.py','tone_metric/score_registration.py'
    ])
    for token in ('IMSLP02556', 'Berg Op.1', 'berg.pdf', 'source_filename =='):
        assert token not in runtime, token
    print('CLEAN OVERRIDE/LEGACY AUDIT: PASS')


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'synthetic': synthetic()
    elif mode == 'berg': berg(sys.argv[2])
    elif mode == 'bwv': bwv(sys.argv[2])
    elif mode == 'clean': clean()
    else: raise SystemExit(mode)
