from __future__ import annotations

import ast
import tempfile
from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile

import recursive_levels_clean_v9 as v9


def make_measures(count: int, meter=(4, 4)):
    length = Fraction(meter[0] * 4, meter[1])
    return [
        v9.Measure(i, str(i + 1), i * length, (i + 1) * length, Fraction(0), meter)
        for i in range(count)
    ]


def times(points):
    return [Fraction(p["time_quarter"]) for p in points]


def isolation_audit():
    source = Path("recursive_levels_clean_v9.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    forbidden = (
        "level1_clean_", "level1_level2_", "recursive_levels_clean_v7",
        "recursive_levels_clean_v8", "tone_metric", "dissertation_", "app",
    )
    offenders = [name for name in imported if any(name.startswith(p) for p in forbidden)]
    assert offenders == [], offenders
    assert "previous_version_imports\": False" in source
    assert "canonical-omr-notation-columns" in source


def analytical_regression():
    assert v9.sequence(2, 34)[:7] == [1, 2, 3, 5, 9, 17, 33]
    assert v9.sequence(3, 29)[:5] == [1, 2, 4, 10, 28]
    measures = make_measures(3)
    attacks = [v9.Attack(Fraction(i)) for i in range(12)]
    result = v9.analyze_recursive_levels(measures, attacks)
    assert times(result["level1_points"]) == [Fraction(x) for x in (0, 1, 2, 4, 8)]
    assert times(result["level2_points"]) == [Fraction(x) for x in (2, 3, 4, 5, 6, 8, 9, 10)]
    assert times(result["level3_points"]) == [Fraction(x) for x in (6, 7, 8, 10, 11)]
    assert result["highest_level"] == 3
    covered = set()
    for pts in result["levels"].values():
        covered.update(times(pts))
    assert covered == {Fraction(i) for i in range(12)}


def write_omr(path: Path, body: str):
    xml = f'''<sheet><picture width="800" height="300"/><page><system>{body}</system></page></sheet>'''.encode()
    with ZipFile(path, "w") as zf:
        zf.writestr("sheet#1/sheet#1.xml", xml)


def canonical_notation_regression():
    measure = make_measures(1)[0]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # Dotted quarter + following attack: dot extends duration, never creates an attack.
        dot_body = '''
        <stack id="1" left="100" right="500"><slot id="1" time-offset="0"/><slot id="2" time-offset="3/8"/></stack>
        <part><measure id="1"><head-chords>c1 c2</head-chords><voice id="1"><slots>
          <entry><key>1</key><value chord="c1" status="BEGIN"/></entry><entry><key>2</key><value chord="c2" status="BEGIN"/></entry>
        </slots></voice></measure></part>
        <head-chord staff="1" id="c1"><bounds x="140" y="100" w="20" h="60"/></head-chord>
        <head shape="NOTEHEAD_BLACK" staff="1" id="h1"><bounds x="140" y="140" w="16" h="14"/></head>
        <augmentation-dot id="d1"><bounds x="168" y="140" w="5" h="5"/></augmentation-dot>
        <head-chord staff="1" id="c2"><bounds x="300" y="100" w="20" h="60"/></head-chord>
        <head shape="NOTEHEAD_BLACK" staff="1" id="h2"><bounds x="300" y="140" w="16" h="14"/></head>
        <relation source="c1" target="h1"><containment/></relation><relation source="h1" target="d1"><augmentation/></relation><relation source="c2" target="h2"><containment/></relation>
        '''
        dot = td / "dot.omr"; write_omr(dot, dot_body)
        cols, meta = v9.recover_canonical_columns(dot, [measure])
        attacks = [c for c in cols if c.attack]
        assert [c.onset_quarter for c in attacks] == [Fraction(0), Fraction(3, 2)]
        assert attacks[0].duration_quarter == Fraction(3, 2)
        assert meta["augmentation_dot_count"] >= 1

        # Rest occupies metric time but is not an attack.
        rest_body = '''
        <stack id="1" left="100" right="500"><slot id="1" time-offset="0"/><slot id="2" time-offset="1/4"/><slot id="3" time-offset="1/2"/></stack>
        <part><measure id="1"><head-chords>c1 c2</head-chords><rest-chords>r1</rest-chords><voice id="1"><slots>
          <entry><key>1</key><value chord="c1" status="BEGIN"/></entry><entry><key>2</key><value chord="r1" status="BEGIN"/></entry><entry><key>3</key><value chord="c2" status="BEGIN"/></entry>
        </slots></voice></measure></part>
        <head-chord id="c1"><bounds x="140" y="100" w="20" h="60"/></head-chord><head shape="NOTEHEAD_BLACK" id="h1"><bounds x="140" y="140" w="16" h="14"/></head>
        <rest-chord id="r1"><bounds x="220" y="120" w="20" h="30"/></rest-chord><rest shape="QUARTER_REST" id="rr1"><bounds x="220" y="120" w="20" h="30"/></rest>
        <head-chord id="c2"><bounds x="320" y="100" w="20" h="60"/></head-chord><head shape="NOTEHEAD_BLACK" id="h2"><bounds x="320" y="140" w="16" h="14"/></head>
        <relation source="c1" target="h1"><containment/></relation><relation source="r1" target="rr1"><containment/></relation><relation source="c2" target="h2"><containment/></relation>
        '''
        rest = td / "rest.omr"; write_omr(rest, rest_body)
        cols, _ = v9.recover_canonical_columns(rest, [measure])
        assert [c.onset_quarter for c in cols] == [Fraction(0), Fraction(1), Fraction(2)]
        assert [c.attack for c in cols] == [True, False, True]

        # Tie right endpoint is a continuation, not a new attack.
        tie_body = '''
        <stack id="1" left="100" right="500"><slot id="1" time-offset="0"/><slot id="2" time-offset="1/4"/></stack>
        <part><measure id="1"><head-chords>c1 c2</head-chords><voice id="1"><slots>
          <entry><key>1</key><value chord="c1" status="BEGIN"/></entry><entry><key>2</key><value chord="c2" status="BEGIN"/></entry>
        </slots></voice></measure></part>
        <head-chord id="c1"><bounds x="140" y="100" w="20" h="60"/></head-chord><head shape="NOTEHEAD_BLACK" id="h1"><bounds x="140" y="140" w="16" h="14"/></head>
        <head-chord id="c2"><bounds x="300" y="100" w="20" h="60"/></head-chord><head shape="NOTEHEAD_BLACK" id="h2"><bounds x="300" y="140" w="16" h="14"/></head>
        <slur tie="true" id="t1"><bounds x="150" y="80" w="160" h="30"/></slur>
        <relation source="c1" target="h1"><containment/></relation><relation source="c2" target="h2"><containment/></relation>
        <relation source="t1" target="h1"><slur-head side="LEFT"/></relation><relation source="t1" target="h2"><slur-head side="RIGHT"/></relation>
        '''
        tie = td / "tie.omr"; write_omr(tie, tie_body)
        cols, _ = v9.recover_canonical_columns(tie, [measure])
        assert [c.attack for c in cols] == [True, False]
        assert cols[1].tie_continuation_only
        hits = v9.attacks_from_canonical(cols, [measure])
        assert [h.onset for h in hits] == [Fraction(0)]


def main():
    isolation_audit()
    analytical_regression()
    canonical_notation_regression()
    print("recursive-levels-clean-v9-validation: PASS")


if __name__ == "__main__":
    main()
