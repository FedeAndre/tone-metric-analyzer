import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from core import extract_hit_strikes


def make_omr(xml: str) -> Path:
    root = Path(tempfile.mkdtemp())
    path = root / 'score.omr'
    with ZipFile(path, 'w') as zf:
        zf.writestr('sheet#1/sheet#1.xml', xml)
    return path


def head(head_id: str, x: float, y: float = 180) -> str:
    return (
        f'<head id="{head_id}">'
        f'<bounds x="{x - 10}" y="{y - 10}" w="20" h="20"/>'
        f'</head>'
    )


def chord(chord_id: str, staff: str = '10') -> str:
    return f'<head-chord id="{chord_id}" staff="{staff}"/>'


def relation(source: str, target: str, body: str) -> str:
    return f'<relation source="{source}" target="{target}">{body}</relation>'


def score(objects: str, relations: str, stacks: str | None = None) -> str:
    if stacks is None:
        stacks = '<stack id="1" left="100" right="500"/>'
    return f'''<sheet>
      <scale><interline main="20"/></scale>
      <picture width="1000" height="800"/>
      <page><system id="1">
        {stacks}
        <staff id="10">
          <line><point y="100"/></line>
          <line><point y="300"/></line>
        </staff>
      </system></page>
      {objects}
      <relations>{relations}</relations>
    </sheet>'''


class GeometryHitTests(unittest.TestCase):
    def test_one_visible_chord_is_one_hit(self):
        xml = score(
            chord('c1') + head('h1', 200),
            relation('c1', 'h1', '<containment/>'),
        )
        count, strikes = extract_hit_strikes(make_omr(xml))
        self.assertEqual((count, len(strikes)), (1, 1))
        self.assertEqual(strikes[0].x, 200)

    def test_simultaneous_aligned_chords_merge_globally(self):
        objects = chord('c1') + head('h1', 200) + chord('c2') + head('h2', 211)
        relations = (
            relation('c1', 'h1', '<containment/>')
            + relation('c2', 'h2', '<containment/>')
        )
        count, strikes = extract_hit_strikes(make_omr(score(objects, relations)))
        self.assertEqual((count, len(strikes)), (1, 1))

    def test_two_separate_visible_attacks_never_merge(self):
        # Exact class of the Buxtehude m.6 failure: the two attacks are 28px apart.
        objects = chord('c1') + head('h1', 234) + chord('c2') + head('h2', 262)
        relations = (
            relation('c1', 'h1', '<containment/>')
            + relation('c2', 'h2', '<containment/>')
        )
        count, strikes = extract_hit_strikes(make_omr(score(objects, relations)))
        self.assertEqual((count, len(strikes)), (2, 2))
        self.assertEqual([s.x for s in strikes], [234, 262])

    def test_bogus_rhythmic_slot_data_cannot_merge_visible_attacks(self):
        stacks = '''<stack id="1" left="100" right="500">
          <slot id="1" x-offset="100" time-offset="0"/>
        </stack>
        <part><measure><voice><slots>
          <entry><key>1</key><value chord="c1" status="BEGIN"/></entry>
          <entry><key>1</key><value chord="c2" status="BEGIN"/></entry>
        </slots></voice></measure></part>'''
        objects = chord('c1') + head('h1', 234) + chord('c2') + head('h2', 262)
        relations = (
            relation('c1', 'h1', '<containment/>')
            + relation('c2', 'h2', '<containment/>')
        )
        count, strikes = extract_hit_strikes(
            make_omr(score(objects, relations, stacks))
        )
        self.assertEqual((count, len(strikes)), (2, 2))

    def test_tied_continuation_is_not_new_hit(self):
        objects = chord('c1') + head('h1', 200) + '<slur id="s1" tie="true"/>'
        relations = (
            relation('c1', 'h1', '<containment/>')
            + relation('s1', 'h1', '<slur-head side="RIGHT"/>')
        )
        with self.assertRaisesRegex(ValueError, 'No sounding note attacks'):
            extract_hit_strikes(make_omr(score(objects, relations)))

    def test_mixed_tied_and_new_chord_counts_once(self):
        objects = (
            chord('c1') + head('h1', 200) + head('h2', 205)
            + '<slur id="s1" tie="true"/>'
        )
        relations = (
            relation('c1', 'h1', '<containment/>')
            + relation('c1', 'h2', '<containment/>')
            + relation('s1', 'h1', '<slur-head side="RIGHT"/>')
        )
        count, strikes = extract_hit_strikes(make_omr(score(objects, relations)))
        self.assertEqual((count, len(strikes)), (1, 1))

    def test_measure_partition_prevents_cross_barline_merge(self):
        stacks = (
            '<stack id="1" left="100" right="300"/>'
            '<stack id="2" left="300" right="500"/>'
        )
        objects = chord('c1') + head('h1', 296) + chord('c2') + head('h2', 304)
        relations = (
            relation('c1', 'h1', '<containment/>')
            + relation('c2', 'h2', '<containment/>')
        )
        count, strikes = extract_hit_strikes(
            make_omr(score(objects, relations, stacks))
        )
        self.assertEqual((count, len(strikes)), (2, 2))

    def test_dense_but_distinct_attacks_remain_distinct(self):
        xs = [180, 205, 230, 255]
        objects = ''.join(
            chord(f'c{i}') + head(f'h{i}', x)
            for i, x in enumerate(xs)
        )
        relations = ''.join(
            relation(f'c{i}', f'h{i}', '<containment/>')
            for i in range(len(xs))
        )
        count, strikes = extract_hit_strikes(make_omr(score(objects, relations)))
        self.assertEqual((count, len(strikes)), (4, 4))
        self.assertEqual([s.x for s in strikes], xs)


if __name__ == '__main__':
    unittest.main()
