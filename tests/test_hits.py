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


def head(head_id: str, x: float, y: float = 180, pitch: int | None = None) -> str:
    p = '' if pitch is None else f' pitch="{pitch}"'
    return f'<head id="{head_id}"{p}><bounds x="{x-10}" y="{y-10}" w="20" h="20"/></head>'


def chord(chord_id: str, staff: str = '10') -> str:
    return f'<head-chord id="{chord_id}" staff="{staff}"/>'


def stem(stem_id: str, x: float, y: float, h: float) -> str:
    return f'<stem id="{stem_id}"><bounds x="{x-2}" y="{y}" w="4" h="{h}"/></stem>'


def relation(source: str, target: str, body: str) -> str:
    return f'<relation source="{source}" target="{target}">{body}</relation>'


def entry(chord_id: str, slot: str) -> str:
    return f'<entry><key>{slot}</key><value chord="{chord_id}" status="BEGIN"/></entry>'


def score(objects: str, relations: str, stacks: str | None = None, parts: str = '') -> str:
    if stacks is None:
        stacks = '<stack id="1" left="100" right="500"/>'
    return f'''<sheet>
      <scale><interline main="20"/></scale>
      <picture width="1000" height="800"/>
      <page><system id="1">
        {stacks}
        {parts}
        <staff id="10">
          <line><point y="100"/></line><line><point y="300"/></line>
        </staff>
      </system></page>
      {objects}<relations>{relations}</relations>
    </sheet>'''


def displaced_pair(slot_a=None, slot_b=None):
    objects = (
        chord('c1') + head('h1', 200, 220, 5) + stem('s1', 188, 225, 60)
        + chord('c2') + head('h2', 226, 190, 2) + stem('s2', 238, 120, 65)
    )
    relations = (
        relation('c1','h1','<containment/>') + relation('c1','s1','<chord-stem/>')
        + relation('c2','h2','<containment/>') + relation('c2','s2','<chord-stem/>')
    )
    entries = ''
    if slot_a is not None:
        entries += entry('c1', slot_a)
    if slot_b is not None:
        entries += entry('c2', slot_b)
    parts = f'<part><measure><voice><slots>{entries}</slots></voice></measure></part>' if entries else ''
    return objects, relations, parts


class HitTests(unittest.TestCase):
    def test_visible_chord(self):
        xml = score(chord('c1') + head('h1', 200), relation('c1','h1','<containment/>'))
        count, strikes = extract_hit_strikes(make_omr(xml))
        self.assertEqual((count, len(strikes)), (1, 1))
        self.assertEqual(strikes[0].x, 200)

    def test_aligned_chords_merge(self):
        objects = chord('c1') + head('h1', 200) + chord('c2') + head('h2', 211)
        relations = relation('c1','h1','<containment/>') + relation('c2','h2','<containment/>')
        count, strikes = extract_hit_strikes(make_omr(score(objects, relations)))
        self.assertEqual((count, len(strikes)), (1, 1))

    def test_dense_sequential_stays_separate(self):
        xs = [180, 205, 230, 255]
        objects = ''.join(chord(f'c{i}') + head(f'h{i}', x) for i, x in enumerate(xs))
        relations = ''.join(relation(f'c{i}', f'h{i}', '<containment/>') for i in range(len(xs)))
        count, strikes = extract_hit_strikes(make_omr(score(objects, relations)))
        self.assertEqual((count, len(strikes)), (4, 4))
        self.assertEqual([s.x for s in strikes], xs)

    def test_tied_continuation_removed(self):
        objects = chord('c1') + head('h1', 200) + '<slur id="sl" tie="true"/>'
        relations = relation('c1','h1','<containment/>') + relation('sl','h1','<slur-head side="RIGHT"/>')
        with self.assertRaisesRegex(ValueError, 'No sounding'):
            extract_hit_strikes(make_omr(score(objects, relations)))

    def test_cross_barline_never_merge(self):
        stacks = '<stack id="1" left="100" right="300"/><stack id="2" left="300" right="500"/>'
        objects = chord('c1') + head('h1', 296) + chord('c2') + head('h2', 304)
        relations = relation('c1','h1','<containment/>') + relation('c2','h2','<containment/>')
        count, strikes = extract_hit_strikes(make_omr(score(objects, relations, stacks)))
        self.assertEqual((count, len(strikes)), (2, 2))

    def test_displaced_same_slot_merges(self):
        objects, relations, parts = displaced_pair('1', '1')
        count, strikes = extract_hit_strikes(make_omr(score(objects, relations, parts=parts)))
        self.assertEqual((count, len(strikes)), (1, 1))
        self.assertEqual(strikes[0].x, 200)

    def test_displaced_unvoiced_pair_merges(self):
        objects, relations, parts = displaced_pair(None, None)
        count, strikes = extract_hit_strikes(make_omr(score(objects, relations, parts=parts)))
        self.assertEqual((count, len(strikes)), (1, 1))
        self.assertEqual(strikes[0].x, 200)

    def test_displaced_one_unvoiced_merges(self):
        objects, relations, parts = displaced_pair('1', None)
        count, strikes = extract_hit_strikes(make_omr(score(objects, relations, parts=parts)))
        self.assertEqual((count, len(strikes)), (1, 1))
        self.assertEqual(strikes[0].x, 200)

    def test_displaced_different_explicit_slots_do_not_merge(self):
        objects, relations, parts = displaced_pair('1', '2')
        count, strikes = extract_hit_strikes(make_omr(score(objects, relations, parts=parts)))
        self.assertEqual((count, len(strikes)), (2, 2))
        self.assertEqual([s.x for s in strikes], [200, 226])


if __name__ == '__main__':
    unittest.main()
