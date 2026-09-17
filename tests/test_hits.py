import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from core import extract_hit_strikes


def make_omr(xml: str) -> Path:
    td = Path(tempfile.mkdtemp())
    p = td / 'score.omr'
    with ZipFile(p, 'w') as zf:
        zf.writestr('sheet#1/sheet#1.xml', xml)
    return p


BASE_HEAD = '''
<sheet><picture width="1000" height="800"/><page><system id="1">
  <stack id="1" left="100" right="900"><slot id="1" x-offset="50" time-offset="0"/></stack>
  <part id="1"><measure id="1"><voice id="1"><slots>{entries}</slots></voice></measure></part>
  <staff><line><point x="0" y="100"/><point x="100" y="100"/></line><line><point x="0" y="300"/><point x="100" y="300"/></line></staff>
</system></page>
{objects}
<relations>{relations}</relations>
</sheet>'''


def entry(chord: str, key: str = '1', status: str = 'BEGIN') -> str:
    return f'<entry><key>{key}</key><value chord="{chord}" status="{status}"/></entry>'


def head(hid: str, x: int, y: int = 180) -> str:
    return f'<head id="{hid}"><bounds x="{x}" y="{y}" w="20" h="20"/></head>'


def chord(cid: str) -> str:
    return f'<head-chord id="{cid}"/>'


def contains(cid: str, hid: str) -> str:
    return f'<relation source="{cid}" target="{hid}"><containment/></relation>'


class HitStrikeTests(unittest.TestCase):
    def test_strike_uses_notehead_not_slot_x(self):
        xml = BASE_HEAD.format(
            entries=entry('c1'),
            objects=chord('c1') + head('h1', 400),
            relations=contains('c1', 'h1'),
        )
        hit_count, strikes = extract_hit_strikes(make_omr(xml))
        self.assertEqual(hit_count, 1)
        self.assertEqual(len(strikes), 1)
        self.assertEqual(strikes[0].x, 410.0)
        self.assertNotEqual(strikes[0].x, 150.0)

    def test_rest_or_non_head_chord_is_not_a_hit(self):
        xml = BASE_HEAD.format(
            entries=entry('r1'),
            objects='<rest-chord id="r1"><bounds x="400" y="180" w="20" h="20"/></rest-chord>',
            relations='',
        )
        with self.assertRaisesRegex(ValueError, 'No sounding note attacks'):
            extract_hit_strikes(make_omr(xml))

    def test_tied_continuation_is_not_a_new_hit(self):
        xml = BASE_HEAD.format(
            entries=entry('c1'),
            objects=chord('c1') + head('h1', 400) + '<slur id="s1" tie="true"/>',
            relations=contains('c1', 'h1') + '<relation source="s1" target="h1"><slur-head side="RIGHT"/></relation>',
        )
        with self.assertRaisesRegex(ValueError, 'No sounding note attacks'):
            extract_hit_strikes(make_omr(xml))

    def test_mixed_tied_and_new_chord_marks_only_new_attack_column(self):
        xml = BASE_HEAD.format(
            entries=entry('c1'),
            objects=chord('c1') + head('h_tied', 300) + head('h_new', 500) + '<slur id="s1" tie="true"/>',
            relations=(contains('c1', 'h_tied') + contains('c1', 'h_new') +
                       '<relation source="s1" target="h_tied"><slur-head side="RIGHT"/></relation>'),
        )
        hit_count, strikes = extract_hit_strikes(make_omr(xml))
        self.assertEqual(hit_count, 1)
        self.assertEqual([s.x for s in strikes], [510.0])

    def test_simultaneous_visually_separated_attacks_are_all_visible(self):
        xml = BASE_HEAD.format(
            entries=entry('c1') + entry('c2'),
            objects=chord('c1') + head('h1', 300) + chord('c2') + head('h2', 500),
            relations=contains('c1', 'h1') + contains('c2', 'h2'),
        )
        hit_count, strikes = extract_hit_strikes(make_omr(xml))
        self.assertEqual(hit_count, 1)
        self.assertEqual([s.x for s in strikes], [310.0, 510.0])

    def test_same_visible_attack_column_is_not_duplicated(self):
        xml = BASE_HEAD.format(
            entries=entry('c1') + entry('c2'),
            objects=chord('c1') + head('h1', 400) + chord('c2') + head('h2', 402),
            relations=contains('c1', 'h1') + contains('c2', 'h2'),
        )
        hit_count, strikes = extract_hit_strikes(make_omr(xml))
        self.assertEqual(hit_count, 1)
        self.assertEqual(len(strikes), 1)
        self.assertAlmostEqual(strikes[0].x, 411.0)


if __name__ == '__main__':
    unittest.main()
