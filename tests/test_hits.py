import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from core import extract_hit_strikes


def make_omr(xml: str) -> Path:
    td = Path(tempfile.mkdtemp())
    path = td / 'score.omr'
    with ZipFile(path, 'w') as zf:
        zf.writestr('sheet#1/sheet#1.xml', xml)
    return path


def entry(chord: str, key: str = '1', status: str = 'BEGIN') -> str:
    return f'<entry><key>{key}</key><value chord="{chord}" status="{status}"/></entry>'


def relation(source: str, target: str, body: str) -> str:
    return f'<relation source="{source}" target="{target}">{body}</relation>'


def head(head_id: str, x: int) -> str:
    return f'<head id="{head_id}"><bounds x="{x}" y="180" w="20" h="20"/></head>'


def chord(chord_id: str, staff: str = '10') -> str:
    return f'<head-chord id="{chord_id}" staff="{staff}"/>'


def score(entries_a: str, objects: str, relations: str, entries_b: str = '') -> str:
    return f'''<sheet>
      <picture width="1000" height="800"/>
      <page><system id="1">
        <stack id="1" left="100" right="400">
          <slot id="1" x-offset="50" time-offset="0"/>
          <slot id="2" x-offset="150" time-offset="1/4"/>
        </stack>
        <part id="p1"><measure id="1"><voice id="1"><slots>{entries_a}</slots></voice></measure></part>
        <part id="p2"><measure id="1"><voice id="1"><slots>{entries_b}</slots></voice></measure></part>
        <staff id="10">
          <line><point x="0" y="100"/><point x="100" y="100"/></line>
          <line><point x="0" y="300"/><point x="100" y="300"/></line>
        </staff>
      </system></page>
      {objects}
      <relations>{relations}</relations>
    </sheet>'''


class HitStrikeTests(unittest.TestCase):
    def test_single_new_chord_creates_one_slot_strike(self):
        xml = score(
            entry('c1'),
            chord('c1') + head('h1', 400),
            relation('c1', 'h1', '<containment/>'),
        )
        hit_count, strikes = extract_hit_strikes(make_omr(xml))
        self.assertEqual(hit_count, 1)
        self.assertEqual(len(strikes), 1)
        self.assertEqual(strikes[0].x, 150.0)

    def test_simultaneous_voices_and_staves_merge_to_one_strike(self):
        xml = score(
            entry('c1'),
            chord('c1') + head('h1', 300) + chord('c2') + head('h2', 500),
            relation('c1', 'h1', '<containment/>') + relation('c2', 'h2', '<containment/>'),
            entries_b=entry('c2'),
        )
        hit_count, strikes = extract_hit_strikes(make_omr(xml))
        self.assertEqual(hit_count, 1)
        self.assertEqual(len(strikes), 1)

    def test_two_different_slots_create_two_strikes(self):
        xml = score(
            entry('c1', '1') + entry('c2', '2'),
            chord('c1') + head('h1', 300) + chord('c2') + head('h2', 500),
            relation('c1', 'h1', '<containment/>') + relation('c2', 'h2', '<containment/>'),
        )
        hit_count, strikes = extract_hit_strikes(make_omr(xml))
        self.assertEqual(hit_count, 2)
        self.assertEqual([s.x for s in strikes], [150.0, 250.0])

    def test_tied_continuation_creates_no_hit(self):
        xml = score(
            entry('c1'),
            chord('c1') + head('h1', 400) + '<slur id="s1" tie="true"/>',
            relation('c1', 'h1', '<containment/>') +
            relation('s1', 'h1', '<slur-head side="RIGHT"/>'),
        )
        with self.assertRaisesRegex(ValueError, 'No sounding note attacks'):
            extract_hit_strikes(make_omr(xml))

    def test_mixed_tied_and_new_notes_still_create_one_hit(self):
        xml = score(
            entry('c1'),
            chord('c1') + head('h1', 300) + head('h2', 400) + '<slur id="s1" tie="true"/>',
            relation('c1', 'h1', '<containment/>') +
            relation('c1', 'h2', '<containment/>') +
            relation('s1', 'h1', '<slur-head side="RIGHT"/>'),
        )
        hit_count, strikes = extract_hit_strikes(make_omr(xml))
        self.assertEqual(hit_count, 1)
        self.assertEqual(len(strikes), 1)

    def test_unvoiced_semantic_head_chord_is_recovered(self):
        xml = score(
            entry('c1'),
            chord('c1') + head('h1', 100) + chord('c2') + head('h2', 500),
            relation('c1', 'h1', '<containment/>') + relation('c2', 'h2', '<containment/>'),
        )
        hit_count, strikes = extract_hit_strikes(make_omr(xml))
        self.assertEqual(hit_count, 2)
        self.assertEqual(sorted(round(s.x) for s in strikes), [150, 510])

    def test_recovered_chord_near_existing_hit_is_not_doubled(self):
        xml = score(
            entry('c1'),
            chord('c1') + head('h1', 100) + chord('c2') + head('h2', 142),
            relation('c1', 'h1', '<containment/>') + relation('c2', 'h2', '<containment/>'),
        )
        hit_count, strikes = extract_hit_strikes(make_omr(xml))
        self.assertEqual(hit_count, 1)
        self.assertEqual(len(strikes), 1)

    def test_unvoiced_tied_continuation_is_not_recovered(self):
        xml = score(
            entry('c1'),
            chord('c1') + head('h1', 100) + chord('c2') + head('h2', 500) + '<slur id="s1" tie="true"/>',
            relation('c1', 'h1', '<containment/>') +
            relation('c2', 'h2', '<containment/>') +
            relation('s1', 'h2', '<slur-head side="RIGHT"/>'),
        )
        hit_count, strikes = extract_hit_strikes(make_omr(xml))
        self.assertEqual(hit_count, 1)
        self.assertEqual(len(strikes), 1)

    def test_non_head_chord_does_not_create_hit(self):
        xml = score(
            entry('r1'),
            '<rest-chord id="r1" staff="10"/>',
            '',
        )
        with self.assertRaisesRegex(ValueError, 'No sounding note attacks'):
            extract_hit_strikes(make_omr(xml))


if __name__ == '__main__':
    unittest.main()
