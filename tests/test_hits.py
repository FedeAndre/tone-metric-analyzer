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


def head(head_id: str, x: int, y: int = 180, pitch: str = '0') -> str:
    return (
        f'<head id="{head_id}" pitch="{pitch}">'
        f'<bounds x="{x - 10}" y="{y - 10}" w="20" h="20"/>'
        f'</head>'
    )


def chord(chord_id: str, staff: str = '10', x: int = 100, y: int = 150) -> str:
    return (
        f'<head-chord id="{chord_id}" staff="{staff}">'
        f'<bounds x="{x}" y="{y}" w="30" h="80"/>'
        f'</head-chord>'
    )


def relation(source: str, target: str, body: str) -> str:
    return f'<relation source="{source}" target="{target}">{body}</relation>'


def entry(chord_id: str, slot_id: str, status: str = 'BEGIN') -> str:
    return (
        f'<entry><key>{slot_id}</key>'
        f'<value chord="{chord_id}" status="{status}"/></entry>'
    )


def score(slots: str, entries: str, objects: str, relations: str, right: int = 500) -> str:
    return f'''<sheet>
      <picture width="1000" height="800"/>
      <page><system id="1">
        <stack id="1" left="100" right="{right}">{slots}</stack>
        <part id="p1"><measure id="1"><voice id="1"><slots>{entries}</slots></voice></measure></part>
        <staff id="10">
          <line><point x="0" y="100"/></line>
          <line><point x="0" y="300"/></line>
        </staff>
      </system></page>
      {objects}
      <relations>{relations}</relations>
    </sheet>'''


class HitStrikeTests(unittest.TestCase):
    def test_simultaneous_entries_merge_to_one_hit(self):
        slots = '<slot id="1" x-offset="100" time-offset="0"/>'
        objects = chord('c1') + head('h1', 200) + chord('c2') + head('h2', 205)
        relations = (
            relation('c1', 'h1', '<containment/>')
            + relation('c2', 'h2', '<containment/>')
        )
        hit_count, strikes = extract_hit_strikes(
            make_omr(score(slots, entry('c1', '1') + entry('c2', '1'), objects, relations))
        )
        self.assertEqual((hit_count, len(strikes)), (1, 1))

    def test_tied_continuation_creates_no_hit(self):
        slots = '<slot id="1" x-offset="100" time-offset="0"/>'
        objects = chord('c1') + head('h1', 200) + '<slur id="sl" tie="true"/>'
        relations = (
            relation('c1', 'h1', '<containment/>')
            + relation('sl', 'h1', '<slur-head side="RIGHT"/>')
        )
        with self.assertRaisesRegex(ValueError, 'No sounding'):
            extract_hit_strikes(make_omr(score(slots, entry('c1', '1'), objects, relations)))

    def test_mixed_tied_and_new_chord_still_creates_one_hit(self):
        slots = '<slot id="1" x-offset="100" time-offset="0"/>'
        objects = (
            chord('c1')
            + head('h1', 200)
            + head('h2', 220)
            + '<slur id="sl" tie="true"/>'
        )
        relations = (
            relation('c1', 'h1', '<containment/>')
            + relation('c1', 'h2', '<containment/>')
            + relation('sl', 'h1', '<slur-head side="RIGHT"/>')
        )
        hit_count, strikes = extract_hit_strikes(
            make_omr(score(slots, entry('c1', '1'), objects, relations))
        )
        self.assertEqual((hit_count, len(strikes)), (1, 1))

    def test_unvoiced_real_note_is_recovered(self):
        slots = '<slot id="1" x-offset="100" time-offset="0"/>'
        objects = chord('c1') + head('h1', 200) + chord('c2', x=300) + head('h2', 330)
        relations = (
            relation('c1', 'h1', '<containment/>')
            + relation('c2', 'h2', '<containment/>')
        )
        hit_count, strikes = extract_hit_strikes(
            make_omr(score(slots, entry('c1', '1'), objects, relations))
        )
        self.assertEqual((hit_count, len(strikes)), (2, 2))

    def test_unvoiced_chord_with_any_shared_head_is_not_doubled(self):
        slots = '<slot id="1" x-offset="100" time-offset="0"/>'
        objects = (
            chord('c1') + head('h1', 250)
            + chord('c2', x=220) + head('h2a', 220) + head('h2b', 252)
        )
        relations = (
            relation('c1', 'h1', '<containment/>')
            + relation('c2', 'h2a', '<containment/>')
            + relation('c2', 'h2b', '<containment/>')
        )
        hit_count, strikes = extract_hit_strikes(
            make_omr(score(slots, entry('c1', '1'), objects, relations))
        )
        self.assertEqual((hit_count, len(strikes)), (1, 1))

    def test_inverted_slot_x_is_repaired_without_losing_hits(self):
        slots = (
            '<slot id="1" x-offset="100" time-offset="0"/>'
            '<slot id="2" x-offset="220" time-offset="1/4"/>'
            '<slot id="3" x-offset="180" time-offset="1/2"/>'
        )
        objects = (
            chord('c1') + head('h1', 200)
            + chord('c2') + head('h2', 320)
            + chord('c3') + head('h3', 280)
        )
        relations = (
            relation('c1', 'h1', '<containment/>')
            + relation('c2', 'h2', '<containment/>')
            + relation('c3', 'h3', '<containment/>')
        )
        hit_count, strikes = extract_hit_strikes(
            make_omr(
                score(
                    slots,
                    entry('c1', '1') + entry('c2', '2') + entry('c3', '3'),
                    objects,
                    relations,
                )
            )
        )
        xs = [strike.x for strike in strikes]
        self.assertEqual((hit_count, len(strikes)), (3, 3))
        self.assertTrue(xs[0] < xs[1] < xs[2])
        self.assertGreaterEqual(min(b - a for a, b in zip(xs, xs[1:])), 12)

    def test_distinct_times_at_same_engraving_x_remain_distinct(self):
        slots = (
            '<slot id="1" x-offset="100" time-offset="0"/>'
            '<slot id="2" x-offset="102" time-offset="1/4"/>'
        )
        objects = chord('c1') + head('h1', 202) + chord('c2') + head('h2', 203)
        relations = (
            relation('c1', 'h1', '<containment/>')
            + relation('c2', 'h2', '<containment/>')
        )
        hit_count, strikes = extract_hit_strikes(
            make_omr(
                score(
                    slots,
                    entry('c1', '1') + entry('c2', '2'),
                    objects,
                    relations,
                    right=400,
                )
            )
        )
        xs = [strike.x for strike in strikes]
        self.assertEqual((hit_count, len(strikes)), (2, 2))
        self.assertGreaterEqual(xs[1] - xs[0], 12)

    def test_rest_chord_does_not_create_hit(self):
        slots = '<slot id="1" x-offset="100" time-offset="0"/>'
        objects = '<rest-chord id="r1" staff="10"/>'
        with self.assertRaisesRegex(ValueError, 'No sounding'):
            extract_hit_strikes(make_omr(score(slots, entry('r1', '1'), objects, '')))


if __name__ == '__main__':
    unittest.main()
