import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from core import extract_hit_strikes


def make_omr(xml: str) -> Path:
    root = Path(tempfile.mkdtemp())
    path = root / "score.omr"
    with ZipFile(path, "w") as archive:
        archive.writestr("sheet#1/sheet#1.xml", xml)
    return path


def head(head_id: str, x: float, y: float = 180) -> str:
    return (
        f'<head id="{head_id}">'
        f'<bounds x="{x - 10}" y="{y - 10}" w="20" h="20"/>'
        f"</head>"
    )


def chord(
    chord_id: str,
    staff: str,
    x: float,
    y: float = 140,
    w: float = 24,
    h: float = 80,
) -> str:
    return (
        f'<head-chord id="{chord_id}" staff="{staff}">'
        f'<bounds x="{x - w / 2}" y="{y}" w="{w}" h="{h}"/>'
        f"</head-chord>"
    )


def containment(chord_id: str, head_id: str) -> str:
    return (
        f'<relation source="{chord_id}" target="{head_id}">'
        "<containment/>"
        "</relation>"
    )


def entry(chord_id: str, slot_id: str, status: str = "BEGIN") -> str:
    return (
        f"<entry><key>{slot_id}</key>"
        f'<value chord="{chord_id}" status="{status}"/></entry>'
    )


def score(
    *,
    stacks: str,
    entries: str,
    objects: str,
    relations: str,
    staffs: tuple[str, ...] = ("10",),
) -> str:
    staff_xml = "".join(
        (
            f'<staff id="{staff_id}">'
            '<line><point x="0" y="100"/></line>'
            '<line><point x="1000" y="300"/></line>'
            "</staff>"
        )
        for staff_id in staffs
    )
    return f"""<sheet>
      <scale><interline main="20"/></scale>
      <picture width="1000" height="800"/>
      <page><system id="1">
        {stacks}
        <part id="p1"><measure id="1"><voice id="1"><slots>
          {entries}
        </slots></voice></measure></part>
        {staff_xml}
      </system></page>
      {objects}
      <relations>{relations}</relations>
    </sheet>"""


class ReconciledHitEngineTests(unittest.TestCase):
    def test_one_symbolic_attack_is_one_hit(self):
        xml = score(
            stacks=(
                '<stack id="1" left="100" right="500">'
                '<slot id="1" x-offset="100" time-offset="0"/>'
                "</stack>"
            ),
            entries=entry("c1", "1"),
            objects=chord("c1", "10", 200) + head("h1", 200),
            relations=containment("c1", "h1"),
        )
        count, strikes, diagnostics = extract_hit_strikes(make_omr(xml))
        self.assertEqual((count, len(strikes)), (1, 1))
        self.assertEqual(diagnostics.base_events, 1)
        self.assertEqual(diagnostics.recovered_events, 0)

    def test_simultaneous_voices_share_one_exact_onset_even_when_displaced(self):
        xml = score(
            stacks=(
                '<stack id="1" left="100" right="500">'
                '<slot id="1" x-offset="140" time-offset="0"/>'
                "</stack>"
            ),
            entries=entry("c1", "1") + entry("c2", "1"),
            objects=(
                chord("c1", "10", 180)
                + head("h1", 180)
                + chord("c2", "10", 225)
                + head("h2", 225)
            ),
            relations=containment("c1", "h1") + containment("c2", "h2"),
        )
        count, strikes, _ = extract_hit_strikes(make_omr(xml))
        self.assertEqual((count, len(strikes)), (1, 1))

    def test_distinct_symbolic_times_survive_inverted_engraving(self):
        xml = score(
            stacks=(
                '<stack id="1" left="100" right="500">'
                '<slot id="1" x-offset="220" time-offset="0"/>'
                '<slot id="2" x-offset="180" time-offset="1/4"/>'
                "</stack>"
            ),
            entries=entry("c1", "1") + entry("c2", "2"),
            objects=(
                chord("c1", "10", 320)
                + head("h1", 320)
                + chord("c2", "10", 280)
                + head("h2", 280)
            ),
            relations=containment("c1", "h1") + containment("c2", "h2"),
        )
        count, strikes, _ = extract_hit_strikes(make_omr(xml))
        self.assertEqual((count, len(strikes)), (2, 2))
        self.assertLess(strikes[0].x, strikes[1].x)

    def test_tied_continuation_is_not_a_hit(self):
        xml = score(
            stacks=(
                '<stack id="1" left="100" right="500">'
                '<slot id="1" x-offset="100" time-offset="0"/>'
                "</stack>"
            ),
            entries=entry("c1", "1"),
            objects=(
                chord("c1", "10", 200)
                + head("h1", 200)
                + '<slur id="s1" tie="true"/>'
            ),
            relations=(
                containment("c1", "h1")
                + '<relation source="s1" target="h1">'
                '<slur-head side="RIGHT"/>'
                "</relation>"
            ),
        )
        with self.assertRaisesRegex(ValueError, "No sounding"):
            extract_hit_strikes(make_omr(xml))

    def test_mixed_tied_and_new_chord_is_one_hit(self):
        xml = score(
            stacks=(
                '<stack id="1" left="100" right="500">'
                '<slot id="1" x-offset="100" time-offset="0"/>'
                "</stack>"
            ),
            entries=entry("c1", "1"),
            objects=(
                chord("c1", "10", 200)
                + head("h1", 195)
                + head("h2", 205)
                + '<slur id="s1" tie="true"/>'
            ),
            relations=(
                containment("c1", "h1")
                + containment("c1", "h2")
                + '<relation source="s1" target="h1">'
                '<slur-head side="RIGHT"/>'
                "</relation>"
            ),
        )
        count, strikes, _ = extract_hit_strikes(make_omr(xml))
        self.assertEqual((count, len(strikes)), (1, 1))

    def test_unvoiced_same_staff_dense_notes_are_not_collapsed(self):
        xml = score(
            stacks=(
                '<stack id="1" left="100" right="500">'
                '<slot id="1" x-offset="80" time-offset="0"/>'
                "</stack>"
            ),
            entries=entry("c1", "1"),
            objects=(
                chord("c1", "10", 180)
                + head("h1", 180)
                + chord("c2", "10", 250, w=30)
                + head("h2", 250)
                + chord("c3", "10", 270, w=30)
                + head("h3", 270)
            ),
            relations=(
                containment("c1", "h1")
                + containment("c2", "h2")
                + containment("c3", "h3")
            ),
        )
        count, strikes, diagnostics = extract_hit_strikes(make_omr(xml))
        self.assertEqual((count, len(strikes)), (3, 3))
        self.assertEqual(diagnostics.recovered_events, 2)

    def test_unvoiced_cross_staff_alignment_reconciles_to_existing_hit(self):
        xml = score(
            stacks=(
                '<stack id="1" left="100" right="500">'
                '<slot id="1" x-offset="100" time-offset="0"/>'
                "</stack>"
            ),
            entries=entry("c1", "1"),
            objects=(
                chord("c1", "10", 220)
                + head("h1", 220)
                + chord("c2", "11", 224, y=320)
                + head("h2", 224, y=360)
            ),
            relations=containment("c1", "h1") + containment("c2", "h2"),
            staffs=("10", "11"),
        )
        count, strikes, diagnostics = extract_hit_strikes(make_omr(xml))
        self.assertEqual((count, len(strikes)), (1, 1))
        self.assertEqual(diagnostics.reconciled_chords, 1)

    def test_two_unvoiced_cross_staff_chords_form_one_recovered_global_hit(self):
        xml = score(
            stacks=(
                '<stack id="1" left="100" right="500">'
                '<slot id="1" x-offset="80" time-offset="0"/>'
                "</stack>"
            ),
            entries=entry("c1", "1"),
            objects=(
                chord("c1", "10", 180)
                + head("h1", 180)
                + chord("c2", "10", 320)
                + head("h2", 320)
                + chord("c3", "11", 326, y=320)
                + head("h3", 326, y=360)
            ),
            relations=(
                containment("c1", "h1")
                + containment("c2", "h2")
                + containment("c3", "h3")
            ),
            staffs=("10", "11"),
        )
        count, strikes, diagnostics = extract_hit_strikes(make_omr(xml))
        self.assertEqual((count, len(strikes)), (2, 2))
        self.assertEqual(diagnostics.recovered_events, 1)

    def test_cross_barline_chords_are_assigned_to_one_measure_only(self):
        stacks = (
            '<stack id="1" left="100" right="300">'
            '<slot id="1" x-offset="100" time-offset="0"/>'
            "</stack>"
            '<stack id="2" left="300" right="500">'
            '<slot id="1" x-offset="100" time-offset="0"/>'
            "</stack>"
        )
        xml = score(
            stacks=stacks,
            entries="",
            objects=(
                chord("c1", "10", 296)
                + head("h1", 296)
                + chord("c2", "10", 304)
                + head("h2", 304)
            ),
            relations=containment("c1", "h1") + containment("c2", "h2"),
        )
        count, strikes, diagnostics = extract_hit_strikes(make_omr(xml))
        self.assertEqual((count, len(strikes)), (2, 2))
        self.assertEqual(diagnostics.assigned_chords, 2)

    def test_non_head_graphics_never_create_hits(self):
        xml = score(
            stacks='<stack id="1" left="100" right="500"/>',
            entries="",
            objects=(
                '<alter id="a1" staff="10"><bounds x="200" y="150" '
                'w="20" h="40"/></alter>'
                '<rest-chord id="r1" staff="10">'
                '<bounds x="250" y="150" w="20" h="40"/>'
                "</rest-chord>"
            ),
            relations="",
        )
        with self.assertRaisesRegex(ValueError, "No sounding"):
            extract_hit_strikes(make_omr(xml))


if __name__ == "__main__":
    unittest.main()
