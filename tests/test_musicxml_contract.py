import tempfile
import unittest
from pathlib import Path
from fractions import Fraction

from tone_metric.musicxml import parse_musicxml


HEADER = '''<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1">{measures}</part>
</score-partwise>'''


def parse_xml(text):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "score.musicxml"
        p.write_text(text, encoding="utf-8")
        return parse_musicxml(p)


class MusicXMLContractTests(unittest.TestCase):
    def test_pickup_is_right_aligned_inside_full_measure(self):
        m = '''
        <measure number="1" implicit="yes">
          <attributes><divisions>4</divisions><time><beats>3</beats><beat-type>4</beat-type></time></attributes>
          <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice></note>
          <note><pitch><step>D</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice></note>
        </measure>'''
        hits, measures, warnings = parse_xml(HEADER.format(measures=m))
        self.assertEqual(measures[0].full_duration, Fraction(3))
        self.assertEqual(measures[0].actual_duration, Fraction(2))
        self.assertEqual(measures[0].pickup_shift, Fraction(1))
        self.assertEqual([h.onset for h in hits], [Fraction(1), Fraction(2)])

    def test_tie_grace_rest_and_simultaneous_chord_rules(self):
        m = '''
        <measure number="1">
          <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
          <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><tie type="start"/></note>
          <note><chord/><pitch><step>E</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice></note>
          <note><grace/><pitch><step>F</step><octave>4</octave></pitch><voice>1</voice></note>
          <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><tie type="stop"/></note>
          <note><rest/><duration>4</duration><voice>1</voice></note>
          <note><pitch><step>G</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice></note>
        </measure>'''
        hits, measures, warnings = parse_xml(HEADER.format(measures=m))
        self.assertEqual([h.onset for h in hits], [Fraction(0), Fraction(3)])
        self.assertEqual(set(hits[0].to_dict()["pitches"]), {"C4", "E4"})
        self.assertEqual(hits[0].source_count if hasattr(hits[0], 'source_count') else len(hits[0].sources), 2)
        self.assertEqual(set(hits[1].to_dict()["pitches"]), {"G4"})

    def test_missing_initial_meter_fails_instead_of_assuming_4_4(self):
        m = """
        <measure number="1">
          <attributes><divisions>4</divisions></attributes>
          <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice></note>
        </measure>"""
        with self.assertRaisesRegex(ValueError, "will not assume 4/4"):
            parse_xml(HEADER.format(measures=m))


if __name__ == "__main__":
    unittest.main()
