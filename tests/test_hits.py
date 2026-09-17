import tempfile
import unittest
from pathlib import Path
from fractions import Fraction
from core import Hit, extract_hits


def write_xml(body: str) -> Path:
    td = tempfile.mkdtemp()
    p = Path(td) / 'score.musicxml'
    p.write_text(body, encoding='utf-8')
    return p


class HitTests(unittest.TestCase):
    def test_only_new_sounding_onsets(self):
        p = write_xml('''<score-partwise version="4.0">
        <part-list><score-part id="P1"><part-name>P1</part-name></score-part></part-list>
        <part id="P1"><measure number="1"><attributes><divisions>4</divisions></attributes>
          <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice></note>
          <note><chord/><pitch><step>E</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice></note>
          <note><rest/><duration>4</duration><voice>1</voice></note>
          <note><grace/><pitch><step>D</step><octave>4</octave></pitch><voice>1</voice></note>
          <note><pitch><step>F</step><octave>4</octave></pitch><duration>4</duration><tie type="start"/><voice>1</voice></note>
          <note><pitch><step>F</step><octave>4</octave></pitch><duration>4</duration><tie type="stop"/><voice>1</voice></note>
        </measure></part></score-partwise>''')
        self.assertEqual(extract_hits(p), [Hit(0, Fraction(0)), Hit(0, Fraction(2))])

    def test_simultaneous_voices_merge_globally(self):
        p = write_xml('''<score-partwise version="4.0">
        <part-list><score-part id="P1"><part-name>P1</part-name></score-part><score-part id="P2"><part-name>P2</part-name></score-part></part-list>
        <part id="P1"><measure number="1"><attributes><divisions>4</divisions></attributes>
          <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration></note>
          <note><pitch><step>D</step><octave>4</octave></pitch><duration>4</duration></note>
        </measure></part>
        <part id="P2"><measure number="1"><attributes><divisions>4</divisions></attributes>
          <note><pitch><step>G</step><octave>3</octave></pitch><duration>8</duration></note>
        </measure></part>
        </score-partwise>''')
        self.assertEqual(extract_hits(p), [Hit(0, Fraction(0)), Hit(0, Fraction(1))])

    def test_backup_restores_independent_voice_time(self):
        p = write_xml('''<score-partwise version="4.0">
        <part-list><score-part id="P1"><part-name>P1</part-name></score-part></part-list>
        <part id="P1"><measure number="1"><attributes><divisions>4</divisions></attributes>
          <note><pitch><step>C</step><octave>4</octave></pitch><duration>8</duration><voice>1</voice></note>
          <backup><duration>8</duration></backup>
          <note><pitch><step>E</step><octave>4</octave></pitch><duration>4</duration><voice>2</voice></note>
          <note><pitch><step>F</step><octave>4</octave></pitch><duration>4</duration><voice>2</voice></note>
        </measure></part></score-partwise>''')
        self.assertEqual(extract_hits(p), [Hit(0, Fraction(0)), Hit(0, Fraction(1))])


if __name__ == '__main__':
    unittest.main()
