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


def head(hid, x, staff='10'):
    return f'<head id="{hid}" pitch="0" shape="NOTEHEAD_BLACK" staff="{staff}"><bounds x="{x-10}" y="180" w="20" h="20"/></head>'


def stem(sid, x, staff='10'):
    return f'<stem id="{sid}" staff="{staff}"><bounds x="{x-2}" y="140" w="4" h="60"/></stem>'


def chord(cid, x, staff='10'):
    return f'<head-chord id="{cid}" staff="{staff}"><bounds x="{x-12}" y="140" w="24" h="70"/></head-chord>'


def relations(cid, hid, sid=None):
    r = f'<relation source="{cid}" target="{hid}"><containment/></relation>'
    if sid:
        r += f'<relation source="{cid}" target="{sid}"><chord-stem/></relation>'
    return r


def entry(cid, slot):
    return f'<entry><key>{slot}</key><value chord="{cid}" status="BEGIN"/></entry>'


def score(objects, rels, part_entries='', second_part_entries='', extra_staff=False, stacks=None):
    if stacks is None:
        stacks = '<stack id="1" left="100" right="500"><slot id="1" time-offset="0" x-offset="100"/><slot id="2" time-offset="1/4" x-offset="180"/></stack>'
    staff20 = '<staff id="20"><line><point y="400"/></line><line><point y="500"/></line></staff>' if extra_staff else ''
    part2 = f'<part><measure><voice>{second_part_entries}</voice></measure></part>' if second_part_entries else ''
    return f'''<sheet><scale><interline main="20"/></scale><picture width="1000" height="800"/><page><system id="1">{stacks}<part><measure><voice>{part_entries}</voice></measure></part>{part2}<staff id="10"><line><point y="100"/></line><line><point y="300"/></line></staff>{staff20}</system></page>{objects}<relations>{rels}</relations></sheet>'''


class SequenceHitTests(unittest.TestCase):
    def test_one_attack_is_one_real_head_strike(self):
        objs = chord('c1',200)+head('h1',200)+stem('s1',212)
        n,s,d = extract_hit_strikes(make_omr(score(objs,relations('c1','h1','s1'),entry('c1','1'))))
        self.assertEqual(n,1); self.assertEqual(s[0].x,200); self.assertEqual(d.synthetic_strike_positions,0)

    def test_same_staff_same_time_polyphony_is_one_local_onset(self):
        objs = chord('c1',180)+head('h1',180)+chord('c2',230)+head('h2',230)
        rels = relations('c1','h1')+relations('c2','h2')
        n,s,_ = extract_hit_strikes(make_omr(score(objs,rels,entry('c1','1')+entry('c2','1'))))
        self.assertEqual(n,1); self.assertIn(s[0].x,{180,230})

    def test_same_staff_different_known_times_never_collapse(self):
        objs = chord('c1',205)+head('h1',205)+chord('c2',208)+head('h2',208)
        rels = relations('c1','h1')+relations('c2','h2')
        n,_,_ = extract_hit_strikes(make_omr(score(objs,rels,entry('c1','1')+entry('c2','2'))))
        self.assertEqual(n,2)

    def test_cross_staff_equal_known_time_matches_despite_displacement(self):
        objs = chord('c1',180,'10')+head('h1',180,'10')+chord('c2',250,'20')+head('h2',250,'20')
        rels = relations('c1','h1')+relations('c2','h2')
        xml = score(objs,rels,entry('c1','1'),entry('c2','1'),extra_staff=True)
        n,_,_ = extract_hit_strikes(make_omr(xml)); self.assertEqual(n,1)

    def test_cross_staff_different_known_time_never_matches(self):
        objs = chord('c1',200,'10')+head('h1',200,'10')+chord('c2',202,'20')+head('h2',202,'20')
        rels = relations('c1','h1')+relations('c2','h2')
        xml = score(objs,rels,entry('c1','1'),entry('c2','2'),extra_staff=True)
        n,_,_ = extract_hit_strikes(make_omr(xml)); self.assertEqual(n,2)

    def test_untimed_cross_staff_aligned_note_matches_once(self):
        objs = chord('c1',200,'10')+head('h1',200,'10')+chord('u1',210,'20')+head('hu1',210,'20')
        rels = relations('c1','h1')+relations('u1','hu1')
        xml = score(objs,rels,entry('c1','1'),'',extra_staff=True)
        n,_,_ = extract_hit_strikes(make_omr(xml)); self.assertEqual(n,1)

    def test_order_preserving_alignment_cannot_absorb_two_notes_into_one(self):
        # Top staff has two untimed notes around one lower timed note. One may
        # align; the other must survive as its own hit.
        objs = (chord('u1',190,'10')+head('h1',190,'10')+chord('u2',215,'10')+head('h2',215,'10')+
                chord('c3',202,'20')+head('h3',202,'20'))
        rels = relations('u1','h1')+relations('u2','h2')+relations('c3','h3')
        xml = score(objs,rels,'',entry('c3','1'),extra_staff=True)
        n,_,_ = extract_hit_strikes(make_omr(xml)); self.assertEqual(n,2)

    def test_untimed_same_staff_dense_notes_stay_separate(self):
        objs = chord('u1',200)+head('h1',200)+chord('u2',225)+head('h2',225)
        rels = relations('u1','h1')+relations('u2','h2')
        n,_,_ = extract_hit_strikes(make_omr(score(objs,rels)))
        self.assertEqual(n,2)

    def test_tie_right_continuation_removed(self):
        objs = chord('c1',200)+head('h1',200)+'<slur id="sl" tie="true"/>'
        rels = relations('c1','h1')+'<relation source="sl" target="h1"><slur-head side="RIGHT"/></relation>'
        with self.assertRaisesRegex(ValueError,'No sounding'):
            extract_hit_strikes(make_omr(score(objs,rels,entry('c1','1'))))

    def test_mixed_tie_and_new_head_remains_attack(self):
        objs = chord('c1',200)+head('h1',195)+head('h2',205)+'<slur id="sl" tie="true"/>'
        rels = relations('c1','h1')+f'<relation source="c1" target="h2"><containment/></relation>'+ '<relation source="sl" target="h1"><slur-head side="RIGHT"/></relation>'
        n,s,_ = extract_hit_strikes(make_omr(score(objs,rels,entry('c1','1'))))
        self.assertEqual(n,1); self.assertEqual(s[0].x,205)

    def test_cross_barline_measure_membership_is_independent(self):
        stacks = '<stack id="1" left="100" right="300"/><stack id="2" left="300" right="500"/>'
        objs = chord('c1',296)+head('h1',296)+chord('c2',304)+head('h2',304)
        rels = relations('c1','h1')+relations('c2','h2')
        n,s,_ = extract_hit_strikes(make_omr(score(objs,rels,stacks=stacks)))
        self.assertEqual(n,2); self.assertEqual({x.measure_index for x in s},{1,2})


if __name__ == '__main__': unittest.main()
