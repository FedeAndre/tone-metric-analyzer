from fractions import Fraction
import ast
from pathlib import Path

from level1_clean_v3 import Measure, Attack, analyze_level1, parse_meter, sequence

assert parse_meter("auto") == (4,4)
assert parse_meter("") == (4,4)
assert sequence(2, 34)[:7] == [1,2,3,5,9,17,33]
assert sequence(3, 29)[:5] == [1,2,4,10,28]

measures=[]
for i in range(10):
    start=Fraction(i*4)
    measures.append(Measure(i,str(i+1),start,start+4,Fraction(0),(4,4)))
attacks=[Attack(Fraction(x)) for x in (0,1,4)]
res=analyze_level1(measures,attacks)
positions=[Fraction(p['time_quarter']) for p in res['points']]
assert positions == [Fraction(x) for x in (0,1,2,4,8,16,32)]
labels={Fraction(p['time_quarter']):p['label'] for p in res['points']}
assert labels[0]=='1' and labels[1]=='1' and labels[2]=='(1)' and labels[4]=='1'

source=Path('level1_clean_v3.py').read_text(encoding='utf-8')
tree=ast.parse(source)
imports=[]
for node in ast.walk(tree):
    if isinstance(node,ast.Import): imports.extend(a.name for a in node.names)
    elif isinstance(node,ast.ImportFrom): imports.append(node.module or '')
forbidden=('tone_metric','dissertation_','level1_clean_app','level1_standalone_app')
assert not any(name.startswith(forbidden) for name in imports), imports

print('level1-clean-v3-validation: PASS')
