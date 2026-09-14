from fractions import Fraction
from pathlib import Path
import ast

from lxml import etree
from level1_clean_app import (
    MeasureRow,
    AttackRow,
    geometric_sequence,
    analyze_level1,
    local_name,
)


def m(i, number, start, num=4, den=4):
    full = Fraction(num * 4, den)
    return MeasureRow(
        index=i,
        number=str(number),
        start=Fraction(start),
        end=Fraction(start) + full,
        actual_duration=full,
        full_duration=full,
        pickup_shift=Fraction(0),
        numerator=num,
        denominator=den,
    )


assert geometric_sequence(2, 34)[:7] == [1, 2, 3, 5, 9, 17, 33]
assert geometric_sequence(3, 29)[:5] == [1, 2, 4, 10, 28]

measures = [m(i, i + 1, i * 4) for i in range(10)]
attacks = [
    AttackRow(Fraction(0), 0, "1"),
    AttackRow(Fraction(1), 0, "1"),
    AttackRow(Fraction(4), 1, "2"),
]
result = analyze_level1(measures, attacks)
positions = [Fraction(p["time_quarter"]) for p in result["points"]]
assert positions == [Fraction(x) for x in (0, 1, 2, 4, 8, 16, 32)]
labels = {Fraction(p["time_quarter"]): p["label"] for p in result["points"]}
assert labels[Fraction(0)] == "1"
assert labels[Fraction(1)] == "1"
assert labels[Fraction(2)] == "(1)"
assert labels[Fraction(4)] == "1"

# Non-element nodes from lxml must be ignored safely.
root = etree.fromstring(b"<score><!--comment--><part/></score>")
comment = list(root)[0]
assert local_name(comment.tag) == ""
assert local_name(list(root)[1].tag) == "part"

# Meter change restarts Level 1.
measures2 = [
    m(0, 1, 0, 4, 4),
    m(1, 2, 4, 4, 4),
    m(2, 3, 8, 3, 4),
    m(3, 4, 11, 3, 4),
]
result2 = analyze_level1(measures2, [])
seg0 = [Fraction(p["time_quarter"]) for p in result2["points"] if p["segment_index"] == 0]
seg1 = [Fraction(p["time_quarter"]) for p in result2["points"] if p["segment_index"] == 1]
assert seg0 == [Fraction(0), Fraction(1), Fraction(2), Fraction(4)]
assert seg1[:3] == [Fraction(8), Fraction(9), Fraction(11)]

# Isolation audit.
source = Path("level1_clean_app.py").read_text(encoding="utf-8")
tree = ast.parse(source)
imports = []
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        imports.extend(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
        imports.append(node.module or "")
forbidden = ("tone_metric", "dissertation_", "level1_standalone_app", "app")
assert not any(name.startswith(forbidden) for name in imports), imports

print("level1-clean-validation: PASS")
