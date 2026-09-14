from __future__ import annotations

import ast
from fractions import Fraction
from pathlib import Path

from PIL import Image, ImageDraw

from level1_clean_v4 import (
    Attack,
    Measure,
    analyze_level1,
    detect_system_regions,
    sequence,
)


# Dissertation Level-1 sequence identities.
assert sequence(2, 34)[:7] == [1, 2, 3, 5, 9, 17, 33]
assert sequence(3, 29)[:5] == [1, 2, 4, 10, 28]

# Analytical regression: same clean Level-1 positions as the previous isolated build.
measures = [
    Measure(i, str(i + 1), Fraction(i * 4), Fraction((i + 1) * 4), Fraction(0), (4, 4))
    for i in range(10)
]
attacks = [Attack(Fraction(0)), Attack(Fraction(1)), Attack(Fraction(4))]
res = analyze_level1(measures, attacks)
positions = [Fraction(p["time_quarter"]) for p in res["points"]]
assert positions == [Fraction(x) for x in (0, 1, 2, 4, 8, 16, 32)]
labels = {Fraction(p["time_quarter"]): p["label"] for p in res["points"]}
assert labels[Fraction(0)] == "1"
assert labels[Fraction(1)] == "1"
assert labels[Fraction(2)] == "(1)"
assert labels[Fraction(4)] == "1"

# PDF system detector regression: two printed systems, each with two five-line staves.
im = Image.new("RGB", (1200, 1600), "white")
d = ImageDraw.Draw(im)
for base in (300, 430, 900, 1030):
    for j in range(5):
        y = base + j * 10
        d.line((100, y, 1100, y), fill="black", width=2)
regions = detect_system_regions(im, 2)
assert len(regions) == 2, regions
assert regions[0]["staff_top"] < 500, regions
assert regions[1]["staff_top"] > 700, regions
assert regions[0]["left"] < 180 and regions[0]["right"] > 1000, regions

# Isolation audit: v4 must not import any old analyzer implementation.
source = Path("level1_clean_v4.py").read_text(encoding="utf-8")
tree = ast.parse(source)
imports = []
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        imports.extend(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
        imports.append(node.module or "")
forbidden = ("tone_metric", "dissertation_", "level1_clean_v3", "level1_clean_app", "level1_standalone_app")
assert not any(name.startswith(forbidden) for name in imports), imports

print("level1-clean-v4-validation: PASS")
