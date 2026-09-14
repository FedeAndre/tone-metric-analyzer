from __future__ import annotations

import tempfile
import zipfile
from fractions import Fraction
from pathlib import Path

import fitz
from lxml import etree

import level1_clean_v4 as v4
import level1_clean_v5 as v5


def build_synthetic_omr(path: Path):
    root = etree.Element("sheet")
    etree.SubElement(root, "picture", width="1000", height="1000")
    page = etree.SubElement(root, "page", id="1", **{"measure-count": "1"})
    system = etree.SubElement(page, "system", id="1", indented="true")
    stack = etree.SubElement(system, "stack", id="1", left="100", right="900", expected="1", duration="1")
    for i, (t, xoff) in enumerate((("0", "100"), ("1/4", "300"), ("1/2", "500"), ("3/4", "700")), 1):
        etree.SubElement(stack, "slot", id=str(i), **{"x-offset": xoff, "time-offset": t})
    part = etree.SubElement(system, "part", id="1")
    staff = etree.SubElement(part, "staff", id="1", left="100", right="900")
    lines = etree.SubElement(staff, "lines")
    for j, y in enumerate((400, 410, 420, 430, 440), 1):
        line = etree.SubElement(lines, "line", thickness="2", glyph=str(j))
        etree.SubElement(line, "point", x="100", y=str(y))
        etree.SubElement(line, "point", x="900", y=str(y))

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("sheet#1/sheet#1.xml", etree.tostring(root, xml_declaration=True, encoding="UTF-8"))
        zf.writestr("book.xml", b"<book/>")


def build_synthetic_pdf(path: Path):
    doc = fitz.open()
    page = doc.new_page(width=500, height=500)
    for y in (200, 205, 210, 215, 220):
        page.draw_line((50, y), (450, y), color=(0, 0, 0), width=0.7)
    doc.save(path)
    doc.close()


def main():
    # Guard the user's non-negotiable constraint: v5 must reuse the exact v4
    # analytical functions rather than introducing a modified Level-1 engine.
    assert v5.core.parse_score is v4.parse_score
    assert v5.core.analyze_level1 is v4.analyze_level1
    assert v5.core.sequence is v4.sequence

    with tempfile.TemporaryDirectory(prefix="validate-level1-v5-") as td:
        td = Path(td)
        omr = td / "synthetic.omr"
        pdf = td / "synthetic.pdf"
        build_synthetic_omr(omr)
        build_synthetic_pdf(pdf)

        geom = v5.parse_omr_geometry(omr, 1)
        assert len(geom) == 1 and len(geom[0]["systems"]) == 1
        stack = geom[0]["systems"][0]["stacks"][0]
        assert v5._nearest_slot_x(stack, Fraction(0)) == 200.0
        assert v5._nearest_slot_x(stack, Fraction(1, 4)) == 400.0
        assert v5._nearest_slot_x(stack, Fraction(1, 2)) == 600.0
        assert v5._nearest_slot_x(stack, Fraction(3, 4)) == 800.0

        measure = v4.Measure(0, "1", Fraction(0), Fraction(4), Fraction(0), (4, 4))
        result = {
            "points": [
                {"measure_index": 0, "time_quarter": "0", "attack": True, "label": "1"},
                {"measure_index": 0, "time_quarter": "1", "attack": True, "label": "1"},
                {"measure_index": 0, "time_quarter": "2", "attack": True, "label": "1"},
                {"measure_index": 0, "time_quarter": "3", "attack": True, "label": "1"},
            ]
        }
        pages, diagnostics = v5.render_original_pdf_with_level1_exact(pdf, omr, [measure], result)
        assert len(pages) == 1 and pages[0].startswith("data:image/png;base64,")
        s = diagnostics[0]["systems"][0]
        assert s["attack_labels_omr_slot_aligned"] == 4
        assert s["attack_labels_without_exact_slot"] == 0

    print("level1-clean-v5-validation: PASS")


if __name__ == "__main__":
    main()
