from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import level1_clean_v4 as v4
import level1_clean_v5 as v5
import level1_level2_clean_v6 as v6


def make_measures(count: int):
    return [v4.Measure(i, str(i + 1), Fraction(i * 4), Fraction((i + 1) * 4), Fraction(0), (4, 4)) for i in range(count)]


def main():
    # Non-regression: Level 1 and exact OMR helpers are literally the same objects.
    assert v6.core.analyze_level1 is v4.analyze_level1
    assert v6.v5._nearest_slot_x is v5._nearest_slot_x
    assert v6.v5._interpolated_x is v5._interpolated_x
    assert v6.v5.parse_omr_geometry is v5.parse_omr_geometry

    # Dissertation Example 1.2, 3 measures of 4/4.
    measures = make_measures(3)
    attacks = [v4.Attack(Fraction(i)) for i in range(12)]
    l1_before = v4.analyze_level1(measures, attacks)
    result = v6.analyze_levels12(measures, attacks)

    # Frozen Level 1 must be byte-for-byte-equivalent as Python data.
    assert result["points"] == l1_before["points"]
    assert result["level1_points"] == l1_before["points"]
    assert [Fraction(p["time_quarter"]) for p in result["level1_points"]] == [Fraction(x) for x in (0, 1, 2, 4, 8)]

    # Dissertation pp. 15-16 / Example 1.2:
    # L2 begins only where an L1 span contains a gap; it restarts at each L1.
    expected_l2 = [Fraction(x) for x in (2, 3, 4, 5, 6, 8, 9, 10)]
    actual_l2 = [Fraction(p["time_quarter"]) for p in result["level2_points"]]
    assert actual_l2 == expected_l2, (actual_l2, expected_l2)

    # First five 4/4 measures reproduce the dissertation recursive placement:
    # m1.3,m1.4,m2.1,m2.2,m2.3,m3.1,m3.2,m3.3,m4.1,m5.1,m5.2,m5.3.
    m5 = make_measures(5)
    r5 = v6.analyze_levels12(m5, [v4.Attack(Fraction(i)) for i in range(20)])
    expected5 = [Fraction(x) for x in (2,3,4,5,6,8,9,10,12,16,17,18)]
    assert [Fraction(p["time_quarter"]) for p in r5["level2_points"]] == expected5

    # Real regression assets when present: Buxtehude and BWV 661. No Level-1
    # output may change, and every attack label in L1/L2 must resolve to an exact
    # OMR slot (no proportional fallback for attacks).
    base = Path("/mnt/data") if Path("/mnt/data").exists() else Path(".")
    for stem, pdf_name in (("Durch_Aamds_Buxtehude", "Durch_Aamds_Buxtehude.pdf"), ("bwv661-a4", "bwv661-a4.pdf")):
        mxl = base / f"{stem}.mxl"
        omr = base / f"{stem}.omr"
        pdf = base / pdf_name
        if not (mxl.exists() and omr.exists() and pdf.exists()):
            continue
        measures_real, attacks_real = v4.parse_score(mxl, (4, 4))
        before = v4.analyze_level1(measures_real, attacks_real)
        rr = v6.analyze_levels12(measures_real, attacks_real)
        assert rr["points"] == before["points"]
        pages, diagnostics = v6.render_original_pdf_with_levels12_exact(pdf, omr, measures_real, rr)
        assert len(pages) >= 1
        for page_diag in diagnostics:
            for system in page_diag["systems"]:
                assert system["levels"]["1"]["attack_labels_without_exact_slot"] == 0
                assert system["levels"]["2"]["attack_labels_without_exact_slot"] == 0

    print("level1-level2-clean-v6-validation: PASS")


if __name__ == "__main__":
    main()
