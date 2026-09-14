from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import level1_clean_v4 as v4
import level1_clean_v5 as v5
import level1_level2_clean_v6 as v6
import recursive_levels_clean_v7 as v7


def make_measures(count: int, meter=(4, 4)):
    beat_measure = Fraction(meter[0] * 4, meter[1])
    return [
        v4.Measure(
            i,
            str(i + 1),
            i * beat_measure,
            (i + 1) * beat_measure,
            Fraction(0),
            meter,
        )
        for i in range(count)
    ]


def times(points):
    return [Fraction(p["time_quarter"]) for p in points]


def main():
    # Strict non-regression: the frozen prior analytical and registration objects
    # are still the same functions used by the earlier validated builds.
    assert v7.v6.core.analyze_level1 is v4.analyze_level1
    assert v7.v6.analyze_level2 is v6.analyze_level2
    assert v7.v5._nearest_slot_x is v5._nearest_slot_x
    assert v7.v5._interpolated_x is v5._interpolated_x
    assert v7.v5.parse_omr_geometry is v5.parse_omr_geometry

    # Dissertation Examples 1.1-1.3: 3 measures of 4/4, all quarter-note attacks.
    measures = make_measures(3)
    attacks = [v4.Attack(Fraction(i)) for i in range(12)]
    frozen12 = v6.analyze_levels12(measures, attacks)
    result = v7.analyze_recursive_levels(measures, attacks)

    # L1 and L2 must be exactly unchanged as Python data.
    assert result["level1_points"] == frozen12["level1_points"]
    assert result["level2_points"] == frozen12["level2_points"]
    assert result["points"] == frozen12["points"]

    # Dissertation Level 1: quarter positions 1,2,3,5,9 -> times 0,1,2,4,8.
    assert times(result["level1_points"]) == [Fraction(x) for x in (0, 1, 2, 4, 8)]

    # Dissertation Level 2 / Example 1.2 remains frozen.
    assert times(result["level2_points"]) == [Fraction(x) for x in (2, 3, 4, 5, 6, 8, 9, 10)]

    # Dissertation Example 1.3: Level 3 maps quarter notes 7,8,9 and 11,12.
    # Zero-based quarter times are therefore 6,7,8 and 10,11.
    assert result["highest_level"] == 3
    assert times(result["level3_points"]) == [Fraction(x) for x in (6, 7, 8, 10, 11)]

    # Every tactus position in the 12-quarter excerpt must now be covered.
    covered = set()
    for pts in result["levels"].values():
        covered.update(times(pts))
    assert covered == {Fraction(i) for i in range(12)}
    assert result["recursive_levels_complete"] is True

    # Longer 5-measure synthetic excerpt must continue recursively as needed,
    # while preserving the exact v6 L1/L2 output.
    measures5 = make_measures(5)
    attacks5 = [v4.Attack(Fraction(i)) for i in range(20)]
    frozen5 = v6.analyze_levels12(measures5, attacks5)
    r5 = v7.analyze_recursive_levels(measures5, attacks5)
    assert r5["level1_points"] == frozen5["level1_points"]
    assert r5["level2_points"] == frozen5["level2_points"]
    assert r5["highest_level"] == 4
    assert times(r5["level3_points"]) == [Fraction(x) for x in (6,7,8,10,11,12,13,14,16,18,19)]
    assert times(r5["level4_points"]) == [Fraction(x) for x in (14,15,16)]
    covered5 = set()
    for pts in r5["levels"].values():
        covered5.update(times(pts))
    assert covered5 == {Fraction(i) for i in range(20)}

    # Meter changes reset the recursive construction independently per segment.
    m44 = make_measures(2, (4, 4))
    # Reindex/start a following 3/4 segment manually.
    m34 = [
        v4.Measure(2, "3", Fraction(8), Fraction(11), Fraction(0), (3, 4)),
        v4.Measure(3, "4", Fraction(11), Fraction(14), Fraction(0), (3, 4)),
    ]
    mixed = m44 + m34
    mixed_attacks = [v4.Attack(Fraction(i)) for i in range(14)]
    rm = v7.analyze_recursive_levels(mixed, mixed_attacks)
    assert rm["recursive_levels_complete"] is True
    for si, all_times in v7._all_tactus_times(mixed).items():
        covered_si = set()
        for pts in rm["levels"].values():
            covered_si.update(
                Fraction(p["time_quarter"])
                for p in pts
                if int(p["segment_index"]) == si
            )
        assert all_times <= covered_si

    # Real regression assets when present: Buxtehude and BWV 661.
    # L1/L2 must remain identical and every attack label in every generated
    # tactus level must resolve to an exact OMR slot.
    base = Path("/mnt/data") if Path("/mnt/data").exists() else Path(".")
    for stem, pdf_name in (
        ("Durch_Aamds_Buxtehude", "Durch_Aamds_Buxtehude.pdf"),
        ("bwv661-a4", "bwv661-a4.pdf"),
    ):
        mxl = base / f"{stem}.mxl"
        omr = base / f"{stem}.omr"
        pdf = base / pdf_name
        if not (mxl.exists() and omr.exists() and pdf.exists()):
            continue
        measures_real, attacks_real = v4.parse_score(mxl, (4, 4))
        before12 = v6.analyze_levels12(measures_real, attacks_real)
        rr = v7.analyze_recursive_levels(measures_real, attacks_real)
        assert rr["level1_points"] == before12["level1_points"]
        assert rr["level2_points"] == before12["level2_points"]
        pages, diagnostics = v7.render_original_pdf_with_recursive_levels_exact(
            pdf, omr, measures_real, rr
        )
        assert len(pages) >= 1
        for page_diag in diagnostics:
            for system in page_diag["systems"]:
                for lev_diag in system["levels"].values():
                    assert lev_diag["attack_labels_without_exact_slot"] == 0

    print("recursive-levels-clean-v7-validation: PASS")


if __name__ == "__main__":
    main()
