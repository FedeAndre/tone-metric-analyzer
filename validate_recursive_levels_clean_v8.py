from __future__ import annotations

import ast
from fractions import Fraction
from pathlib import Path

import recursive_levels_clean_v8 as v8


def make_measures(count: int, meter=(4, 4)):
    measure_len = Fraction(meter[0] * 4, meter[1])
    return [
        v8.Measure(i, str(i + 1), i * measure_len, (i + 1) * measure_len, Fraction(0), meter)
        for i in range(count)
    ]


def times(points):
    return [Fraction(p["time_quarter"]) for p in points]


def isolation_audit():
    source = Path("recursive_levels_clean_v8.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    forbidden = (
        "level1_clean_", "level1_level2_", "recursive_levels_clean_v7",
        "tone_metric", "dissertation_", "dissertation_app", "app",
    )
    offenders = [name for name in imported if any(name.startswith(prefix) for prefix in forbidden)]
    assert offenders == [], offenders
    assert "_interpolated_x" not in source
    assert "v4." not in source and "v5." not in source and "v6." not in source and "v7." not in source
    assert "strict-exact-omr-slot-no-fallback" in source


def analytical_regression():
    assert v8.sequence(2, 34)[:7] == [1, 2, 3, 5, 9, 17, 33]
    assert v8.sequence(3, 29)[:5] == [1, 2, 4, 10, 28]

    measures = make_measures(3)
    attacks = [v8.Attack(Fraction(i)) for i in range(12)]
    result = v8.analyze_recursive_levels(measures, attacks)
    assert times(result["level1_points"]) == [Fraction(x) for x in (0, 1, 2, 4, 8)]
    assert times(result["level2_points"]) == [Fraction(x) for x in (2, 3, 4, 5, 6, 8, 9, 10)]
    assert result["highest_level"] == 3
    assert times(result["level3_points"]) == [Fraction(x) for x in (6, 7, 8, 10, 11)]
    covered = set()
    for pts in result["levels"].values():
        covered.update(times(pts))
    assert covered == {Fraction(i) for i in range(12)}

    measures5 = make_measures(5)
    attacks5 = [v8.Attack(Fraction(i)) for i in range(20)]
    r5 = v8.analyze_recursive_levels(measures5, attacks5)
    assert r5["highest_level"] == 4
    assert times(r5["level3_points"]) == [Fraction(x) for x in (6,7,8,10,11,12,13,14,16,18,19)]
    assert times(r5["level4_points"]) == [Fraction(x) for x in (14,15,16)]

    m44 = make_measures(2, (4, 4))
    m34 = [
        v8.Measure(2, "3", Fraction(8), Fraction(11), Fraction(0), (3, 4)),
        v8.Measure(3, "4", Fraction(11), Fraction(14), Fraction(0), (3, 4)),
    ]
    mixed = m44 + m34
    rm = v8.analyze_recursive_levels(mixed, [v8.Attack(Fraction(i)) for i in range(14)])
    assert rm["recursive_levels_complete"] is True
    for si, required in v8._all_tactus_times(mixed).items():
        got = set()
        for pts in rm["levels"].values():
            got.update(
                Fraction(p["time_quarter"])
                for p in pts
                if int(p["segment_index"]) == si
            )
        assert required <= got


def real_asset_regression():
    base = Path("/mnt/data") if Path("/mnt/data").exists() else Path(".")
    tested = 0
    for stem, pdf_name in (
        ("Durch_Aamds_Buxtehude", "Durch_Aamds_Buxtehude.pdf"),
        ("bwv661-a4", "bwv661-a4.pdf"),
    ):
        mxl = base / f"{stem}.mxl"
        omr = base / f"{stem}.omr"
        pdf = base / pdf_name
        if not (mxl.exists() and omr.exists() and pdf.exists()):
            continue
        measures, attacks = v8.parse_score(mxl, (4, 4))
        result = v8.analyze_recursive_levels(measures, attacks)
        pages, diagnostics = v8.render_original_pdf_with_recursive_levels_exact(pdf, omr, measures, result)
        assert pages
        for page_diag in diagnostics:
            for system in page_diag["systems"]:
                for lev_diag in system["levels"].values():
                    assert lev_diag["missing_attack_slots"] == 0
        tested += 1
    return tested


def main():
    isolation_audit()
    analytical_regression()
    tested = real_asset_regression()
    print(f"recursive-levels-clean-v8-validation: PASS (real_assets={tested})")


if __name__ == "__main__":
    main()
