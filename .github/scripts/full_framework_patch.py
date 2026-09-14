from pathlib import Path

p = Path('tone_metric/canonical_score.py')
s = p.read_text()
old = 'from .models import Hit\n'
new = 'from .models import Hit, MeasureInfo\n'
assert old in s
s = s.replace(old, new, 1)

marker = '\ndef recover_canonical_columns(omr_path: str | Path, measures, cluster_tolerance_px: float = 12.0) -> tuple[list[CanonicalColumn], dict]:\n'
assert marker in s
helper = r'''

class CanonicalFrameworkError(RuntimeError):
    """Raised when OMR and MusicXML cannot be reconciled without guessing."""


def _omr_measure_stack_count(omr_path: str | Path) -> int:
    path = Path(omr_path)
    count = 0
    with ZipFile(path, "r") as zf:
        members = sorted(
            [n for n in zf.namelist() if re.search(r"sheet#\d+/sheet#\d+\.xml$", n, re.I)],
            key=_sheet_no,
        )
        for member in members:
            root = etree.fromstring(zf.read(member))
            for system in [e for e in root.iter() if _local(e.tag) == "system"]:
                count += sum(1 for child in system if _local(child.tag) == "stack")
    return count


def _numeric_measure_number(value) -> int | None:
    m = re.fullmatch(r"\s*(-?\d+)\s*", str(value or ""))
    return int(m.group(1)) if m else None


def _reconcile_measure_framework_with_omr(omr_path, measures, symbolic_hits=None) -> tuple[dict, list[str]]:
    """Expand sparse MusicXML measure metadata to the finalized OMR stack order.

    Audiveris can retain all physical measure stacks in the finalized OMR while
    omitting individual measures from MusicXML.  When the surviving MusicXML
    numbers unambiguously span the complete OMR score, restore only the missing
    measure shells and keep known MusicXML measures at their true physical ordinal.
    Meter is inherited across a gap only when both known neighbors agree.
    """
    original = list(measures)
    symbolic_hits = list(symbolic_hits or [])
    omr_count = _omr_measure_stack_count(omr_path)
    info = {
        "musicxml_measure_count": len(original),
        "omr_measure_stack_count": omr_count,
        "framework_measure_count": len(original),
        "framework_reconciled": False,
        "synthesized_measure_count": 0,
        "synthesized_measure_numbers": [],
        "framework_alignment": "musicxml-sequential",
    }
    warnings: list[str] = []
    if omr_count <= len(original):
        if omr_count < len(original):
            warnings.append(
                f"Finalized OMR contains {omr_count} measure stacks but MusicXML contains {len(original)} measures; no OMR expansion was required."
            )
        return info, warnings

    numbers = [_numeric_measure_number(m.number) for m in original]
    if not original or any(n is None for n in numbers):
        raise CanonicalFrameworkError(
            f"Finalized OMR contains {omr_count} measure stacks but MusicXML contains only {len(original)} measures, and MusicXML numbering is not a complete numeric ordinal map."
        )
    nums = [int(n) for n in numbers]
    if any(b <= a for a, b in zip(nums, nums[1:])):
        raise CanonicalFrameworkError(
            "Sparse MusicXML measure numbers are not strictly increasing, so omitted OMR measures cannot be restored without guessing."
        )
    base_number = nums[0]
    target_indices = [n - base_number for n in nums]
    if target_indices[0] != 0 or target_indices[-1] != omr_count - 1:
        raise CanonicalFrameworkError(
            f"MusicXML measure numbering spans {nums[0]}–{nums[-1]}, which does not unambiguously cover all {omr_count} finalized OMR measure stacks."
        )
    if len(set(target_indices)) != len(target_indices) or any(i < 0 or i >= omr_count for i in target_indices):
        raise CanonicalFrameworkError(
            "MusicXML measure numbers cannot be mapped one-to-one onto the finalized OMR measure stacks."
        )

    expanded: list[MeasureInfo | None] = [None] * omr_count
    old_to_new: dict[int, int] = {}
    for old_index, (measure, target_index) in enumerate(zip(original, target_indices)):
        expanded[target_index] = measure
        old_to_new[old_index] = target_index

    missing_indices = [i for i, measure in enumerate(expanded) if measure is None]
    for i in missing_indices:
        prev = next((expanded[j] for j in range(i - 1, -1, -1) if expanded[j] is not None), None)
        nxt = next((expanded[j] for j in range(i + 1, omr_count) if expanded[j] is not None), None)
        if prev is None or nxt is None:
            raise CanonicalFrameworkError(
                f"OMR measure {base_number + i} is missing from MusicXML at an unbounded score edge; its meter cannot be restored safely."
            )
        prev_meter = (int(prev.numerator), int(prev.denominator))
        next_meter = (int(nxt.numerator), int(nxt.denominator))
        if prev_meter != next_meter:
            raise CanonicalFrameworkError(
                f"OMR measure {base_number + i} is missing from MusicXML between conflicting meters {prev_meter[0]}/{prev_meter[1]} and {next_meter[0]}/{next_meter[1]}; analysis will not guess the meter change."
            )
        numerator, denominator = prev_meter
        full = Fraction(numerator * 4, denominator)
        expanded[i] = MeasureInfo(
            index=i,
            number=str(base_number + i),
            start=Fraction(0),
            full_duration=full,
            actual_duration=full,
            pickup_shift=Fraction(0),
            numerator=numerator,
            denominator=denominator,
            implicit=False,
        )

    restored = [m for m in expanded if m is not None]
    if len(restored) != omr_count:
        raise CanonicalFrameworkError(
            "Internal framework reconciliation failed to restore every finalized OMR measure stack."
        )

    start = Fraction(0)
    for i, measure in enumerate(restored):
        measure.index = i
        measure.start = start
        start += measure.full_duration

    for hit in symbolic_hits:
        old_index = int(hit.measure_index)
        if old_index not in old_to_new:
            raise CanonicalFrameworkError(
                f"Symbolic hit references unknown MusicXML measure index {old_index} during OMR reconciliation."
            )
        new_index = old_to_new[old_index]
        measure = restored[new_index]
        hit.measure_index = new_index
        hit.onset = measure.start + hit.offset_in_measure
        for src in hit.sources:
            src.measure_index = new_index
            src.onset = measure.start + src.offset_in_measure

    measures[:] = restored
    synthesized_numbers = [str(base_number + i) for i in missing_indices]
    info.update({
        "framework_measure_count": len(restored),
        "framework_reconciled": True,
        "synthesized_measure_count": len(missing_indices),
        "synthesized_measure_numbers": synthesized_numbers,
        "framework_alignment": "finalized-omr-stack-order+musicxml-number-anchors",
    })
    warnings.append(
        f"Restored {len(missing_indices)} MusicXML-omitted measure shells from the finalized OMR stack order; no meter was guessed across a conflicting change."
    )
    return info, warnings
'''
s = s.replace(marker, helper + marker, 1)

old = '''def build_hits_from_canonical_score(omr_path, measures, symbolic_hits=None):
    """Return global sonic attacks on the canonical score-time grid.

    Symbolic MusicXML note details are attached only when they agree exactly with the
    recovered measure/time.  They are never allowed to move a visual attack.
    """
    columns, meta = recover_canonical_columns(omr_path, measures)
    symbolic_hits = list(symbolic_hits or [])
'''
new = '''def build_hits_from_canonical_score(omr_path, measures, symbolic_hits=None):
    """Return global sonic attacks on the canonical score-time grid.

    Symbolic MusicXML note details are attached only when they agree exactly with the
    recovered measure/time.  They are never allowed to move a visual attack.
    """
    symbolic_hits = list(symbolic_hits or [])
    framework_meta, framework_warnings = _reconcile_measure_framework_with_omr(
        omr_path, measures, symbolic_hits=symbolic_hits
    )
    columns, meta = recover_canonical_columns(omr_path, measures)
'''
assert old in s
s = s.replace(old, new, 1)

old = '''    meta = dict(meta)
    meta["canonical_hit_count"] = len(hits)
    meta["symbolic_hit_count"] = len(symbolic_hits)
    meta["canonical_attack_keys"] = [f"{h.measure_index}:{h.offset_in_measure}" for h in hits]
    warnings = []
    if not hits:
        warnings.append("Canonical score-time recovery produced no usable attacks; symbolic MusicXML hits were retained.")
    return hits, warnings, meta
'''
new = '''    meta = dict(meta)
    meta.update(framework_meta)
    meta["measure_count"] = len(meta.get("measures", []))
    meta["canonical_hit_count"] = len(hits)
    meta["symbolic_hit_count"] = len(symbolic_hits)
    meta["canonical_attack_keys"] = [f"{h.measure_index}:{h.offset_in_measure}" for h in hits]
    warnings = list(framework_warnings)
    if not hits:
        warnings.append("Canonical score-time recovery produced no usable attacks.")
    return hits, warnings, meta
'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)

p = Path('app.py')
s = p.read_text()
old = 'from tone_metric.canonical_score import build_hits_from_canonical_score\n'
new = 'from tone_metric.canonical_score import build_hits_from_canonical_score, CanonicalFrameworkError\n'
assert old in s
s = s.replace(old, new, 1)
old = '''        if suffix == ".pdf" and omr_path is not None:
            try:
                recovered_hits, canonical_warnings, canonical_score_meta = build_hits_from_canonical_score(
                    omr_path, measures, symbolic_hits=hits
                )
                if recovered_hits:
                    analysis_hits = recovered_hits
                    parse_warnings = list(parse_warnings) + list(canonical_warnings)
            except Exception as exc:
                parse_warnings = list(parse_warnings) + [
                    f"Canonical score-time recovery failed; symbolic MusicXML hits were used: {exc}"
                ]
'''
new = '''        if suffix == ".pdf" and omr_path is not None:
            try:
                recovered_hits, canonical_warnings, canonical_score_meta = build_hits_from_canonical_score(
                    omr_path, measures, symbolic_hits=hits
                )
            except CanonicalFrameworkError as exc:
                raise HTTPException(422, f"Canonical score framework could not be reconciled safely: {exc}")
            except Exception as exc:
                raise HTTPException(422, f"Canonical score-time recovery failed: {exc}")
            if not recovered_hits:
                raise HTTPException(422, "Canonical score-time recovery produced no usable attacks.")
            analysis_hits = recovered_hits
            parse_warnings = list(parse_warnings) + list(canonical_warnings)
'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
