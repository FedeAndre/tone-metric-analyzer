from __future__ import annotations

from collections import Counter, defaultdict

import dissertation_system_aligned_app as aligned

# Presentation-only refinement on top of the validated current analyzer.
# No level, wave, pivot, tree, parser, or timing calculation is changed here.
current = aligned.current
old = aligned.old
base = aligned.base
app = aligned.app
APP_VERSION = "1.2.2-dissertation-system-aligned"
app.version = APP_VERSION

_OBSOLETE_LAYOUT_WARNING = (
    "The MusicXML export did not provide usable note layout coordinates, "
    "so coloring on the original PDF may be unavailable or approximate for this score."
)

# The exact-PDF view now uses saved Audiveris OMR geometry rather than MusicXML
# note layout, so this legacy warning is no longer applicable to PDF visualization.
_original_warnings = base._warnings

def _warnings_without_obsolete_layout_message(result: dict, parser_warnings: list[str]) -> list[str]:
    return [w for w in _original_warnings(result, parser_warnings) if str(w).strip() != _OBSOLETE_LAYOUT_WARNING]

base._warnings = _warnings_without_obsolete_layout_message


def _chronological_key(row: dict) -> tuple:
    try:
        event = int(row.get("event_index", -1))
    except Exception:
        event = -1
    try:
        measure = int(row.get("measure_index", -1))
    except Exception:
        measure = -1
    try:
        onset = float(row.get("onset_quarter", 0) or 0)
    except Exception:
        onset = 0.0
    return (measure, event, onset)


def _recover_visual_systems(page: dict) -> None:
    """Recover score-line membership when OMR exported one collapsed system id.

    The musical analysis is untouched. This function only re-labels the already
    registered visual anchors so each printed score line receives its own graphic
    band. New score lines are detected by the leftward x reset in chronological
    attack order, then all other visual objects are attached to the same line by
    measure/event identity or nearest physical y position.
    """
    layers = list(page.get("layer_anchors", []) or [])
    if len(layers) < 2:
        return

    existing = {int(r.get("system_index", 0)) for r in layers}
    if len(existing) > 1:
        return

    ordered = sorted(layers, key=_chronological_key)
    groups: list[list[dict]] = [[]]
    prev_x = None
    prev_y = None
    for row in ordered:
        try:
            x = float(row.get("cx_norm", 0.0))
        except Exception:
            x = 0.0
        try:
            y = float(row.get("cy_norm", 0.0))
        except Exception:
            y = 0.0

        new_line = False
        if prev_x is not None:
            # Primary signal: the engraving cursor returns from the right side of
            # the page to the left side at a system break.
            if (prev_x - x) >= 0.24 and prev_x >= 0.50 and x <= 0.62:
                new_line = True
            # Secondary signal for short systems / sparse scores.
            elif prev_y is not None and abs(y - prev_y) >= 0.075 and (prev_x - x) >= 0.10:
                new_line = True
        if new_line and groups[-1]:
            groups.append([])
        groups[-1].append(row)
        prev_x, prev_y = x, y

    groups = [g for g in groups if g]
    if len(groups) <= 1:
        return

    event_to_system: dict[int, int] = {}
    measure_votes: dict[int, Counter] = defaultdict(Counter)
    system_centers: dict[int, float] = {}

    for sys_idx, rows in enumerate(groups):
        ys = []
        for row in rows:
            row["system_index"] = sys_idx
            row["physical_system_index"] = sys_idx
            try:
                event_to_system[int(row.get("event_index", -1))] = sys_idx
            except Exception:
                pass
            try:
                measure_votes[int(row.get("measure_index", -1))][sys_idx] += 1
            except Exception:
                pass
            try:
                y = float(row.get("cy_norm", 0.0))
                if 0.0 < y < 1.0:
                    ys.append(y)
            except Exception:
                pass
        system_centers[sys_idx] = sum(ys) / len(ys) if ys else (sys_idx + 0.5) / len(groups)

    measure_to_system = {
        measure: votes.most_common(1)[0][0]
        for measure, votes in measure_votes.items()
        if votes
    }

    def choose_system(row: dict) -> int:
        try:
            event = int(row.get("event_index", -1))
            if event in event_to_system:
                return event_to_system[event]
        except Exception:
            pass
        try:
            measure = int(row.get("measure_index", -1))
            if measure in measure_to_system:
                return measure_to_system[measure]
        except Exception:
            pass
        try:
            y = float(row.get("cy_norm", 0.0))
            if 0.0 < y < 1.0:
                return min(system_centers, key=lambda k: abs(system_centers[k] - y))
        except Exception:
            pass
        return 0

    for key in ("structural_anchors", "wave_anchors", "pivot_anchors", "tree_nodes", "tree_branches"):
        for row in page.get(key, []) or []:
            sys_idx = choose_system(row)
            row["system_index"] = sys_idx
            if "physical_system_index" in row:
                row["physical_system_index"] = sys_idx

    # Rebuild display bounds from the now-separated score lines. These bounds are
    # presentation geometry only; they do not feed back into the analysis.
    bounds = []
    all_rows = []
    for key in ("layer_anchors", "wave_anchors", "tree_nodes"):
        all_rows.extend(page.get(key, []) or [])
    for sys_idx in range(len(groups)):
        rows = [r for r in all_rows if int(r.get("system_index", 0)) == sys_idx]
        xs, ys = [], []
        for row in rows:
            try:
                x = float(row.get("cx_norm", 0.0))
                if 0.0 <= x <= 1.0:
                    xs.append(x)
            except Exception:
                pass
            try:
                y = float(row.get("cy_norm", 0.0))
                if 0.0 < y < 1.0:
                    ys.append(y)
            except Exception:
                pass
        if not ys:
            center = system_centers[sys_idx]
            ys = [center]
        top = max(0.0, min(ys) - 0.028)
        bottom = min(1.0, max(ys) + 0.028)
        left = max(0.0, min(xs) - 0.015) if xs else 0.04
        right = min(1.0, max(xs) + 0.015) if xs else 0.96
        bounds.append({
            "system_index": sys_idx,
            "top_norm": top,
            "bottom_norm": bottom,
            "guard_bottom_norm": bottom,
            "left_norm": left,
            "right_norm": right,
        })

    bounds.sort(key=lambda r: r["top_norm"])
    for i, row in enumerate(bounds[:-1]):
        row["guard_bottom_norm"] = (row["bottom_norm"] + bounds[i + 1]["top_norm"]) / 2.0
    page["system_bounds"] = bounds


_original_canonical_overlay = current._canonical_overlay

def _canonical_overlay_per_score_line(levels_payload: dict, canonical_meta: dict) -> dict:
    overlay = _original_canonical_overlay(levels_payload, canonical_meta)
    for page in overlay.get("pages", []) or []:
        _recover_visual_systems(page)
    matching = dict(overlay.get("matching", {}) or {})
    matching["presentation_system_alignment"] = "chronological-x-reset-per-printed-score-line"
    overlay["matching"] = matching
    return overlay

current._canonical_overlay = _canonical_overlay_per_score_line
