from __future__ import annotations

"""Dissertation-faithful tone-metric tree derivation.

Trees are a read-only derivative of the already validated tone-metric wave/grid.
This module never reconstructs notation, musical time, recursive Levels, pivots,
or PDF coordinates.

The dissertation's Chapter 4 tree rules are implemented literally:

1. each *sonic event* is placed at the lowest occupied tone-metric Level on its
   column;
2. branches proceed left-to-right and may connect equal Levels, or a lower-Level
   event to a higher-Level event, but never a higher-Level event to a lower one;
3. more than one branch may leave an event, but at most one branch may arrive at
   any event;
4. when more than one earlier event could legally connect to the same event, the
   closest earlier event is privileged.

Parenthetical structural positions are not sonic events and therefore are not
Tree nodes.  Simultaneous attacks have already been merged upstream into one
canonical attack event.
"""

from collections import defaultdict
from fractions import Fraction


def _frac(value, default=None):
    try:
        return Fraction(str(value))
    except Exception:
        return default


def _point_key(point: dict) -> tuple[int, Fraction] | None:
    off = _frac(point.get("offset_in_measure_quarter"))
    if off is None:
        return None
    try:
        mi = int(point.get("measure_index", -1))
    except Exception:
        return None
    return mi, off


def _levels(point: dict) -> list[int]:
    out = []
    for raw in point.get("levels", []) or []:
        try:
            level = int(raw)
        except Exception:
            continue
        if level > 0:
            out.append(level)
    return sorted(set(out))


def build_tree_profile(wave_profile: list[dict]) -> dict:
    """Build tree nodes and directed branches in score time.

    Only wave points carrying an actual attack become nodes.  Each node is
    placed at its *lowest* occupied recursive Level.  For every later node, the
    nearest earlier node whose lowest Level is less than or equal to the target
    Level is selected as its single parent.  This is the deterministic form of
    the dissertation's one-incoming-branch + closest-eligible-branch rule.
    """
    by_segment: dict[int, list[dict]] = defaultdict(list)
    for point in wave_profile:
        if not bool(point.get("attack")):
            continue
        levels = _levels(point)
        if not levels:
            continue
        try:
            segment_index = int(point.get("segment_index", 0))
        except Exception:
            segment_index = 0
        key = _point_key(point)
        onset = _frac(point.get("onset_quarter"))
        if key is None or onset is None:
            continue
        by_segment[segment_index].append({
            "segment_index": segment_index,
            "measure_index": int(point.get("measure_index", -1)),
            "measure_number": point.get("measure_number"),
            "offset_in_measure_quarter": point.get("offset_in_measure_quarter"),
            "onset_quarter": point.get("onset_quarter"),
            "levels": levels,
            "lowest_level": min(levels),
            "highest_level": max(levels),
            "attack": True,
            "tree_source": "actual-wave-event-lowest-recursive-level",
            "_time": onset,
        })

    nodes: list[dict] = []
    branches: list[dict] = []
    branch_index = 0
    for segment_index in sorted(by_segment):
        points = sorted(
            by_segment[segment_index],
            key=lambda p: (
                p["_time"],
                int(p["measure_index"]),
                _frac(p["offset_in_measure_quarter"], Fraction(0)),
            ),
        )
        segment_nodes = []
        for p in points:
            node = {k: v for k, v in p.items() if k != "_time"}
            node["node_index"] = len(nodes)
            nodes.append(node)
            segment_nodes.append(node)

        for j, target in enumerate(segment_nodes):
            if j == 0:
                continue
            target_level = int(target["lowest_level"])
            source = None
            # Walking backward makes the first legal node the closest earlier
            # event in musical time.  No geometry enters this decision.
            for candidate in reversed(segment_nodes[:j]):
                if int(candidate["lowest_level"]) <= target_level:
                    source = candidate
                    break
            if source is None:
                continue
            source_time = _frac(source["onset_quarter"], Fraction(0))
            target_time = _frac(target["onset_quarter"], Fraction(0))
            branches.append({
                "branch_index": branch_index,
                "segment_index": segment_index,
                "source_node_index": int(source["node_index"]),
                "target_node_index": int(target["node_index"]),
                "source_measure_index": int(source["measure_index"]),
                "source_measure_number": source.get("measure_number"),
                "source_offset_in_measure_quarter": source.get("offset_in_measure_quarter"),
                "source_onset_quarter": source.get("onset_quarter"),
                "source_level": int(source["lowest_level"]),
                "target_measure_index": int(target["measure_index"]),
                "target_measure_number": target.get("measure_number"),
                "target_offset_in_measure_quarter": target.get("offset_in_measure_quarter"),
                "target_onset_quarter": target.get("onset_quarter"),
                "target_level": int(target["lowest_level"]),
                "distance_quarter": str(target_time - source_time),
                "same_level": int(source["lowest_level"]) == int(target["lowest_level"]),
                "ascending_level": int(source["lowest_level"]) < int(target["lowest_level"]),
                "tree_source": "closest-earlier-eligible-event",
            })
            branch_index += 1

    incoming = defaultdict(int)
    outgoing = defaultdict(int)
    for b in branches:
        incoming[int(b["target_node_index"])] += 1
        outgoing[int(b["source_node_index"])] += 1
    for n in nodes:
        ni = int(n["node_index"])
        n["incoming_branches"] = int(incoming.get(ni, 0))
        n["outgoing_branches"] = int(outgoing.get(ni, 0))
        n["root"] = incoming.get(ni, 0) == 0

    return {
        "nodes": nodes,
        "branches": branches,
        "tree_source": "dissertation-lowest-level-left-to-right-closest-eligible",
    }


def register_tree_profile(
    tree_profile: dict,
    wave_anchors_by_page: dict[int, list[dict]],
) -> tuple[dict[int, list[dict]], dict[int, list[dict]], dict, list[str]]:
    """Register tree nodes/branches only to existing actual-attack wave anchors.

    No event x-coordinate is estimated.  Same-system branches retain both exact
    event endpoints.  Cross-system/page branches are emitted as two endpoint
    pieces so the relation remains visible without inventing an intermediate
    musical coordinate.
    """
    lookup = {}
    duplicates = []
    for page, rows in wave_anchors_by_page.items():
        for row in rows:
            if not bool(row.get("attack")):
                continue
            key = _point_key(row)
            if key is None:
                continue
            if key in lookup:
                duplicates.append(key)
                continue
            lookup[key] = (int(page), row)

    nodes_by_page: dict[int, list[dict]] = defaultdict(list)
    mapped_nodes: dict[int, dict] = {}
    missing_nodes = []
    for node in tree_profile.get("nodes", []) or []:
        key = _point_key(node)
        if key is None:
            missing_nodes.append({**node, "reason": "invalid-tree-node-key"})
            continue
        found = lookup.get(key)
        if found is None:
            missing_nodes.append({**node, "reason": "existing-actual-wave-anchor-not-found"})
            continue
        page, anchor = found
        row = {
            **node,
            "page_index": int(anchor.get("page_index", page)),
            "system_index": int(anchor.get("system_index", 0)),
            "physical_system_index": int(anchor.get("physical_system_index", anchor.get("system_index", 0))),
            "cx_norm": float(anchor.get("cx_norm")),
            "cy_norm": float(anchor.get("cy_norm", 0.0)),
            "recovered_x_abs": float(anchor.get("recovered_x_abs")),
            "registration_source": "existing-actual-wave-anchor-only",
            "anchor_registration_source": anchor.get("registration_source"),
            "tree_coordinate_synthesis": False,
        }
        ni = int(node["node_index"])
        mapped_nodes[ni] = row
        nodes_by_page[row["page_index"]].append(row)

    branches_by_page: dict[int, list[dict]] = defaultdict(list)
    missing_branches = []
    cross_system = 0
    mapped_branches = 0
    for branch in tree_profile.get("branches", []) or []:
        source = mapped_nodes.get(int(branch.get("source_node_index", -1)))
        target = mapped_nodes.get(int(branch.get("target_node_index", -1)))
        if source is None or target is None:
            missing_branches.append({
                **branch,
                "reason": "registered-tree-node-endpoint-missing",
                "missing_source": source is None,
                "missing_target": target is None,
            })
            continue
        source_page = int(source["page_index"])
        target_page = int(target["page_index"])
        source_system = int(source["system_index"])
        target_system = int(target["system_index"])
        common = {
            **branch,
            "registration_source": "existing-tree-node-wave-anchors-only",
            "tree_coordinate_synthesis": False,
            "source_page_index": source_page,
            "source_system_index": source_system,
            "source_cx_norm": float(source["cx_norm"]),
            "source_x_abs": float(source["recovered_x_abs"]),
            "target_page_index": target_page,
            "target_system_index": target_system,
            "target_cx_norm": float(target["cx_norm"]),
            "target_x_abs": float(target["recovered_x_abs"]),
        }
        if source_page == target_page and source_system == target_system:
            branches_by_page[source_page].append({
                **common,
                "page_index": source_page,
                "system_index": source_system,
                "span_role": "complete",
            })
        else:
            cross_system += 1
            branches_by_page[source_page].append({
                **common,
                "page_index": source_page,
                "system_index": source_system,
                "span_role": "source-endpoint",
                "cx_norm": float(source["cx_norm"]),
                "endpoint_level": int(branch["source_level"]),
            })
            branches_by_page[target_page].append({
                **common,
                "page_index": target_page,
                "system_index": target_system,
                "span_role": "target-endpoint",
                "cx_norm": float(target["cx_norm"]),
                "endpoint_level": int(branch["target_level"]),
            })
        mapped_branches += 1

    for rows in nodes_by_page.values():
        rows.sort(key=lambda r: (
            int(r.get("system_index", 0)),
            _frac(r.get("onset_quarter"), Fraction(0)),
            int(r.get("node_index", -1)),
        ))
    for rows in branches_by_page.values():
        rows.sort(key=lambda r: (
            int(r.get("system_index", 0)),
            int(r.get("branch_index", -1)),
            str(r.get("span_role", "")),
        ))

    warnings = []
    if duplicates:
        warnings.append(f"{len(duplicates)} duplicate actual wave-anchor key(s) were found while registering tree nodes.")
    if missing_nodes:
        warnings.append(f"{len(missing_nodes)} tree event node(s) could not be matched to an existing actual wave anchor.")
    if missing_branches:
        warnings.append(f"{len(missing_branches)} tree branch(es) could not be registered because an endpoint node was unavailable.")

    nodes = tree_profile.get("nodes", []) or []
    branches = tree_profile.get("branches", []) or []
    root_count = sum(bool(n.get("root")) for n in nodes)
    stats = {
        "tree_event_nodes_expected": len(nodes),
        "tree_event_nodes_mapped": len(mapped_nodes),
        "tree_event_nodes_missing": len(missing_nodes),
        "tree_branches_expected": len(branches),
        "tree_branches_mapped": mapped_branches,
        "tree_branches_missing": len(missing_branches),
        "tree_roots": int(root_count),
        "tree_cross_system_branches": int(cross_system),
        "tree_visual_pieces": sum(len(v) for v in branches_by_page.values()),
        "tree_registration": "score-time-actual-event-tree->existing-actual-wave-anchors-only",
        "tree_coordinate_synthesis": False,
        "tree_missing_nodes": missing_nodes,
        "tree_missing_branches": missing_branches,
        "tree_duplicate_actual_wave_anchor_keys": [f"{mi}:{off}" for mi, off in duplicates],
    }
    return dict(nodes_by_page), dict(branches_by_page), stats, warnings
