from __future__ import annotations

from collections import defaultdict

from hit_model import (
    Chord,
    GlobalOnset,
    LocalOnset,
    GAP_COST,
    KNOWN_MATCH_BASE_COST,
    SAME_STAFF_COLLAPSE_INTERLINES,
    UNKNOWN_MATCH_BASE_COST,
    UNKNOWN_MATCH_MAX_INTERLINES,
)


def _same_staff_local_sequence(chords: list[Chord], interline: float) -> list[LocalOnset]:
    """Create one ordered onset sequence for a staff.

    Exact semantic BEGIN time groups simultaneous voices on the same staff.
    Untimed recognized chords remain independent attacks unless an actual
    notehead is essentially coincident with a neighboring onset.
    """
    known: dict[object, list[Chord]] = defaultdict(list)
    untimed: list[Chord] = []
    for chord in chords:
        if chord.symbolic_time is None:
            untimed.append(chord)
        else:
            known[chord.symbolic_time].append(chord)

    items = [LocalOnset(group[0].staff, group, time) for time, group in known.items()]
    items.extend(LocalOnset(chord.staff, [chord], None) for chord in untimed)
    items.sort(key=lambda item: item.visual_x)

    tol = SAME_STAFF_COLLAPSE_INTERLINES * interline
    out: list[LocalOnset] = []
    for item in items:
        if not out:
            out.append(item)
            continue
        prev = out[-1]
        if item.symbolic_time is None or prev.symbolic_time is None:
            d = min(abs(a - b) for a in item.attack_head_xs for b in prev.attack_head_xs)
            if d <= tol:
                if item.symbolic_time is not None and prev.symbolic_time is not None and item.symbolic_time != prev.symbolic_time:
                    out.append(item)
                else:
                    time = prev.symbolic_time if prev.symbolic_time is not None else item.symbolic_time
                    out[-1] = LocalOnset(prev.staff, prev.chords + item.chords, time)
                    continue
        out.append(item)
    return out


def _match_cost(global_onset: GlobalOnset, local_onset: LocalOnset, interline: float) -> float | None:
    known = global_onset.known_times
    time = local_onset.symbolic_time
    dx = abs(global_onset.visual_x - local_onset.visual_x)

    if known and time is not None:
        if time not in known:
            return None
        return KNOWN_MATCH_BASE_COST + min(dx / max(4.0 * interline, 1e-9), 0.80)

    tolerance = UNKNOWN_MATCH_MAX_INTERLINES * interline
    if dx > tolerance:
        return None
    return UNKNOWN_MATCH_BASE_COST + dx / max(tolerance, 1e-9)


def _align_staff_sequence(global_seq: list[GlobalOnset], staff_seq: list[LocalOnset], interline: float) -> list[GlobalOnset]:
    """One-to-one, order-preserving staff-sequence alignment."""
    n, m = len(global_seq), len(staff_seq)
    inf = 10**9
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[str, int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(n + 1):
        for j in range(m + 1):
            current = dp[i][j]
            if current >= inf:
                continue
            if i < n and current + GAP_COST < dp[i + 1][j]:
                dp[i + 1][j] = current + GAP_COST
                back[i + 1][j] = ('global', i, j)
            if j < m and current + GAP_COST < dp[i][j + 1]:
                dp[i][j + 1] = current + GAP_COST
                back[i][j + 1] = ('staff', i, j)
            if i < n and j < m:
                cost = _match_cost(global_seq[i], staff_seq[j], interline)
                if cost is not None and current + cost < dp[i + 1][j + 1]:
                    dp[i + 1][j + 1] = current + cost
                    back[i + 1][j + 1] = ('match', i, j)

    if back[n][m] is None and (n or m):
        raise ValueError('Staff-sequence alignment failed')

    operations: list[tuple[str, int, int]] = []
    i, j = n, m
    while i or j:
        step = back[i][j]
        if step is None:
            raise ValueError('Incomplete staff-sequence alignment')
        operations.append(step)
        _, i, j = step
    operations.reverse()

    out: list[GlobalOnset] = []
    gi = si = 0
    for operation, _i, _j in operations:
        if operation == 'match':
            out.append(GlobalOnset(global_seq[gi].items + [staff_seq[si]]))
            gi += 1
            si += 1
        elif operation == 'global':
            out.append(global_seq[gi])
            gi += 1
        else:
            out.append(GlobalOnset([staff_seq[si]]))
            si += 1
    return out


def _actual_strike_x(event: GlobalOnset) -> float:
    heads = event.attack_head_xs
    if not heads:
        raise ValueError('Global onset has no attacking notehead')
    center = event.visual_x
    return min(heads, key=lambda x: (abs(x - center), x))
