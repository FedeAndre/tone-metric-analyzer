from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from statistics import median

UNKNOWN_MATCH_MAX_INTERLINES = 1.50
SAME_STAFF_COLLAPSE_INTERLINES = 0.35
GAP_COST = 1.0
KNOWN_MATCH_BASE_COST = 0.05
UNKNOWN_MATCH_BASE_COST = 0.15


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0


@dataclass
class Chord:
    id: str
    staff: str
    measure_index: int
    attack_head_xs: tuple[float, ...]
    box: Box
    visual_x: float
    symbolic_time: Fraction | None


@dataclass
class LocalOnset:
    staff: str
    chords: list[Chord]
    symbolic_time: Fraction | None

    @property
    def visual_x(self) -> float:
        return float(median([c.visual_x for c in self.chords]))

    @property
    def attack_head_xs(self) -> list[float]:
        return [x for c in self.chords for x in c.attack_head_xs]


@dataclass
class GlobalOnset:
    items: list[LocalOnset] = field(default_factory=list)

    @property
    def visual_x(self) -> float:
        return float(median([item.visual_x for item in self.items]))

    @property
    def known_times(self) -> set[Fraction]:
        return {item.symbolic_time for item in self.items if item.symbolic_time is not None}

    @property
    def attack_head_xs(self) -> list[float]:
        return [x for item in self.items for x in item.attack_head_xs]


@dataclass(frozen=True)
class Strike:
    page_index: int
    system_index: int
    measure_index: int
    x: float
    system_top: float
    system_bottom: float
    omr_width: float
    omr_height: float
    recovered: bool = False


@dataclass(frozen=True)
class Diagnostics:
    global_hits: int
    sounding_chords: int
    timed_chords: int
    untimed_chords: int
    synthetic_strike_positions: int
    sequence_alignment: bool
