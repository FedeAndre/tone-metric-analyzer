from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional


def frac_to_str(x: Fraction) -> str:
    return f"{x.numerator}/{x.denominator}" if x.denominator != 1 else str(x.numerator)


@dataclass
class NoteAttack:
    onset: Fraction
    duration: Fraction
    measure_index: int
    measure_number: str
    offset_in_measure: Fraction
    part_id: str
    voice: str
    staff: str
    pitch: str
    tie_start: bool = False
    tie_stop: bool = False
    page_index: Optional[int] = None
    x_norm: Optional[float] = None
    y_norm: Optional[float] = None


@dataclass
class Hit:
    onset: Fraction
    duration: Fraction
    measure_index: int
    measure_number: str
    offset_in_measure: Fraction
    sources: List[NoteAttack] = field(default_factory=list)

    def to_dict(self):
        return {
            "onset_quarter": frac_to_str(self.onset),
            "duration_quarter": frac_to_str(self.duration),
            "measure_index": self.measure_index,
            "measure_number": self.measure_number,
            "offset_in_measure_quarter": frac_to_str(self.offset_in_measure),
            "attack_key": f"{self.measure_index}:{frac_to_str(self.offset_in_measure)}",
            "source_count": len(self.sources),
            "pitches": sorted({s.pitch for s in self.sources}),
            "source_notes": [
                {
                    "pitch": s.pitch,
                    "part_id": s.part_id,
                    "voice": s.voice,
                    "staff": s.staff,
                    "duration_quarter": frac_to_str(s.duration),
                    "page_index": s.page_index,
                    "x_norm": s.x_norm,
                    "y_norm": s.y_norm,
                }
                for s in self.sources
            ],
        }


@dataclass
class MeasureInfo:
    index: int
    number: str
    start: Fraction
    full_duration: Fraction
    actual_duration: Fraction
    pickup_shift: Fraction
    numerator: int
    denominator: int
    implicit: bool = False

    @property
    def end(self) -> Fraction:
        return self.start + self.full_duration


@dataclass
class MeterSegment:
    start: Fraction
    end: Fraction
    start_measure_index: int
    end_measure_index: int
    numerator: int
    denominator: int
    beat_unit: Fraction
    beat_count: int
    top_base: int
    compound: bool
    hits: List[Hit] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
