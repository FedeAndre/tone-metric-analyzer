from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
import pickle
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import cv2
import fitz
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

from oemer import MODULE_PATH


CHECKPOINTS = {
    "unet_big/model.onnx": "https://github.com/BreezeWhite/oemer/releases/download/checkpoints/1st_model.onnx",
    "seg_net/model.onnx": "https://github.com/BreezeWhite/oemer/releases/download/checkpoints/2nd_model.onnx",
}


def _trim_process_memory() -> None:
    """
    Release Python objects and return freed native heap pages to the OS.

    ONNX Runtime repeatedly allocates large convolution workspaces. Even after
    a session is destroyed, glibc may retain those pages in the process heap.
    Small Railway containers then begin the next page with the previous page's
    high-water mark still resident. malloc_trim is a general process-memory
    cleanup, not a score-specific workaround.
    """
    gc.collect()
    if os.name != "posix":
        return
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass


@dataclass
class Staff:
    id: int
    lines_y: list[float]
    spacing: float
    top: float
    bottom: float


@dataclass
class SystemRegion:
    id: int
    staff_ids: list[int]
    top: float
    bottom: float
    left: float
    right: float


@dataclass
class Barline:
    id: int
    system_id: int
    x: float
    top: float
    bottom: float
    line_count: int = 1


@dataclass
class MeasureRegion:
    id: int
    system_id: int
    measure_in_system: int
    left: float
    right: float
    top: float
    bottom: float


@dataclass
class Notehead:
    id: int
    x1: int
    y1: int
    x2: int
    y2: int
    cx: float
    cy: float
    staff_id: int | None
    staff_pos_halfspaces: int | None
    fill_ratio: float
    head_type: str
    system_id: int | None = None
    measure_local: int | None = None
    stem_id: int | None = None
    stem_source: str | None = None
    dot_id: int | None = None


@dataclass
class Stem:
    id: int
    x1: int
    y1: int
    x2: int
    y2: int
    cx: float
    cy: float
    height: float
    width: float
    staff_id: int | None
    system_id: int | None = None
    measure_local: int | None = None
    direction: str | None = None


@dataclass
class Beam:
    id: int
    points: list[list[float]]
    cx: float
    cy: float
    length: float
    thickness: float
    stem_ids: list[int]


@dataclass
class Rest:
    id: int
    x1: int
    y1: int
    x2: int
    y2: int
    cx: float
    cy: float
    staff_id: int | None
    rest_type: str
    classifier_margin: float
    system_id: int | None = None
    measure_local: int | None = None
    dot_id: int | None = None


@dataclass
class Dot:
    id: int
    cx: float
    cy: float
    area: int
    notehead_id: int | None = None


@dataclass
class TieCandidate:
    id: int
    left_notehead_id: int
    right_notehead_id: int
    side: str
    confidence: float


def ensure_checkpoints() -> None:
    for rel, url in CHECKPOINTS.items():
        dst = Path(MODULE_PATH) / "checkpoints" / rel
        if dst.exists() and dst.stat().st_size > 100000:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading optical checkpoint: {dst.name}")
        urllib.request.urlretrieve(url, dst)


def render_pdf(pdf_path: Path, out_dir: Path, dpi: int = 300) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    pages = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat, alpha=False, colorspace=fitz.csGRAY)
        p = out_dir / f"page_{i+1:02d}.png"
        pix.save(str(p))
        pages.append(p)
    return pages


def _resize_for_optical_model(image: np.ndarray) -> np.ndarray:
    """
    Match the pretrained segmentation model's intended page scale without
    retaining duplicate PIL/cv2 copies. The target is the geometric midpoint
    of the model's documented 3.0M-4.35M pixel operating range.
    """
    h, w = image.shape[:2]
    pixels = h * w
    if 3_000_000 <= pixels <= 4_350_000:
        return image
    target_pixels = (3_000_000 + 4_350_000) / 2.0
    ratio = math.sqrt(target_pixels / float(max(1, pixels)))
    target_w = max(1, int(round(w * ratio)))
    target_h = max(1, int(round(h * ratio)))
    print(f"Optical model raster: {target_w} {target_h}")
    return cv2.resize(
        image,
        (target_w, target_h),
        interpolation=cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_CUBIC,
    )


def _low_memory_inference(
    model_dir: Path,
    image_bgr: np.ndarray,
    step_size: int = 192,
) -> np.ndarray:
    """
    Run the pretrained low-level segmentation network with a strict memory bound.

    The upstream helper retains every patch prediction and also builds a full
    floating-point page probability tensor. Here each patch is classified
    immediately and merged by a center-priority overlap mask. Only two uint8
    page maps plus one 256x256 prediction are resident, while a 64-pixel overlap
    prevents hard tile seams.

    No semantic OMR, rhythm, voice, slot, or timing information is involved.
    """
    metadata_path = model_dir / "metadata.pkl"
    model_path = model_dir / "model.onnx"
    with metadata_path.open("rb") as fh:
        metadata = pickle.load(fh)

    input_shape = metadata["input_shape"]
    win_size = int(input_shape[1])

    image = _resize_for_optical_model(image_bgr)
    h, w = image.shape[:2]

    def positions(length: int) -> list[int]:
        out: list[int] = []
        for value in range(0, length, step_size):
            pos = value
            if pos + win_size > length:
                pos = length - win_size
            if not out or pos != out[-1]:
                out.append(pos)
        return out

    y_positions = positions(h)
    x_positions = positions(w)

    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    class_map = np.zeros((h, w), dtype=np.uint8)
    best_weight = np.zeros((h, w), dtype=np.uint8)

    # Center-priority blend mask. A pixel is taken from whichever overlapping
    # tile sees it furthest from a tile edge.
    yy, xx = np.indices((win_size, win_size))
    edge_distance = np.minimum.reduce(
        [
            yy + 1,
            xx + 1,
            win_size - yy,
            win_size - xx,
        ]
    )
    patch_weight = np.clip(edge_distance, 1, 255).astype(np.uint8)

    total = len(y_positions) * len(x_positions)
    index = 0
    for y in y_positions:
        for x in x_positions:
            index += 1
            if index == 1 or index % 20 == 0 or index == total:
                print(f"Optical patch {index}/{total}")

            patch = np.ascontiguousarray(
                image[y:y + win_size, x:x + win_size]
            )
            batch = patch[np.newaxis, ...]
            pred = session.run(
                [output_name],
                {input_name: batch},
            )[0][0]
            labels = np.argmax(pred, axis=-1).astype(np.uint8)

            region_weight = best_weight[
                y:y + win_size,
                x:x + win_size,
            ]
            take = patch_weight > region_weight
            region_class = class_map[
                y:y + win_size,
                x:x + win_size,
            ]
            region_class[take] = labels[take]
            region_weight[take] = patch_weight[take]

            del pred, labels, batch, patch, take

    del best_weight, patch_weight, edge_distance, session, image
    _trim_process_memory()
    return class_map

def run_segmentation(
    img_path: Path,
    cache_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path) as z:
            return (
                z["gray"], z["staff"], z["symbols"],
                z["stems_rests"], z["noteheads"], z["clefs_keys"],
            )

    source_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if source_bgr is None:
        raise RuntimeError(f"Cannot load {img_path}")

    first = _low_memory_inference(
        Path(MODULE_PATH) / "checkpoints" / "unet_big",
        source_bgr,
    )
    staff = (first == 1).astype(np.uint8)
    symbols = (first == 2).astype(np.uint8)
    del first
    _trim_process_memory()

    second = _low_memory_inference(
        Path(MODULE_PATH) / "checkpoints" / "seg_net",
        source_bgr,
    )
    stems_rests = (second == 1).astype(np.uint8)
    noteheads = (second == 2).astype(np.uint8)
    clefs_keys = (second == 3).astype(np.uint8)
    del second
    _trim_process_memory()

    src = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)
    src = cv2.resize(
        src,
        (staff.shape[1], staff.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    del source_bgr
    _trim_process_memory()

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            gray=src,
            staff=staff,
            symbols=symbols,
            stems_rests=stems_rests,
            noteheads=noteheads,
            clefs_keys=clefs_keys,
        )
    return src, staff, symbols, stems_rests, noteheads, clefs_keys

def estimate_page_skew(gray: np.ndarray) -> float:
    """Estimate global staff-line skew from long near-horizontal raster segments."""
    h, w = gray.shape
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 1800.0,
        threshold=max(50, int(round(w * 0.045))),
        minLineLength=max(80, int(round(w * 0.25))),
        maxLineGap=max(8, int(round(w * 0.03))),
    )
    if lines is None:
        return 0.0

    weighted: list[tuple[float, float]] = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = [int(v) for v in line]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        length = math.hypot(dx, dy)
        if abs(angle) <= 3.0 and length >= 0.25 * w:
            weighted.append((angle, length))
    if not weighted:
        return 0.0

    weighted.sort(key=lambda z: z[0])
    half = 0.5 * sum(weight for _, weight in weighted)
    acc = 0.0
    for angle, weight in weighted:
        acc += weight
        if acc >= half:
            return float(angle)
    return float(weighted[-1][0])


def deskew_layers(
    gray: np.ndarray,
    staff: np.ndarray,
    symbols: np.ndarray,
    stems_rests: np.ndarray,
    noteheads: np.ndarray,
    clefs_keys: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    angle = estimate_page_skew(gray)
    if abs(angle) < 0.08:
        return angle, gray, staff, symbols, stems_rests, noteheads, clefs_keys

    h, w = gray.shape
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)

    def rot_gray(a: np.ndarray) -> np.ndarray:
        return cv2.warpAffine(
            a, matrix, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )

    def rot_mask(a: np.ndarray) -> np.ndarray:
        return cv2.warpAffine(
            a, matrix, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.uint8)

    return (
        angle,
        rot_gray(gray),
        rot_mask(staff),
        rot_mask(symbols),
        rot_mask(stems_rests),
        rot_mask(noteheads),
        rot_mask(clefs_keys),
    )


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    runs = []
    a = b = int(idx[0])
    for q in idx[1:]:
        q = int(q)
        if q == b + 1:
            b = q
        else:
            runs.append((a, b))
            a = b = q
    runs.append((a, b))
    return runs


def _staves_from_line_mask(line_mask: np.ndarray, threshold_fraction: float) -> list[Staff]:
    h, w = line_mask.shape
    proj = line_mask.sum(axis=1).astype(np.float32)
    smooth = np.convolve(proj, np.ones(3, dtype=np.float32) / 3.0, mode="same")
    threshold = max(10.0, float(w) * threshold_fraction)
    rows = smooth >= threshold
    centers = [0.5 * (a + b) for a, b in contiguous_runs(rows)]

    merged = []
    for center in centers:
        if merged and center - merged[-1] <= 3.0:
            merged[-1] = 0.5 * (merged[-1] + center)
        else:
            merged.append(center)

    # A page can contain stray long rules/text underlines. Five-line periodicity,
    # rather than an assumed staff count, determines valid staves.
    out: list[Staff] = []
    used_until = -1
    for i in range(max(0, len(merged) - 4)):
        window = merged[i:i+5]
        gaps = np.diff(window)
        med = float(np.median(gaps))
        if not (5.0 <= med <= 45.0):
            continue
        if np.max(np.abs(gaps - med)) > max(3.0, 0.32 * med):
            continue
        if window[0] <= used_until:
            continue
        sid = len(out)
        out.append(Staff(
            id=sid,
            lines_y=[round(float(x), 3) for x in window],
            spacing=med,
            top=float(window[0] - 2.6 * med),
            bottom=float(window[-1] + 2.6 * med),
        ))
        used_until = window[-1] + 0.5 * med
    return out


def detect_staves(gray: np.ndarray, staff_mask: np.ndarray) -> list[Staff]:
    # Two independent visual observations of the same page are evaluated:
    # (1) the neural staff-line segmentation and (2) long horizontal ink in the
    # original raster. No semantic OMR structure is used.
    neural = _staves_from_line_mask(staff_mask, threshold_fraction=0.022)

    ink = (gray < 185).astype(np.uint8)
    w = gray.shape[1]
    kernel_w = max(35, int(round(w * 0.045)))
    horizontal = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, np.ones((1, kernel_w), np.uint8)
    )
    horizontal = cv2.dilate(horizontal, np.ones((1, 5), np.uint8))
    raster = _staves_from_line_mask(horizontal, threshold_fraction=0.050)

    # Select the internally more complete periodic staff interpretation. This is
    # page-agnostic and uses no expected staff count.
    if len(raster) > len(neural):
        return raster
    return neural


def nearest_staff(y: float, staves: list[Staff]) -> Staff | None:
    valid = [s for s in staves if s.top <= y <= s.bottom]
    if valid:
        return min(valid, key=lambda s: abs(y - np.mean(s.lines_y)))
    if not staves:
        return None
    s = min(staves, key=lambda q: abs(y - np.mean(q.lines_y)))
    if abs(y - np.mean(s.lines_y)) <= 4.0 * s.spacing:
        return s
    return None


def global_spacing(staves: list[Staff]) -> float:
    vals = [s.spacing for s in staves if s.spacing > 0]
    return float(np.median(vals)) if vals else 18.0


def detect_systems(
    gray: np.ndarray,
    staves: list[Staff],
) -> list[SystemRegion]:
    """
    Group staves into systems from repeated inter-staff spacing and the actual
    horizontal staff-line span. No expected number of staves/system is used.
    """
    if not staves:
        return []

    ordered = sorted(staves, key=lambda s: float(np.mean(s.lines_y)))
    if len(ordered) == 1:
        groups = [ordered]
    else:
        centers = np.array(
            [float(np.mean(s.lines_y)) for s in ordered],
            dtype=float,
        )
        gaps = np.diff(centers)
        med = float(np.median(gaps))
        mad = float(np.median(np.abs(gaps - med)))
        sp = float(np.median([s.spacing for s in ordered]))
        break_threshold = med + max(2.0 * mad, 1.5 * sp)

        groups: list[list[Staff]] = []
        current = [ordered[0]]
        for i, gap in enumerate(gaps):
            if float(gap) > break_threshold:
                groups.append(current)
                current = [ordered[i + 1]]
            else:
                current.append(ordered[i + 1])
        groups.append(current)

    ink = (gray < 185).astype(np.uint8)
    sp_global = global_spacing(staves)
    horizontal = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        np.ones((1, max(20, int(round(3.0 * sp_global)))), np.uint8),
    )

    systems: list[SystemRegion] = []
    for group in groups:
        xs: list[int] = []
        for staff in group:
            for y in staff.lines_y:
                yi = int(round(y))
                y0 = max(0, yi - 2)
                y1 = min(gray.shape[0], yi + 3)
                cols = np.flatnonzero(
                    np.any(horizontal[y0:y1, :] > 0, axis=0)
                )
                if len(cols):
                    xs.extend(int(x) for x in cols)

        left = float(min(xs)) if xs else 0.0
        right = float(max(xs)) if xs else float(gray.shape[1] - 1)
        sp = float(np.median([s.spacing for s in group]))
        systems.append(
            SystemRegion(
                id=len(systems),
                staff_ids=[s.id for s in group],
                top=max(0.0, group[0].lines_y[0] - sp),
                bottom=min(
                    float(gray.shape[0] - 1),
                    group[-1].lines_y[-1] + sp,
                ),
                left=left,
                right=right,
            )
        )
    return systems


def _merge_close_x(values: list[float], tolerance: float) -> list[tuple[float, int]]:
    if not values:
        return []
    values = sorted(values)
    groups: list[list[float]] = [[values[0]]]
    for value in values[1:]:
        if value - groups[-1][-1] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [
        (float(np.mean(group)), len(group))
        for group in groups
    ]


def _longest_ink_run(column: np.ndarray) -> int:
    data = cv2.morphologyEx(
        (column > 0).astype(np.uint8).reshape(-1, 1),
        cv2.MORPH_CLOSE,
        np.ones((3, 1), np.uint8),
    ).ravel()
    best = 0
    current = 0
    for value in data:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def detect_barlines_and_measures(
    gray: np.ndarray,
    staves: list[Staff],
    systems: list[SystemRegion],
) -> tuple[list[Barline], list[MeasureRegion]]:
    """
    Detect measure boundaries as nearly continuous vertical raster strokes
    spanning a system. This rejects coincident note stems on multiple staves:
    summed vertical ink is insufficient unless a single continuous run crosses
    most of the system height.
    """
    if not systems:
        return [], []

    sp = global_spacing(staves)
    ink = (gray < 185).astype(np.uint8)
    ink = cv2.dilate(ink, np.ones((1, 2), np.uint8))

    barlines: list[Barline] = []
    measures: list[MeasureRegion] = []

    for system in systems:
        top = max(0, int(round(system.top)))
        bottom = min(gray.shape[0], int(round(system.bottom)))
        height = max(1, bottom - top)
        region = ink[top:bottom, :]

        candidates: list[float] = []
        for x in range(region.shape[1]):
            run = _longest_ink_run(region[:, x])
            if run >= 0.72 * height:
                candidates.append(float(x))

        # Convert adjacent ink columns into line centers, then merge double/final
        # barlines into one temporal boundary while retaining line_count.
        runs = contiguous_runs(
            np.isin(
                np.arange(gray.shape[1]),
                np.array(candidates, dtype=int),
            )
        )
        line_centers = [
            (a + b) / 2.0
            for a, b in runs
            if (b - a + 1) <= 1.2 * sp
        ]
        merged = _merge_close_x(line_centers, 0.75 * sp)

        # At a continuation system the left boundary can be represented by the
        # system edge/brace rather than a full vertical barline. Add that edge
        # only when the first detected internal bar is well to its right.
        if not merged or merged[0][0] - system.left > 6.0 * sp:
            merged.insert(0, (system.left, 1))

        if system.right - merged[-1][0] > 6.0 * sp:
            merged.append((system.right, 1))

        # Restrict to the observed staff span and enforce strictly increasing
        # temporal boundaries.
        merged = [
            item for item in merged
            if system.left - sp <= item[0] <= system.right + sp
        ]
        merged.sort(key=lambda item: item[0])

        sys_bars: list[Barline] = []
        for x, line_count in merged:
            bar = Barline(
                id=len(barlines),
                system_id=system.id,
                x=float(x),
                top=float(top),
                bottom=float(bottom),
                line_count=line_count,
            )
            barlines.append(bar)
            sys_bars.append(bar)

        for mi, (left_bar, right_bar) in enumerate(
            zip(sys_bars, sys_bars[1:]),
            start=1,
        ):
            if right_bar.x - left_bar.x < 2.0 * sp:
                continue
            measures.append(
                MeasureRegion(
                    id=len(measures) + 1,
                    system_id=system.id,
                    measure_in_system=mi,
                    left=left_bar.x,
                    right=right_bar.x,
                    top=float(top),
                    bottom=float(bottom),
                )
            )

    return barlines, measures


def assign_regions(
    noteheads: list[Notehead],
    stems: list[Stem],
    systems: list[SystemRegion],
    measures: list[MeasureRegion],
) -> None:
    staff_to_system = {
        staff_id: system.id
        for system in systems
        for staff_id in system.staff_ids
    }
    by_system: dict[int, list[MeasureRegion]] = {}
    for measure in measures:
        by_system.setdefault(measure.system_id, []).append(measure)

    def locate(staff_id: int | None, x: float) -> tuple[int | None, int | None]:
        system_id = staff_to_system.get(staff_id) if staff_id is not None else None
        if system_id is None:
            return None, None
        for measure in by_system.get(system_id, []):
            if measure.left <= x <= measure.right:
                return system_id, measure.id
        return system_id, None

    for nh in noteheads:
        nh.system_id, nh.measure_local = locate(nh.staff_id, nh.cx)
    for stem in stems:
        stem.system_id, stem.measure_local = locate(stem.staff_id, stem.cx)


def comp_stats(binary: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    n, labels, stats, cent = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[i]]
        out.append((x, y, w, h, area))
    return out


def estimate_head_fill(gray: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x, y, w, h = box
    pad_x = max(1, int(round(0.10 * w)))
    pad_y = max(1, int(round(0.10 * h)))
    x1, x2 = max(0, x + pad_x), min(gray.shape[1], x + w - pad_x)
    y1, y2 = max(0, y + pad_y), min(gray.shape[0], y + h - pad_y)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = gray[y1:y2, x1:x2]
    return float(np.mean(crop < 155))


def _adjust_notehead_bbox(
    box: tuple[int, int, int, int],
    note_mask: np.ndarray,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(note_mask.shape[1], x2)
    y2 = min(note_mask.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    region = note_mask[y1:y2, x1:x2]
    ys, xs = np.where(region > 0)
    if len(xs) == 0:
        return None
    return (
        max(0, x1 + int(xs.min()) - 1),
        max(0, y1 + int(ys.min()) - 1),
        min(note_mask.shape[1], x1 + int(xs.max()) + 2),
        min(note_mask.shape[0], y1 + int(ys.max()) + 2),
    )


def _split_notehead_bbox(
    box: tuple[int, int, int, int],
    note_mask: np.ndarray,
    spacing: float,
    depth: int = 0,
) -> list[tuple[int, int, int, int]]:
    x1, y1, x2, y2 = [int(v) for v in box]
    w = x2 - x1
    h = y2 - y1
    expected_w = 1.285714 * spacing
    expected_h = spacing

    # Adjacent chord heads can merge in the segmentation mask. Split only when
    # the component is substantially larger than a single engraved notehead.
    if depth < 4 and w > 1.65 * expected_w:
        n = max(2, int(round(w / expected_w)))
        pieces: list[tuple[int, int, int, int]] = []
        for i in range(n):
            a = round(x1 + i * w / n)
            b = round(x1 + (i + 1) * w / n)
            adjusted = _adjust_notehead_bbox((a, y1, b, y2), note_mask)
            if adjusted is not None:
                pieces.extend(
                    _split_notehead_bbox(adjusted, note_mask, spacing, depth + 1)
                )
        if pieces:
            return pieces

    if depth < 4 and h > 1.55 * expected_h:
        n = max(2, int(round(h / expected_h)))
        pieces = []
        for i in range(n):
            a = round(y1 + i * h / n)
            b = round(y1 + (i + 1) * h / n)
            adjusted = _adjust_notehead_bbox((x1, a, x2, b), note_mask)
            if adjusted is not None:
                pieces.append(adjusted)
        if pieces:
            return pieces

    return [(x1, y1, x2, y2)]


def detect_noteheads(
    gray: np.ndarray,
    note_mask: np.ndarray,
    staves: list[Staff],
) -> list[Notehead]:
    """
    Extract visual noteheads from the low-level notehead mask.

    This deliberately does not use pitch, voice, duration, MusicXML, OMR slots,
    or semantic timing. The morphology is based only on expected notehead
    geometry relative to locally observed staff spacing.
    """
    sp = global_spacing(staves)

    # Preserve oval noteheads while removing thin staff/stem/beam fragments that
    # leak into the neural mask.
    small = max(2, int(round(sp / 3.0)))
    small_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (small, small)
    )
    cleaned = cv2.erode(
        cv2.dilate(note_mask.astype(np.uint8), small_kernel),
        small_kernel,
    )

    morph_size = (
        max(2, int(round(sp * 0.50))),
        max(2, int(round(sp * 0.40))),
    )
    head_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_size)
    head_core = cv2.erode(cleaned, head_kernel)
    restore_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (morph_size[0] + 1, morph_size[1] + 1),
    )
    head_regions = cv2.dilate(head_core, restore_kernel)

    boxes: list[tuple[int, int, int, int]] = []
    for x, y, w, h, _area in comp_stats(head_regions):
        boxes.extend(
            _split_notehead_bbox(
                (x, y, x + w, y + h),
                note_mask,
                sp,
            )
        )

    out: list[Notehead] = []
    for x1, y1, x2, y2 in boxes:
        w = x2 - x1
        h = y2 - y1
        if not (0.45 * sp <= w <= 2.20 * sp):
            continue
        if not (0.45 * sp <= h <= 1.65 * sp):
            continue

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        staff = nearest_staff(cy, staves)
        if staff is None:
            continue

        # The morphology already supplies the strong shape test. Retain only
        # boxes with direct support in the model's low-level notehead pixels.
        region = note_mask[y1:y2, x1:x2]
        if region.size == 0 or float(np.mean(region > 0)) < 0.10:
            continue

        fill = estimate_head_fill(gray, (x1, y1, w, h))
        head_type = "filled" if fill >= 0.62 else "hollow"
        bottom = staff.lines_y[-1]
        pos = int(round((bottom - cy) / (staff.spacing / 2.0)))

        out.append(
            Notehead(
                id=len(out),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                cx=cx,
                cy=cy,
                staff_id=staff.id,
                staff_pos_halfspaces=pos,
                fill_ratio=round(fill, 4),
                head_type=head_type,
            )
        )
    return out

def vertical_components(stems_rests: np.ndarray, staves: list[Staff]) -> list[Stem]:
    sp = global_spacing(staves)
    kh = max(3, int(round(1.15 * sp)))
    vertical = cv2.morphologyEx(stems_rests, cv2.MORPH_OPEN, np.ones((kh,1), np.uint8))
    out = []
    for x,y,w,h,area in comp_stats(vertical):
        if h < 1.25*sp or h > 7.5*sp:
            continue
        if w > 0.55*sp:
            continue
        cx, cy = x+w/2.0, y+h/2.0
        staff = nearest_staff(cy, staves)
        out.append(Stem(
            id=len(out), x1=x, y1=y, x2=x+w, y2=y+h, cx=cx, cy=cy,
            height=float(h), width=float(w), staff_id=None if staff is None else staff.id
        ))
    return out


def _longest_vertical_run(column: np.ndarray) -> tuple[int, int | None, int | None]:
    if column.size == 0:
        return 0, None, None
    data = cv2.morphologyEx(
        column.astype(np.uint8).reshape(-1, 1),
        cv2.MORPH_CLOSE,
        np.ones((3, 1), np.uint8),
    ).ravel()
    best_len = 0
    best_a = None
    best_b = None
    start = None
    for i, value in enumerate(np.r_[data, 0]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - start > best_len:
                best_len = i - start
                best_a = start
                best_b = i - 1
            start = None
    return best_len, best_a, best_b


def _recover_pixel_stem(
    nh: Notehead,
    vertical_ink: np.ndarray,
    spacing: float,
) -> tuple[str, int, int, int] | None:
    """
    Recover a stem directly from raster ink adjoining a notehead.

    The search is local to the two notehead edges and therefore cannot use
    inferred voice/timing information. A candidate must overlap the notehead
    vertically and extend by more than one staff spacing in a valid stem
    direction.
    """
    h, w = vertical_ink.shape
    best: tuple[float, str, int, int, int] | None = None

    for edge_x in (nh.x1, nh.x2):
        xa = max(0, int(round(edge_x - 0.50 * spacing)))
        xb = min(w, int(round(edge_x + 0.50 * spacing)) + 1)

        for direction in ("up", "down"):
            if direction == "up":
                ya = max(0, int(round(nh.cy - 4.7 * spacing)))
                yb = min(h, int(round(nh.cy + 0.45 * spacing)) + 1)
            else:
                ya = max(0, int(round(nh.cy - 0.45 * spacing)))
                yb = min(h, int(round(nh.cy + 4.7 * spacing)) + 1)

            region = vertical_ink[ya:yb, xa:xb]
            for j in range(region.shape[1]):
                run, a, b = _longest_vertical_run(region[:, j])
                if a is None or b is None:
                    continue
                if run < 1.15 * spacing or run > 5.15 * spacing:
                    continue

                gy1 = ya + a
                gy2 = ya + b
                if not (
                    gy1 <= nh.cy + 0.60 * spacing
                    and gy2 >= nh.cy - 0.60 * spacing
                ):
                    continue
                if direction == "up" and gy1 >= nh.cy - 0.75 * spacing:
                    continue
                if direction == "down" and gy2 <= nh.cy + 0.75 * spacing:
                    continue

                x = xa + j
                edge_distance = min(abs(x - nh.x1), abs(x - nh.x2))
                score = float(run) - 1.5 * float(edge_distance)
                candidate = (score, direction, x, gy1, gy2)
                if best is None or candidate[0] > best[0]:
                    best = candidate

    if best is None:
        return None
    _score, direction, x, y1, y2 = best
    return direction, x, y1, y2


def link_noteheads_stems(
    noteheads: list[Notehead],
    stems: list[Stem],
    staves: list[Staff],
    gray: np.ndarray,
) -> tuple[int, int]:
    """
    Link noteheads to stems using two independent visual paths.

    First use globally detected vertical stem components. Only heads still
    unlinked are tested against vertical ink in the original raster. Recovered
    stems are materialized as normal Stem objects; there is no fallback timing
    or semantic OMR path.
    """
    sp = global_spacing(staves)
    component_links = 0

    for nh in noteheads:
        candidates = []
        for st in stems:
            if (
                nh.staff_id is not None
                and st.staff_id is not None
                and nh.staff_id != st.staff_id
            ):
                continue
            dx = min(
                abs(st.cx - nh.x1),
                abs(st.cx - nh.x2),
                abs(st.cx - nh.cx),
            )
            y_ok = (
                st.y1 <= nh.cy + 0.55 * sp
                and st.y2 >= nh.cy - 0.55 * sp
            )
            if y_ok and dx <= 0.58 * sp:
                candidates.append((dx, abs(st.cy - nh.cy), st))

        if candidates:
            _, _, st = min(
                candidates,
                key=lambda z: (z[0], z[1], z[2].id),
            )
            nh.stem_id = st.id
            nh.stem_source = "component"
            component_links += 1
            if st.cy < nh.cy:
                st.direction = "up"
            elif st.cy > nh.cy:
                st.direction = "down"

    # Independent raster evidence for noteheads whose stem was fragmented or
    # omitted by the low-level stem segmentation.
    ink = (gray < 175).astype(np.uint8)
    vertical_ink = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        np.ones((max(3, int(round(1.05 * sp))), 1), np.uint8),
    )
    vertical_ink = cv2.dilate(
        vertical_ink,
        np.ones((1, 2), np.uint8),
    )

    pixel_links = 0
    for nh in noteheads:
        if nh.stem_id is not None:
            continue
        recovered = _recover_pixel_stem(nh, vertical_ink, sp)
        if recovered is None:
            continue

        direction, x, y1, y2 = recovered
        staff = nearest_staff((y1 + y2) / 2.0, staves)
        st = Stem(
            id=len(stems),
            x1=x,
            y1=y1,
            x2=x + 1,
            y2=y2 + 1,
            cx=x + 0.5,
            cy=(y1 + y2 + 1) / 2.0,
            height=float(y2 - y1 + 1),
            width=1.0,
            staff_id=None if staff is None else staff.id,
            direction=direction,
        )
        stems.append(st)
        nh.stem_id = st.id
        nh.stem_source = "pixel"
        pixel_links += 1

    return component_links, pixel_links

def detect_beams(symbols: np.ndarray, staff: np.ndarray, note: np.ndarray, stems_rests: np.ndarray,
                 stems: list[Stem], staves: list[Staff]) -> list[Beam]:
    sp = global_spacing(staves)
    residual = symbols.astype(np.int16) - staff.astype(np.int16) - note.astype(np.int16) - stems_rests.astype(np.int16)
    residual = (residual > 0).astype(np.uint8)
    residual = cv2.morphologyEx(residual, cv2.MORPH_OPEN, np.ones((2,2),np.uint8))

    contours,_ = cv2.findContours(residual, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    beams = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 0.10*sp*sp:
            continue
        rect = cv2.minAreaRect(cnt)
        (cx,cy),(a,b),ang = rect
        length=max(a,b); thick=min(a,b)
        if length < 1.15*sp:
            continue
        if not (0.10*sp <= thick <= 0.85*sp):
            continue
        if length/max(thick,1e-6) < 2.2:
            continue
        pts=cv2.boxPoints(rect)

        edges = []
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            dx = float(p2[0] - p1[0])
            dy = float(p2[1] - p1[1])
            edges.append((math.hypot(dx, dy), dx, dy))
        _, long_dx, long_dy = max(edges, key=lambda item: item[0])
        beam_angle = math.degrees(math.atan2(long_dy, long_dx))
        while beam_angle > 90.0:
            beam_angle -= 180.0
        while beam_angle < -90.0:
            beam_angle += 180.0
        if abs(beam_angle) > 35.0:
            continue

        x1,y1=np.min(pts,axis=0); x2,y2=np.max(pts,axis=0)
        linked=[]
        margin=0.55*sp
        for st in stems:
            tips=[(st.cx,st.y1),(st.cx,st.y2)]
            if any(x1-margin<=x<=x2+margin and y1-margin<=y<=y2+margin for x,y in tips):
                linked.append(st.id)
        if linked:
            beams.append(Beam(
                id=len(beams), points=[[round(float(x),2),round(float(y),2)] for x,y in pts],
                cx=float(cx), cy=float(cy), length=float(length), thickness=float(thick),
                stem_ids=sorted(set(linked))
            ))
    return beams


_SKLEARN_SYMBOL_MODELS: dict[str, dict] = {}


def _load_symbol_model(name: str) -> dict:
    if name not in _SKLEARN_SYMBOL_MODELS:
        path = Path(MODULE_PATH) / "sklearn_models" / f"{name}.model"
        with path.open("rb") as fh:
            _SKLEARN_SYMBOL_MODELS[name] = pickle.load(fh)
    return _SKLEARN_SYMBOL_MODELS[name]


def _classify_symbol(region: np.ndarray, model_name: str) -> tuple[str, float]:
    info = _load_symbol_model(model_name)
    model = info["model"]
    width = int(info["w"])
    height = int(info["h"])
    class_map = info["class_map"]

    patch = np.where(region > 0, 255, 0).astype(np.uint8)
    image = Image.fromarray(patch).resize((width, height))
    x = np.array(image, dtype=np.uint8).reshape(1, -1)
    pred = model.predict(x)[0]
    label = str(class_map[pred])

    margin = 0.0
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(x)).reshape(-1)
        if len(scores) >= 2:
            ordered = np.sort(scores)
            margin = float(ordered[-1] - ordered[-2])
        elif len(scores) == 1:
            margin = float(abs(scores[0]))
    return label, margin


def _bbox_overlap_fraction(
    box: tuple[int, int, int, int],
    other: tuple[int, int, int, int],
) -> float:
    x1, y1, x2, y2 = box
    a1, b1, a2, b2 = other
    ix = max(0, min(x2, a2) - max(x1, a1))
    iy = max(0, min(y2, b2) - max(y1, b1))
    area = max(1, (x2 - x1) * (y2 - y1))
    return float(ix * iy) / float(area)


def detect_rests(
    stems_rests: np.ndarray,
    noteheads: list[Notehead],
    stems: list[Stem],
    beams: list[Beam],
    barlines: list[Barline],
    staves: list[Staff],
    systems: list[SystemRegion],
    measures: list[MeasureRegion],
) -> list[Rest]:
    """
    Detect rest glyphs from the low-level stem/rest mask.

    Straight note stems, barlines, and detected beam polygons are removed as
    visual structures before rest candidates are formed. The pretrained rest
    classifier is used only to name the remaining glyph shape; no timing,
    voice, or MusicXML output from the external OMR system is used.
    """
    sp = global_spacing(staves)
    remove = np.zeros_like(stems_rests, dtype=np.uint8)

    linked_stems = {
        nh.stem_id
        for nh in noteheads
        if nh.stem_id is not None
    }
    for stem in stems:
        if stem.id not in linked_stems:
            continue
        cv2.rectangle(
            remove,
            (
                max(0, int(round(stem.x1 - 2))),
                max(0, int(round(stem.y1 - 2))),
            ),
            (
                min(remove.shape[1] - 1, int(round(stem.x2 + 2))),
                min(remove.shape[0] - 1, int(round(stem.y2 + 2))),
            ),
            1,
            -1,
        )

    for bar in barlines:
        x = int(round(bar.x))
        cv2.rectangle(
            remove,
            (max(0, x - 3), max(0, int(round(bar.top)))),
            (
                min(remove.shape[1] - 1, x + 3),
                min(remove.shape[0] - 1, int(round(bar.bottom))),
            ),
            1,
            -1,
        )

    for beam in beams:
        pts = np.array(beam.points, dtype=np.int32)
        cv2.fillPoly(remove, [pts], 1)

    remove = cv2.dilate(remove, np.ones((3, 3), np.uint8))
    residual = np.where(remove > 0, 0, stems_rests).astype(np.uint8)

    # Remove residual straight-line fragments while preserving irregular rest
    # bodies. Whole/half rests are detected separately from the unsuppressed
    # residual below.
    vertical = cv2.morphologyEx(
        residual,
        cv2.MORPH_OPEN,
        np.ones((max(3, int(round(1.2 * sp))), 1), np.uint8),
    )
    horizontal = cv2.morphologyEx(
        residual,
        cv2.MORPH_OPEN,
        np.ones((1, max(3, int(round(1.2 * sp)))), np.uint8),
    )
    line_mask = np.maximum(
        cv2.dilate(vertical, np.ones((1, 2), np.uint8)),
        cv2.dilate(horizontal, np.ones((2, 1), np.uint8)),
    )
    irregular = np.where(line_mask > 0, 0, residual).astype(np.uint8)
    irregular = cv2.morphologyEx(
        irregular,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    )

    staff_by_id = {staff.id: staff for staff in staves}
    staff_to_system = {
        staff_id: system.id
        for system in systems
        for staff_id in system.staff_ids
    }
    measures_by_system: dict[int, list[MeasureRegion]] = {}
    for measure in measures:
        measures_by_system.setdefault(measure.system_id, []).append(measure)

    def locate(staff: Staff, x: float) -> tuple[int | None, int | None]:
        system_id = staff_to_system.get(staff.id)
        if system_id is None:
            return None, None
        for measure in measures_by_system.get(system_id, []):
            if measure.left <= x <= measure.right:
                return system_id, measure.id
        return system_id, None

    def overlaps_notehead(box: tuple[int, int, int, int]) -> bool:
        return any(
            _bbox_overlap_fraction(
                box,
                (nh.x1, nh.y1, nh.x2, nh.y2),
            ) > 0.10
            for nh in noteheads
        )

    rests: list[Rest] = []

    # Quarter/eighth/shorter rests: irregular, predominantly vertical glyphs.
    for x, y, w, h, area in comp_stats(irregular):
        if area < 0.08 * sp * sp:
            continue
        if not (0.45 * sp <= w <= 1.80 * sp):
            continue
        if not (1.00 * sp <= h <= 3.60 * sp):
            continue
        if w / max(h, 1) > 1.10:
            continue

        box = (x, y, x + w, y + h)
        if overlaps_notehead(box):
            continue

        cx = x + w / 2.0
        cy = y + h / 2.0
        staff = nearest_staff(cy, staves)
        if staff is None:
            continue
        if abs(cy - float(np.mean(staff.lines_y))) > 2.6 * sp:
            continue

        region = irregular[y:y + h, x:x + w]
        label, margin = _classify_symbol(region, "rests")
        if label == "rest_whole":
            # Whole/half rests have a separate strong geometric detector below.
            continue
        if label == "rest_8th":
            label, sub_margin = _classify_symbol(region, "rests_above8")
            margin = max(margin, sub_margin)

        if label not in {
            "rest_quarter",
            "rest_8th",
            "rest_16th",
            "rest_32nd",
            "rest_64th",
        }:
            continue

        system_id, measure_id = locate(staff, cx)
        rests.append(
            Rest(
                id=len(rests),
                x1=x,
                y1=y,
                x2=x + w,
                y2=y + h,
                cx=cx,
                cy=cy,
                staff_id=staff.id,
                rest_type=label,
                classifier_margin=round(float(margin), 4),
                system_id=system_id,
                measure_local=measure_id,
            )
        )

    # Whole/half rests: short horizontal blocks in one of two conventional
    # positions relative to the staff. Use geometry first and classifier support
    # second, so beam fragments elsewhere cannot become rests.
    block_mask = cv2.morphologyEx(
        residual,
        cv2.MORPH_CLOSE,
        np.ones((2, 2), np.uint8),
    )
    existing_boxes = [
        (r.x1, r.y1, r.x2, r.y2)
        for r in rests
    ]
    for x, y, w, h, area in comp_stats(block_mask):
        if area < 0.06 * sp * sp:
            continue
        if not (0.45 * sp <= w <= 1.55 * sp):
            continue
        if not (0.12 * sp <= h <= 0.75 * sp):
            continue
        if w / max(h, 1) < 1.15:
            continue

        box = (x, y, x + w, y + h)
        if overlaps_notehead(box):
            continue
        if any(_bbox_overlap_fraction(box, b) > 0.25 for b in existing_boxes):
            continue

        cx = x + w / 2.0
        cy = y + h / 2.0
        staff = nearest_staff(cy, staves)
        if staff is None:
            continue

        # lines_y is top -> bottom. Whole rest hangs below line 2; half rest
        # sits above the middle line.
        whole_target = staff.lines_y[1] + 0.28 * staff.spacing
        half_target = staff.lines_y[2] - 0.28 * staff.spacing
        whole_dist = abs(cy - whole_target)
        half_dist = abs(cy - half_target)
        if min(whole_dist, half_dist) > 0.55 * staff.spacing:
            continue

        region = block_mask[y:y + h, x:x + w]
        label, margin = _classify_symbol(region, "rests")
        if label != "rest_whole":
            continue

        rest_type = "rest_whole" if whole_dist < half_dist else "rest_half"
        system_id, measure_id = locate(staff, cx)
        rests.append(
            Rest(
                id=len(rests),
                x1=x,
                y1=y,
                x2=x + w,
                y2=y + h,
                cx=cx,
                cy=cy,
                staff_id=staff.id,
                rest_type=rest_type,
                classifier_margin=round(float(margin), 4),
                system_id=system_id,
                measure_local=measure_id,
            )
        )

    rests.sort(key=lambda r: (r.system_id if r.system_id is not None else 9999, r.cy, r.cx))
    for i, rest in enumerate(rests):
        rest.id = i
    return rests


def detect_dots(gray: np.ndarray, staff_mask: np.ndarray, noteheads: list[Notehead], stems: list[Stem],
                beams: list[Beam], staves: list[Staff]) -> list[Dot]:
    sp=global_spacing(staves)
    ink=(gray<160).astype(np.uint8)
    remove=cv2.dilate(staff_mask,np.ones((3,3),np.uint8))
    for nh in noteheads:
        cv2.rectangle(remove,(nh.x1-1,nh.y1-1),(nh.x2+1,nh.y2+1),1,-1)
    for st in stems:
        cv2.rectangle(remove,(st.x1-1,st.y1-1),(st.x2+1,st.y2+1),1,-1)
    for bm in beams:
        pts=np.array(bm.points,dtype=np.int32)
        cv2.fillPoly(remove,[pts],1)
    residual=np.where(remove>0,0,ink).astype(np.uint8)

    comps=[]
    for x,y,w,h,area in comp_stats(residual):
        if 0.015*sp*sp <= area <= 0.22*sp*sp and w<=0.60*sp and h<=0.60*sp:
            comps.append((x+w/2.0,y+h/2.0,area))

    dots=[]
    used=set()
    for nh in noteheads:
        cands=[]
        for j,(cx,cy,area) in enumerate(comps):
            if j in used: continue
            dx=cx-nh.x2
            dy=abs(cy-nh.cy)
            if 0.18*sp<=dx<=1.45*sp and dy<=0.48*sp:
                cands.append((dx,dy,j,cx,cy,area))
        if cands:
            _,_,j,cx,cy,area=min(cands)
            d=Dot(id=len(dots),cx=float(cx),cy=float(cy),area=int(area),notehead_id=nh.id)
            dots.append(d); used.add(j); nh.dot_id=d.id
    return dots


def tie_confidence(gray: np.ndarray, staff_mask: np.ndarray, a: Notehead, b: Notehead, sp: float) -> tuple[str,float]:
    if b.cx <= a.cx:
        return "none",0.0
    gap=b.x1-a.x2
    if gap < 0.5*sp or gap > 8.0*sp:
        return "none",0.0
    if abs(a.cy-b.cy)>0.38*sp:
        return "none",0.0

    ink=(gray<170).astype(np.uint8)
    stafffree=np.where(cv2.dilate(staff_mask,np.ones((3,1),np.uint8))>0,0,ink).astype(np.uint8)
    x1=max(0,a.x2); x2=min(gray.shape[1],b.x1)
    if x2-x1<3:
        return "none",0.0

    best=("none",0.0)
    for side,ya,yb in [
        ("above",min(a.y1,b.y1)-int(1.15*sp),min(a.y1,b.y1)+int(0.10*sp)),
        ("below",max(a.y2,b.y2)-int(0.10*sp),max(a.y2,b.y2)+int(1.15*sp))
    ]:
        ya=max(0,ya); yb=min(gray.shape[0],yb)
        if yb<=ya: continue
        reg=stafffree[ya:yb,x1:x2]
        n,lab,stats,cent=cv2.connectedComponentsWithStats(reg,8)
        for i in range(1,n):
            x,y,w,h,area=[int(v) for v in stats[i]]
            span=w/max(1,x2-x1)
            if span<0.50 or h>1.15*sp:
                continue
            density=area/max(1,w*h)
            if density>0.55:
                continue
            conf=min(1.0,0.55*span+0.45*(1.0-min(1.0,density/0.55)))
            if conf>best[1]:
                best=(side,float(conf))
    return best


def detect_tie_candidates(gray: np.ndarray, staff_mask: np.ndarray, noteheads: list[Notehead], staves: list[Staff]) -> list[TieCandidate]:
    sp=global_spacing(staves)
    by_staff={}
    for n in noteheads:
        by_staff.setdefault(n.staff_id,[]).append(n)
    out=[]
    for sid,heads in by_staff.items():
        if sid is None: continue
        heads=sorted(heads,key=lambda n:n.cx)
        for i,a in enumerate(heads):
            for b in heads[i+1:]:
                if b.cx-a.cx>8.5*sp: break
                if a.staff_pos_halfspaces != b.staff_pos_halfspaces:
                    continue
                side,conf=tie_confidence(gray,staff_mask,a,b,sp)
                if conf>=0.62:
                    out.append(TieCandidate(
                        id=len(out),left_notehead_id=a.id,right_notehead_id=b.id,
                        side=side,confidence=round(conf,4)
                    ))
                    break
    return out


def overlay(gray: np.ndarray, staves: list[Staff], noteheads: list[Notehead], stems: list[Stem],
            beams: list[Beam], rests: list[Rest], dots: list[Dot], ties: list[TieCandidate], out_path: Path) -> None:
    img=Image.fromarray(gray).convert("RGB")
    d=ImageDraw.Draw(img)
    for st in stems:
        d.rectangle((st.x1,st.y1,st.x2,st.y2),outline=(255,135,0),width=2)
    for bm in beams:
        pts=[tuple(p) for p in bm.points]
        d.line(pts+[pts[0]],fill=(220,0,220),width=3)
    for nh in noteheads:
        col=(0,190,0) if nh.head_type=="filled" else (0,150,255)
        d.ellipse((nh.x1,nh.y1,nh.x2,nh.y2),outline=col,width=3)
    for rest in rests:
        d.rectangle((rest.x1,rest.y1,rest.x2,rest.y2),outline=(0,190,190),width=2)
    for dot in dots:
        r=4
        d.ellipse((dot.cx-r,dot.cy-r,dot.cx+r,dot.cy+r),outline=(235,190,0),width=2)
    byid={n.id:n for n in noteheads}
    for tie in ties:
        a,b=byid[tie.left_notehead_id],byid[tie.right_notehead_id]
        y=min(a.y1,b.y1)-8 if tie.side=="above" else max(a.y2,b.y2)+8
        d.line((a.cx,y,b.cx,y),fill=(220,0,0),width=2)
    img.save(out_path)


def process_page(
    page: Path,
    out_dir: Path,
    page_index: int,
    cache_dir: Path | None = None,
) -> dict:
    cache_path = None if cache_dir is None else cache_dir / f"page_{page_index:02d}.npz"
    gray, staff_mask, symbols, stems_rests, note_mask, clefs_keys = run_segmentation(page, cache_path)
    (
        skew_degrees,
        gray,
        staff_mask,
        symbols,
        stems_rests,
        note_mask,
        clefs_keys,
    ) = deskew_layers(gray, staff_mask, symbols, stems_rests, note_mask, clefs_keys)

    staves=detect_staves(gray,staff_mask)
    noteheads=detect_noteheads(gray,note_mask,staves)
    stems=vertical_components(stems_rests,staves)
    component_stem_links, pixel_stem_links = link_noteheads_stems(
        noteheads, stems, staves, gray
    )
    systems = detect_systems(gray, staves)
    barlines, measures = detect_barlines_and_measures(
        gray, staves, systems
    )
    assign_regions(noteheads, stems, systems, measures)
    beams=detect_beams(symbols,staff_mask,note_mask,stems_rests,stems,staves)
    rests=detect_rests(
        stems_rests,
        noteheads,
        stems,
        beams,
        barlines,
        staves,
        systems,
        measures,
    )
    dots=detect_dots(gray,staff_mask,noteheads,stems,beams,staves)
    ties=detect_tie_candidates(gray,staff_mask,noteheads,staves)

    overlay_path=out_dir/f"page_{page_index:02d}_optical_overlay.png"
    overlay(gray,staves,noteheads,stems,beams,rests,dots,ties,overlay_path)

    unlinked_filled=sum(1 for n in noteheads if n.head_type=="filled" and n.stem_id is None)
    print("OPTICAL_PAGE=" + json.dumps({
        "page": page_index,
        "skew_degrees": round(skew_degrees, 4),
        "staves": len(staves),
        "systems": len(systems),
        "measures": len(measures),
        "noteheads": len(noteheads),
        "stems": len(stems),
        "beams": len(beams),
        "rests": len(rests),
        "dots": len(dots),
        "ties": len(ties),
        "component_stem_links": component_stem_links,
        "pixel_stem_links": pixel_stem_links,
        "unlinked_filled": unlinked_filled,
    }, separators=(",", ":")))

    # Rest classifiers are only needed while processing this page. Releasing
    # them prevents page-1 classifier memory from accumulating with page-2 ONNX
    # inference inside small Railway containers.
    _SKLEARN_SYMBOL_MODELS.clear()
    _trim_process_memory()

    return {
        "page":page_index,
        "skew_degrees":round(skew_degrees,4),
        "image_size":[int(gray.shape[1]),int(gray.shape[0])],
        "staff_count":len(staves),
        "system_count":len(systems),
        "measure_count":len(measures),
        "staves":[asdict(x) for x in staves],
        "systems":[asdict(x) for x in systems],
        "barlines":[asdict(x) for x in barlines],
        "measures":[asdict(x) for x in measures],
        "notehead_count":len(noteheads),
        "filled_notehead_count":sum(n.head_type=="filled" for n in noteheads),
        "hollow_notehead_count":sum(n.head_type=="hollow" for n in noteheads),
        "unlinked_filled_noteheads":unlinked_filled,
        "stem_count":len(stems),
        "beam_count":len(beams),
        "rest_count":len(rests),
        "dot_count":len(dots),
        "tie_candidate_count":len(ties),
        "component_stem_links":component_stem_links,
        "pixel_stem_links":pixel_stem_links,
        "noteheads":[asdict(x) for x in noteheads],
        "stems":[asdict(x) for x in stems],
        "beams":[asdict(x) for x in beams],
        "rests":[asdict(x) for x in rests],
        "dots":[asdict(x) for x in dots],
        "tie_candidates":[asdict(x) for x in ties],
        "overlay":overlay_path.name,
    }


def analyze_input(
    input_path: Path,
    out_dir: Path,
    dpi: int = 300,
    cache_dir: Path | None = None,
) -> dict:
    ensure_checkpoints()
    out_dir.mkdir(parents=True, exist_ok=True)

    if input_path.suffix.lower() == ".pdf":
        pages = render_pdf(input_path, out_dir / "rendered", dpi=dpi)
    else:
        pages = [input_path]

    result = {
        "engine": "tma-optical-reader-clean-v1",
        "stage": "optical_notation_graph",
        "timing_source": "none",
        "semantic_timing_used": False,
        "rhythmic_attacks_ready": False,
        "external_low_level_model": "oemer segmentation masks only",
        "pages": [],
    }

    for i, page in enumerate(pages, 1):
        print(f"Processing optical page {i}/{len(pages)}")
        result["pages"].append(
            process_page(page, out_dir, i, cache_dir)
        )

    result["totals"] = {
        "pages": len(result["pages"]),
        "staves": sum(x["staff_count"] for x in result["pages"]),
        "systems": sum(x["system_count"] for x in result["pages"]),
        "measures": sum(x["measure_count"] for x in result["pages"]),
        "noteheads": sum(x["notehead_count"] for x in result["pages"]),
        "filled_noteheads": sum(x["filled_notehead_count"] for x in result["pages"]),
        "hollow_noteheads": sum(x["hollow_notehead_count"] for x in result["pages"]),
        "unlinked_filled_noteheads": sum(x["unlinked_filled_noteheads"] for x in result["pages"]),
        "stems": sum(x["stem_count"] for x in result["pages"]),
        "beams": sum(x["beam_count"] for x in result["pages"]),
        "rests": sum(x["rest_count"] for x in result["pages"]),
        "dots": sum(x["dot_count"] for x in result["pages"]),
        "tie_candidates": sum(x["tie_candidate_count"] for x in result["pages"]),
        "component_stem_links": sum(x["component_stem_links"] for x in result["pages"]),
        "pixel_stem_links": sum(x["pixel_stem_links"] for x in result["pages"]),
    }

    (out_dir / "notation_graph.json").write_text(
        json.dumps(result, indent=2)
    )
    print(
        "OPTICAL_SUMMARY="
        + json.dumps(result["totals"], separators=(",", ":"))
    )
    return result


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("input",type=Path)
    ap.add_argument("--out",type=Path,default=Path("optical_audit"))
    ap.add_argument("--dpi",type=int,default=300)
    ap.add_argument("--cache-dir",type=Path,default=None)
    args=ap.parse_args()

    analyze_input(
        input_path=args.input,
        out_dir=args.out,
        dpi=args.dpi,
        cache_dir=args.cache_dir,
    )


if __name__=="__main__":
    main()
