from __future__ import annotations

import argparse
import json
import math
import os
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import cv2
import fitz
import numpy as np
from PIL import Image, ImageDraw

from oemer import MODULE_PATH
from oemer.inference import inference


CHECKPOINTS = {
    "unet_big/model.onnx": "https://github.com/BreezeWhite/oemer/releases/download/checkpoints/1st_model.onnx",
    "seg_net/model.onnx": "https://github.com/BreezeWhite/oemer/releases/download/checkpoints/2nd_model.onnx",
}


@dataclass
class Staff:
    id: int
    lines_y: list[float]
    spacing: float
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


def run_segmentation(
    img_path: Path,
    cache_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if cache_path is not None and cache_path.exists():
        z = np.load(cache_path)
        return (
            z["gray"], z["staff"], z["symbols"],
            z["stems_rests"], z["noteheads"], z["clefs_keys"],
        )

    first, _ = inference(str(Path(MODULE_PATH) / "checkpoints" / "unet_big"), str(img_path), use_tf=False)
    staff = (first == 1).astype(np.uint8)
    symbols = (first == 2).astype(np.uint8)

    second, _ = inference(str(Path(MODULE_PATH) / "checkpoints" / "seg_net"), str(img_path), use_tf=False)
    stems_rests = (second == 1).astype(np.uint8)
    noteheads = (second == 2).astype(np.uint8)
    clefs_keys = (second == 3).astype(np.uint8)

    src = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if src is None:
        raise RuntimeError(f"Cannot load {img_path}")
    src = cv2.resize(src, (staff.shape[1], staff.shape[0]), interpolation=cv2.INTER_AREA)

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
            beams: list[Beam], dots: list[Dot], ties: list[TieCandidate], out_path: Path) -> None:
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
    beams=detect_beams(symbols,staff_mask,note_mask,stems_rests,stems,staves)
    dots=detect_dots(gray,staff_mask,noteheads,stems,beams,staves)
    ties=detect_tie_candidates(gray,staff_mask,noteheads,staves)

    overlay_path=out_dir/f"page_{page_index:02d}_optical_overlay.png"
    overlay(gray,staves,noteheads,stems,beams,dots,ties,overlay_path)

    unlinked_filled=sum(1 for n in noteheads if n.head_type=="filled" and n.stem_id is None)
    print("OPTICAL_PAGE=" + json.dumps({
        "page": page_index,
        "skew_degrees": round(skew_degrees, 4),
        "staves": len(staves),
        "noteheads": len(noteheads),
        "stems": len(stems),
        "beams": len(beams),
        "dots": len(dots),
        "ties": len(ties),
        "component_stem_links": component_stem_links,
        "pixel_stem_links": pixel_stem_links,
        "unlinked_filled": unlinked_filled,
    }, separators=(",", ":")))

    return {
        "page":page_index,
        "skew_degrees":round(skew_degrees,4),
        "image_size":[int(gray.shape[1]),int(gray.shape[0])],
        "staff_count":len(staves),
        "staves":[asdict(x) for x in staves],
        "notehead_count":len(noteheads),
        "filled_notehead_count":sum(n.head_type=="filled" for n in noteheads),
        "hollow_notehead_count":sum(n.head_type=="hollow" for n in noteheads),
        "unlinked_filled_noteheads":unlinked_filled,
        "stem_count":len(stems),
        "beam_count":len(beams),
        "dot_count":len(dots),
        "tie_candidate_count":len(ties),
        "component_stem_links":component_stem_links,
        "pixel_stem_links":pixel_stem_links,
        "noteheads":[asdict(x) for x in noteheads],
        "stems":[asdict(x) for x in stems],
        "beams":[asdict(x) for x in beams],
        "dots":[asdict(x) for x in dots],
        "tie_candidates":[asdict(x) for x in ties],
        "overlay":overlay_path.name,
    }


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("input",type=Path)
    ap.add_argument("--out",type=Path,default=Path("optical_audit"))
    ap.add_argument("--dpi",type=int,default=300)
    ap.add_argument("--cache-dir",type=Path,default=None)
    args=ap.parse_args()

    ensure_checkpoints()
    args.out.mkdir(parents=True,exist_ok=True)
    if args.input.suffix.lower()==".pdf":
        pages=render_pdf(args.input,args.out/"rendered",dpi=args.dpi)
    else:
        pages=[args.input]

    result={
        "engine":"tma-optical-reader-clean-v1",
        "timing_source":"none",
        "semantic_timing_used":False,
        "external_low_level_model":"oemer segmentation masks only",
        "pages":[],
    }
    for i,p in enumerate(pages,1):
        print(f"Processing optical page {i}/{len(pages)}")
        result["pages"].append(process_page(p,args.out,i,args.cache_dir))

    result["totals"]={
        "pages":len(result["pages"]),
        "staves":sum(x["staff_count"] for x in result["pages"]),
        "noteheads":sum(x["notehead_count"] for x in result["pages"]),
        "filled_noteheads":sum(x["filled_notehead_count"] for x in result["pages"]),
        "hollow_noteheads":sum(x["hollow_notehead_count"] for x in result["pages"]),
        "unlinked_filled_noteheads":sum(x["unlinked_filled_noteheads"] for x in result["pages"]),
        "stems":sum(x["stem_count"] for x in result["pages"]),
        "beams":sum(x["beam_count"] for x in result["pages"]),
        "dots":sum(x["dot_count"] for x in result["pages"]),
        "tie_candidates":sum(x["tie_candidate_count"] for x in result["pages"]),
        "component_stem_links":sum(x["component_stem_links"] for x in result["pages"]),
        "pixel_stem_links":sum(x["pixel_stem_links"] for x in result["pages"]),
    }
    (args.out/"notation_graph.json").write_text(json.dumps(result,indent=2))
    print("OPTICAL_SUMMARY="+json.dumps(result["totals"],separators=(",",":")))


if __name__=="__main__":
    main()
