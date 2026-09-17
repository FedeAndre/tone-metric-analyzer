from __future__ import annotations

from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

from core import Strike


def render_pdf_with_strikes(
    pdf_path: str | Path,
    strikes: list[Strike],
    output_dir: str | Path,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_page: dict[int, list[Strike]] = {}
    for strike in strikes:
        by_page.setdefault(strike.page_index, []).append(strike)

    document = pymupdf.open(str(pdf_path))
    outputs: list[Path] = []

    for page_index, page in enumerate(document):
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(1.5, 1.5),
            alpha=False,
        )
        image = Image.frombytes(
            "RGB",
            (pixmap.width, pixmap.height),
            pixmap.samples,
        )
        draw = ImageDraw.Draw(image)

        for strike in by_page.get(page_index, []):
            scale_x = image.width / strike.omr_width
            scale_y = image.height / strike.omr_height
            x = round(strike.x * scale_x)
            padding = max(4, round(12 * scale_y))
            y1 = max(0, round(strike.system_top * scale_y) - padding)
            y2 = min(
                image.height - 1,
                round(strike.system_bottom * scale_y) + padding,
            )
            line_width = max(2, round(image.width / 1400))
            draw.line(
                (x, y1, x, y2),
                fill=(220, 0, 0),
                width=line_width,
            )

        path = output_dir / f"page-{page_index + 1:03d}.png"
        image.save(path)
        outputs.append(path)

    return outputs
