from __future__ import annotations
from pathlib import Path
import pymupdf
from PIL import Image, ImageDraw
from core import Strike


def render_pdf_with_strikes(pdf_path: str | Path, strikes: list[Strike], out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[int, list[Strike]] = {}
    for strike in strikes:
        grouped.setdefault(strike.page_index, []).append(strike)

    doc = pymupdf.open(str(pdf_path))
    outputs = []
    for page_index, page in enumerate(doc):
        pix = page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5), alpha=False)
        image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
        draw = ImageDraw.Draw(image)
        for strike in grouped.get(page_index, []):
            sx = image.width / strike.omr_width
            sy = image.height / strike.omr_height
            x = round(strike.x * sx)
            pad = max(4, round(12 * sy))
            y1 = max(0, round(strike.system_top * sy) - pad)
            y2 = min(image.height - 1, round(strike.system_bottom * sy) + pad)
            width = max(2, round(image.width / 1400))
            draw.line((x,y1,x,y2), fill=(220,0,0), width=width)
        path = out_dir / f'page-{page_index+1:03d}.png'
        image.save(path)
        outputs.append(path)
    return outputs
