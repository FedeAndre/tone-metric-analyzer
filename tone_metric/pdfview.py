from __future__ import annotations
from pathlib import Path
import fitz


def render_pdf_pages(pdf_path: str | Path, out_dir: str | Path, zoom: float = 1.75) -> list[dict]:
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    pages = []
    matrix = fitz.Matrix(zoom, zoom)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        name = f"page_{i+1}.png"
        path = out_dir / name
        pix.save(path)
        pages.append({"index": i, "filename": name, "width": pix.width, "height": pix.height})
    doc.close()
    return pages
