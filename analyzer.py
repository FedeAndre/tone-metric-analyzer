from __future__ import annotations

import argparse
import json
from pathlib import Path

from optical_reader import analyze_input
from rhythm_reconstructor import parse_meter, reconstruct_attacks
from tma_analysis import analyze_tone_metric


ANALYZER_VERSION = "tma-clean-product-v1"


def analyze_score(
    input_path: Path,
    out_dir: Path,
    *,
    meter_text: str,
    dpi: int = 300,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    meter = parse_meter(meter_text)

    optical_dir = out_dir / "optical"
    cache_dir = out_dir / "optical_cache"
    optical = analyze_input(
        input_path=input_path,
        out_dir=optical_dir,
        dpi=dpi,
        cache_dir=cache_dir,
    )

    rhythm = reconstruct_attacks(optical, meter=meter)
    (out_dir / "rhythm.json").write_text(
        json.dumps(rhythm, indent=2)
    )

    result = {
        "version": ANALYZER_VERSION,
        "semantic_timing_used": False,
        "x_position_used_as_time": False,
        "meter": meter_text,
        "optical": {
            "engine": optical["engine"],
            "totals": optical["totals"],
            "notation_graph": "optical/notation_graph.json",
        },
        "rhythm": rhythm,
        "tma": None,
        "ready": False,
    }

    if rhythm.get("rhythmic_attacks_ready"):
        tma = analyze_tone_metric(rhythm)
        result["tma"] = tma
        result["ready"] = bool(tma.get("ready"))
    else:
        result["blocking_reason"] = (
            "Exact rhythmic attacks were not proven for every retained attack; "
            "Tone-Metric analysis was not run."
        )

    (out_dir / "analysis.json").write_text(
        json.dumps(result, indent=2)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, default=Path("tma_analysis"))
    parser.add_argument("--meter", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    result = analyze_score(
        args.input,
        args.out,
        meter_text=args.meter,
        dpi=args.dpi,
    )
    print(
        "TMA_PRODUCT_SUMMARY="
        + json.dumps({
            "ready": result["ready"],
            "optical_totals": result["optical"]["totals"],
            "rhythm_stats": result["rhythm"]["stats"],
            "tma_stats": (
                None
                if result["tma"] is None
                else result["tma"]["stats"]
            ),
        }, separators=(",", ":"))
    )
    if not result["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
