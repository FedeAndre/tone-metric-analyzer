from pathlib import Path
from app import analyze_musicxml

path = Path("/app/fixtures/tabi.musicxml")
if not path.exists() or path.stat().st_size == 0:
    raise SystemExit("fixture MusicXML missing")
m = analyze_musicxml(path)
assert m["part_count"] >= 1, m
assert m["measure_count"] >= 8, m
assert m["note_elements"] >= 20, m
assert m["global_attack_count"] >= 10, m
assert any(k in m["note_type_counts"] for k in ("eighth", "quarter", "half")), m
print("HOMR_FIXTURE_OK", {k: m[k] for k in ("part_count","measure_count","note_elements","global_attack_count")})
