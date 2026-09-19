from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ALLOWED_TOP = {
    ".github",
    "README.md",
    "requirements.txt",
    "optical_reader.py",
    "app.py",
    "Dockerfile",
    "tests",
}

BANNED_SOURCE_TOKENS = [
    "from core import",
    "import core",
    "audiveris",
    "homr",
    "extract_hit_strikes",
    "canonical_score",
    "build_hits_from_canonical",
    "time-offset",
    "x-offset",
]


def test_clean_tree_has_no_previous_reader_files():
    tops = {p.name for p in ROOT.iterdir() if p.name != ".git"}
    assert tops <= ALLOWED_TOP, sorted(tops - ALLOWED_TOP)


def test_no_legacy_reader_imports_or_timing_paths():
    src = (ROOT / "optical_reader.py").read_text().lower()
    for token in BANNED_SOURCE_TOKENS:
        assert token.lower() not in src, token


def test_external_omr_is_used_only_for_low_level_inference():
    src = (ROOT / "optical_reader.py").read_text()
    assert "from oemer.inference import inference" in src
    forbidden = [
        "MusicXMLBuilder",
        "rhythm_extract",
        "group_extract",
        "symbol_extract",
        "note_extract",
        "staff_extract",
    ]
    for token in forbidden:
        assert token not in src, token


def test_service_layer_is_clean():
    src = (ROOT / "app.py").read_text().lower()
    for token in BANNED_SOURCE_TOKENS:
        assert token.lower() not in src, token
    assert "/api/status" in src
    assert "/api/optical/read" in src
