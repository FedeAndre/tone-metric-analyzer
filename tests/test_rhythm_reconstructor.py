from fractions import Fraction

from rhythm_reconstructor import reconstruct_attacks


def _graph(
    *,
    staff_count=1,
    heads,
    stems,
    rests=None,
    dots=None,
    ties=None,
):
    staves = []
    for i in range(staff_count):
        base = 100 + i * 100
        staves.append({
            "id": i,
            "lines_y": [base + 10*j for j in range(5)],
            "spacing": 10.0,
        })
    return {
        "pages": [{
            "page": 1,
            "staves": staves,
            "systems": [{
                "id": 0,
                "staff_ids": list(range(staff_count)),
                "left": 0,
                "right": 1000,
            }],
            "measures": [{
                "id": 1,
                "system_id": 0,
                "measure_in_system": 1,
                "left": 0,
                "right": 1000,
            }],
            "noteheads": heads,
            "stems": stems,
            "rests": rests or [],
            "dots": dots or [],
            "tie_candidates": ties or [],
        }]
    }


def _quarter_events(xs, *, staff=0, id_base=0):
    heads = []
    stems = []
    for i, x in enumerate(xs):
        stem_id = id_base + i
        head_id = id_base + i
        heads.append({
            "id": head_id,
            "cx": float(x),
            "cy": 120.0 + 100*staff,
            "x1": int(x-5),
            "x2": int(x+5),
            "y1": int(115+100*staff),
            "y2": int(125+100*staff),
            "staff_id": staff,
            "measure_local": 1,
            "stem_id": stem_id,
            "head_type": "filled",
            "dot_id": None,
        })
        stems.append({
            "id": stem_id,
            "staff_id": staff,
            "measure_local": 1,
            "direction": "up",
            "beam_level": 0,
        })
    return heads, stems


def test_irregular_x_spacing_never_becomes_time():
    heads, stems = _quarter_events([80, 115, 510, 930])
    result = reconstruct_attacks(
        _graph(heads=heads, stems=stems),
        meter=(4, 4),
    )
    assert result["rhythmic_attacks_ready"] is True
    assert [row["offset_in_measure_quarter"] for row in result["hits"]] == [
        "0", "1", "2", "3"
    ]
    assert result["x_position_used_as_time"] is False


def test_false_dot_is_not_allowed_to_overfill_measure():
    heads, stems = _quarter_events([100, 300, 500, 700])
    heads[0]["dot_id"] = 0
    result = reconstruct_attacks(
        _graph(
            heads=heads,
            stems=stems,
            dots=[{
                "id": 0,
                "notehead_id": 0,
                "visual_confidence": 0.95,
            }],
        ),
        meter=(4, 4),
    )
    assert result["rhythmic_attacks_ready"] is True
    assert result["attacks"][0]["dotted"] is False
    assert [row["offset_in_measure_quarter"] for row in result["hits"]] == [
        "0", "1", "2", "3"
    ]


def test_dot_is_accepted_only_when_metric_equation_requires_it():
    heads = [
        {
            "id": 0, "cx": 100.0, "cy": 120.0,
            "x1": 95, "x2": 105, "y1": 115, "y2": 125,
            "staff_id": 0, "measure_local": 1,
            "stem_id": 0, "head_type": "hollow", "dot_id": 0,
        },
        {
            "id": 1, "cx": 700.0, "cy": 120.0,
            "x1": 695, "x2": 705, "y1": 115, "y2": 125,
            "staff_id": 0, "measure_local": 1,
            "stem_id": 1, "head_type": "filled", "dot_id": None,
        },
    ]
    stems = [
        {"id": 0, "staff_id": 0, "measure_local": 1,
         "direction": "up", "beam_level": 0},
        {"id": 1, "staff_id": 0, "measure_local": 1,
         "direction": "up", "beam_level": 0},
    ]
    result = reconstruct_attacks(
        _graph(
            heads=heads,
            stems=stems,
            dots=[{
                "id": 0,
                "notehead_id": 0,
                "visual_confidence": 0.95,
            }],
        ),
        meter=(4, 4),
    )
    assert result["rhythmic_attacks_ready"] is True
    assert result["attacks"][0]["dotted"] is True
    assert result["attacks"][0]["duration_quarter"] == "3"
    assert [row["offset_in_measure_quarter"] for row in result["hits"]] == [
        "0", "3"
    ]


def test_rests_advance_time_but_never_create_hits():
    heads = [
        {
            "id": 0, "cx": 100.0, "cy": 120.0,
            "x1": 95, "x2": 105, "y1": 115, "y2": 125,
            "staff_id": 0, "measure_local": 1,
            "stem_id": 0, "head_type": "filled", "dot_id": None,
        },
        {
            "id": 1, "cx": 700.0, "cy": 120.0,
            "x1": 695, "x2": 705, "y1": 115, "y2": 125,
            "staff_id": 0, "measure_local": 1,
            "stem_id": 1, "head_type": "hollow", "dot_id": None,
        },
    ]
    stems = [
        {"id": 0, "staff_id": 0, "measure_local": 1,
         "direction": "up", "beam_level": 0},
        {"id": 1, "staff_id": 0, "measure_local": 1,
         "direction": "up", "beam_level": 0},
    ]
    rests = [{
        "id": 0,
        "cx": 400.0,
        "cy": 120.0,
        "staff_id": 0,
        "measure_local": 1,
        "rest_type": "rest_quarter",
    }]
    result = reconstruct_attacks(
        _graph(heads=heads, stems=stems, rests=rests),
        meter=(4, 4),
    )
    assert result["rhythmic_attacks_ready"] is True
    assert [row["offset_in_measure_quarter"] for row in result["hits"]] == [
        "0", "2"
    ]
    assert len(result["hits"]) == 2


def test_tied_continuation_keeps_duration_but_not_reattack():
    heads, stems = _quarter_events([100, 300, 500, 700])
    ties = [{
        "id": 0,
        "left_notehead_id": 0,
        "right_notehead_id": 1,
    }]
    result = reconstruct_attacks(
        _graph(heads=heads, stems=stems, ties=ties),
        meter=(4, 4),
    )
    assert result["rhythmic_attacks_ready"] is True
    assert [row["offset_in_measure_quarter"] for row in result["hits"]] == [
        "0", "2", "3"
    ]


def test_simultaneous_attacks_merge_across_staves():
    h0, s0 = _quarter_events([100, 300, 500, 700], staff=0, id_base=0)
    h1, s1 = _quarter_events([102, 305, 498, 704], staff=1, id_base=10)
    result = reconstruct_attacks(
        _graph(
            staff_count=2,
            heads=h0+h1,
            stems=s0+s1,
        ),
        meter=(4, 4),
    )
    assert result["rhythmic_attacks_ready"] is True
    assert len(result["hits"]) == 4
    assert all(row["source_count"] == 2 for row in result["hits"])
