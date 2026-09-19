from tma_analysis import analyze_tone_metric


def test_exact_hits_feed_levels_wave_pivots_and_trees():
    rhythm = {
        "rhythmic_attacks_ready": True,
        "meter": {
            "numerator": 4,
            "denominator": 4,
            "measure_duration_quarter": "4",
        },
        "measures": [{
            "measure_index": 0,
            "measure_number": "1",
            "start_quarter": "0",
            "full_duration_quarter": "4",
            "pickup_shift_quarter": "0",
        }],
        "hits": [
            {
                "measure_index": 0,
                "measure_number": "1",
                "offset_in_measure_quarter": str(i),
                "onset_quarter": str(i),
                "source_count": 1,
                "sources": [{
                    "duration_quarter": "1",
                    "page": 1,
                    "staff_id": 0,
                    "voice": "up",
                    "notehead_ids": [i],
                }],
            }
            for i in range(4)
        ],
    }
    result = analyze_tone_metric(rhythm)
    assert result["ready"] is True
    assert result["stats"]["hit_count"] == 4
    assert result["stats"]["wave_points"] == 4
    assert result["stats"]["tree_nodes"] == 4
    assert result["stats"]["max_level"] >= 1
    assert len(result["levels"]["segments"]) == 1


def test_tma_refuses_unresolved_timing():
    rhythm = {
        "rhythmic_attacks_ready": False,
        "meter": {
            "numerator": 4,
            "denominator": 4,
            "measure_duration_quarter": "4",
        },
    }
    try:
        analyze_tone_metric(rhythm)
    except ValueError as exc:
        assert "fully resolved exact attack timeline" in str(exc)
    else:
        raise AssertionError("unresolved timing must not enter TMA")
