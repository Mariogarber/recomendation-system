import pandas as pd

from analysis.diagnostics import _hypothesis_summary, _segment_mask, _support_bucket


def test_support_bucket_matches_expected_ranges():
    assert _support_bucket(0) == "0"
    assert _support_bucket(1) == "1"
    assert _support_bucket(4) == "2-5"
    assert _support_bucket(8) == "6-20"
    assert _support_bucket(30) == ">20"


def test_segment_mask_splits_short_band_correctly():
    frame = pd.DataFrame({"history_count": [2, 3, 4, 5]})
    assert _segment_mask(frame, "2").tolist() == [True, False, False, False]
    assert _segment_mask(frame, "2-3").tolist() == [True, True, False, False]
    assert _segment_mask(frame, "4-5").tolist() == [False, False, True, True]


def test_hypothesis_summary_flags_overcorrection_and_mix():
    frame = pd.DataFrame(
        {
            "snapshot": ["vX"] * 6,
            "history_band": ["2-5"] * 6,
            "history_count": [2, 2, 3, 3, 2, 3],
            "deep_available": [True] * 6,
            "worse_than_incumbent": [True, True, False, False, True, False],
            "error_delta_vs_incumbent": [0.25, 0.2, -0.15, -0.1, 0.18, -0.05],
            "abs_correction": [0.9, 0.8, 0.2, 0.3, 0.7, 0.2],
            "alpha": [0.9, 0.85, 0.3, 0.35, 0.8, 0.4],
            "deep_abs_error": [1.1, 1.0, 0.4, 0.45, 0.95, 0.5],
            "incumbent_abs_error": [0.85, 0.8, 0.55, 0.55, 0.77, 0.55],
        }
    )
    crosscuts = pd.DataFrame(
        {
            "snapshot": ["vX"] * 4,
            "crosscut": ["prefix_similarity_bucket", "prefix_similarity_bucket", "history_variance_bucket", "item_status"],
            "bucket": ["sim_q1", "sim_q4", "var_q4", "new_item"],
            "delta_mae": [0.03, -0.01, 0.015, -0.02],
            "n_samples": [2, 2, 2, 2],
        }
    )

    summary = _hypothesis_summary(frame, crosscuts)

    assert summary["hypotheses"]["H1_overcorrection"]["supported"] is True
    assert summary["hypotheses"]["H3_subpopulation_mix"]["supported"] is True
