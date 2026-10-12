"""Unit tests for the pure logic in scripts/ordinal_analysis.py.

Covers the GG mapping, ordinal C-index, and tie-aware pairwise AUC. No GPU,
no checkpoints, no inference -- safe to run anywhere the env imports.

Run:
    uv run python test_ordinal_analysis.py      # plain asserts, prints PASSED
    uv run pytest test_ordinal_analysis.py -q   # or via pytest
"""

import numpy as np

from agora.scripts.ordinal_analysis import (
    MODELS,
    SEEDS,
    _ordinal_c_index,
    _pairwise_auc,
    gleason_to_gg,
)


def test_gleason_to_gg():
    cases = {
        # benign / missing folds into GG1 (<=6, low-risk)
        "0+0": 1, "nan": 1, "N/A": 1, "": 1, None: 1,
        # <=6 -> GG1
        "3+3": 1, "3+2": 1, "2+4": 1, "2+3": 1,
        # 7 splits by primary: 3+4 GG2, 4+3 GG3
        "3+4": 2, "4+3": 3,
        # 8 -> GG4
        "4+4": 4, "3+5": 4, "5+3": 4,
        # 9-10 -> GG5
        "4+5": 5, "5+4": 5, "5+5": 5,
        # bytes + stray leading quote (seen in the raw CSV) must still parse
        b"3+4": 2, '"3+4': 2, " 4+3 ": 3,
    }
    for raw, expected in cases.items():
        got = gleason_to_gg(raw)
        assert got == expected, f"{raw!r} -> {got}, expected {expected}"

    assert gleason_to_gg("garbage") is None  # unparseable -> None (dropped later)


def test_ordinal_c_index():
    # perfectly grade-ordered scores -> C-index 1.0
    assert _ordinal_c_index([0.1, 0.2, 0.3, 0.4], [1, 2, 3, 4]) == 1.0
    # perfectly reversed -> 0.0
    assert _ordinal_c_index([0.4, 0.3, 0.2, 0.1], [1, 2, 3, 4]) == 0.0
    # all scores tied across grades -> 0.5 (pure ties)
    assert _ordinal_c_index([0.5, 0.5, 0.5], [1, 2, 3]) == 0.5
    # same grade never forms a comparable pair -> NaN
    assert np.isnan(_ordinal_c_index([0.1, 0.9], [2, 2]))


def test_pairwise_auc():
    assert _pairwise_auc([0.1, 0.2], [0.3, 0.4]) == 1.0
    assert _pairwise_auc([0.3, 0.4], [0.1, 0.2]) == 0.0
    assert _pairwise_auc([0.5, 0.5], [0.5, 0.5]) == 0.5  # all ties -> 0.5
    assert np.isnan(_pairwise_auc([], [0.1]))  # empty side -> NaN


def test_model_wiring():
    # the 4 best clinical (flatten + att) configs, seeds 7-11
    assert set(v[0] for v in MODELS.values()) == {
        "cnn3d", "resnet3d", "omnirad", "radiodino"
    }
    assert SEEDS == [7, 8, 9, 10, 11]


if __name__ == "__main__":
    test_gleason_to_gg()
    test_ordinal_c_index()
    test_pairwise_auc()
    test_model_wiring()
    print("ALL CHECKS PASSED")
