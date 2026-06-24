from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from social_extract.ocr import _coerce_rapidocr_result


def test_rapidocr_numpy_array_result_is_coerced_without_truth_value_error() -> None:
    raw = SimpleNamespace(
        txts=np.array(["Slide title", "Body text"]),
        scores=np.array([0.9, 0.8]),
        boxes=np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]]),
    )

    result = _coerce_rapidocr_result(raw)

    assert result.texts == ["Slide title", "Body text"]
    assert result.scores == [0.9, 0.8]
    assert result.confidence == pytest.approx(0.85)
    assert result.boxes == [[[0, 0], [1, 0], [1, 1], [0, 1]]]
