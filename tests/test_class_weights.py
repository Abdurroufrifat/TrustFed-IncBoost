import numpy as np
import pandas as pd

from trustfed_incboost.models.class_weights import class_weight_map, sample_weights


def test_paper_printed_matches_equation_eight() -> None:
    y = pd.Series([0] * 8 + [1] * 2)
    weights = class_weight_map(y, "paper_printed")
    assert np.isclose(weights[0], 1.6)
    assert np.isclose(weights[1], 0.4)


def test_balanced_gives_larger_minority_weight() -> None:
    y = pd.Series([0] * 8 + [1] * 2)
    weights = class_weight_map(y, "balanced")
    assert np.isclose(weights[0], 10 / 16)
    assert np.isclose(weights[1], 10 / 4)
    assert weights[1] > weights[0]
    samples, _ = sample_weights(y, "balanced")
    assert samples.shape == (10,)

