import numpy as np
import pandas as pd

from trustfed_incboost.data.audit import dataframe_fingerprint
from trustfed_incboost.data.splitting import make_split


def test_fingerprint_disjoint_split_keeps_duplicates_together() -> None:
    rng = np.random.default_rng(5)
    X = pd.DataFrame(rng.normal(size=(200, 6)), columns=list("abcdef"))
    y = pd.Series(np.where(np.arange(200) % 5 == 0, 1, 0))
    X = pd.concat([X, X.iloc[:20]], ignore_index=True)
    y = pd.concat([y, y.iloc[:20]], ignore_index=True)
    split = make_split(X, y, "fingerprint_disjoint", 42, 0.2, 0.2)
    hashes = dataframe_fingerprint(X)
    train = set(hashes.iloc[split.train])
    validation = set(hashes.iloc[split.validation])
    test = set(hashes.iloc[split.test])
    assert not train & validation
    assert not train & test
    assert not validation & test


def test_random_split_covers_every_row() -> None:
    X = pd.DataFrame({"a": np.arange(100), "b": np.arange(100) ** 2})
    y = pd.Series([0, 1] * 50)
    split = make_split(X, y, "random_stratified", 42, 0.2, 0.2)
    combined = np.concatenate([split.train, split.validation, split.test])
    assert len(np.unique(combined)) == len(X)

