from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


@dataclass
class NumericPreprocessor:
    use_ica: bool
    ica_components: int | str | None
    seed: int

    def __post_init__(self) -> None:
        self.columns_: list[str] = []
        self.imputer_: SimpleImputer | None = None
        self.scaler_: StandardScaler | None = None
        self.ica_: FastICA | None = None
        self.resolved_ica_components_: int | None = None

    def _numeric_frame(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        if fit:
            excluded = {"__source_file__"}
            self.columns_ = [
                column
                for column in X.select_dtypes(include=[np.number, "bool"]).columns
                if column not in excluded
            ]
            if not self.columns_:
                raise ValueError("No numeric feature columns were found.")
        missing = [column for column in self.columns_ if column not in X.columns]
        if missing:
            raise ValueError(f"Dataset is missing fitted feature columns: {missing}")
        return X[self.columns_].replace([np.inf, -np.inf], np.nan)

    def fit(self, X: pd.DataFrame) -> "NumericPreprocessor":
        numeric = self._numeric_frame(X, fit=True)
        self.imputer_ = SimpleImputer(strategy="median", add_indicator=False)
        imputed = self.imputer_.fit_transform(numeric)
        self.scaler_ = StandardScaler()
        scaled = self.scaler_.fit_transform(imputed)
        if self.use_ica:
            if self.ica_components in {None, "auto"}:
                components = min(20, scaled.shape[1], max(1, scaled.shape[0] - 1))
            else:
                components = int(self.ica_components)
            if not 1 <= components <= min(scaled.shape):
                raise ValueError(
                    f"ICA components {components} invalid for matrix shape {scaled.shape}."
                )
            self.resolved_ica_components_ = components
            self.ica_ = FastICA(
                n_components=components,
                whiten="unit-variance",
                max_iter=1000,
                tol=1e-4,
                random_state=self.seed,
            )
            self.ica_.fit(scaled)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.imputer_ is None or self.scaler_ is None:
            raise RuntimeError("Preprocessor must be fitted before transform.")
        numeric = self._numeric_frame(X, fit=False)
        scaled = self.scaler_.transform(self.imputer_.transform(numeric))
        return self.ica_.transform(scaled) if self.ica_ is not None else scaled

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

