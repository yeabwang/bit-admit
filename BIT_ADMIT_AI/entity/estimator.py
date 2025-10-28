import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from BIT_ADMIT_AI.exceptions import BitAdmitAIException


class TargetValueMap:
    """Utility for encoding and decoding target labels.

    We keep the fitted ``LabelEncoder`` instances so that downstream
    components.
    """

    def __init__(self, encoders: Dict[str, LabelEncoder]):
        self._encoders = encoders

    @classmethod
    def fit(cls, targets: pd.DataFrame) -> Tuple["TargetValueMap", pd.DataFrame]:
        try:
            encoded = targets.copy()
            encoders: Dict[str, LabelEncoder] = {}

            for column in targets.columns:
                encoder = LabelEncoder()
                encoded[column] = encoder.fit_transform(targets[column])
                encoders[column] = encoder

            return cls(encoders), encoded
        except Exception as exc:
            raise BitAdmitAIException(exc, sys) from exc

    def transform(self, targets: pd.DataFrame) -> pd.DataFrame:
        try:
            encoded = targets.copy()
            for column, encoder in self._encoders.items():
                encoded[column] = encoder.transform(targets[column])
            return encoded
        except Exception as exc:
            raise BitAdmitAIException(exc, sys) from exc

    def inverse_transform(self, encoded: pd.DataFrame) -> pd.DataFrame:
        try:
            decoded = encoded.copy()
            for column, encoder in self._encoders.items():
                decoded[column] = encoder.inverse_transform(encoded[column])
            return decoded
        except Exception as exc:
            raise BitAdmitAIException(exc, sys) from exc

    def decode_prediction(
        self, encoded_values: Iterable[int], order: List[str]
    ) -> Dict[str, str]:
        try:
            decoded = {}
            for column, value in zip(order, encoded_values):
                encoder = self._encoders[column]
                decoded[column] = encoder.inverse_transform(np.array([value]))[0]
            return decoded
        except Exception as exc:
            raise BitAdmitAIException(exc, sys) from exc

    def mapping(self) -> Dict[str, Dict[int, str]]:
        mapping: Dict[str, Dict[int, str]] = {}
        for column, encoder in self._encoders.items():
            encoded_values = encoder.transform(encoder.classes_)
            mapping[column] = dict(
                zip(encoded_values.tolist(), encoder.classes_.tolist())
            )
        return mapping

    def encoders(self) -> Dict[str, LabelEncoder]:
        return self._encoders
