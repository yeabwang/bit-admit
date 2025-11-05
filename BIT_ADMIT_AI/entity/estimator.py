"""Target encoding utilities

Wraps a scikit-learn LabelEncoder per target column and:
- fit/transform/inverse_transform for target DataFrames,
- decoding of encoded predictions,
- access to integer-to-label mappings.

All methods raise BitAdmitAIException on failure.
"""

import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from BIT_ADMIT_AI.exceptions import BitAdmitAIException


class TargetValueMap:
    """Encode/decode categorical target columns with one LabelEncoder per column.

    Args:
        encoders: Dict mapping target column name to a fitted LabelEncoder.

    Attributes:
        _encoders: Internal mapping of column -> LabelEncoder.
    """

    def __init__(self, encoders: Dict[str, LabelEncoder]):
        self._encoders = encoders

    @classmethod
    def fit(cls, targets: pd.DataFrame) -> Tuple["TargetValueMap", pd.DataFrame]:
        """Fit encoders on target columns and return encoded targets.

        Args:
            targets: DataFrame of categorical target columns.

        Returns:
            Tuple of:
            - TargetValueMap: Fitted map object.
            - pandas.DataFrame: Encoded targets.

        Raises:
            BitAdmitAIException: If fitting fails.
        """
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
        """Encode target columns using fitted encoders.

        Args:
            targets: DataFrame with the same target columns seen during fit.

        Returns:
            pandas.DataFrame: Encoded targets.

        Raises:
            BitAdmitAIException: If transform fails or unseen labels are present.
        """
        try:
            encoded = targets.copy()
            for column, encoder in self._encoders.items():
                encoded[column] = encoder.transform(targets[column])
            return encoded
        except Exception as exc:
            raise BitAdmitAIException(exc, sys) from exc

    def inverse_transform(self, encoded: pd.DataFrame) -> pd.DataFrame:
        """Decode integer-encoded targets back to original labels.

        Args:
            encoded: DataFrame of encoded targets.

        Returns:
            pandas.DataFrame: Decoded targets with original labels.

        Raises:
            BitAdmitAIException: If decoding fails.
        """
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
        """Decode a sequence of predicted integers to label strings.

        Args:
            encoded_values: Iterable of encoded predictions (ints).
            order: Column order corresponding to encoded_values.

        Returns:
            Dict[str, str]: Mapping column -> decoded label.

        Raises:
            BitAdmitAIException: If decoding fails.
        """
        try:
            decoded = {}
            for column, value in zip(order, encoded_values):
                encoder = self._encoders[column]
                decoded[column] = encoder.inverse_transform(np.array([value]))[0]
            return decoded
        except Exception as exc:
            raise BitAdmitAIException(exc, sys) from exc

    def mapping(self) -> Dict[str, Dict[int, str]]:
        """Get integer-to-label mapping for each target column.

        Returns:
            Dict[column, Dict[encoded_int, label_str]].

        """
        mapping: Dict[str, Dict[int, str]] = {}
        for column, encoder in self._encoders.items():
            encoded_values = encoder.transform(encoder.classes_)
            mapping[column] = dict(
                zip(encoded_values.tolist(), encoder.classes_.tolist())
            )
        return mapping

    def encoders(self) -> Dict[str, LabelEncoder]:
        """Expose the underlying encoders.

        Returns:
            Dict[str, LabelEncoder]: Column -> fitted encoder.
        """
        return self._encoders
