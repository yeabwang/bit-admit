import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from BIT_ADMIT_AI.constant import TARGET_COLUMNS, BEST_MODEL_PATH
from BIT_ADMIT_AI.entity.estimator import TargetValueMap
from BIT_ADMIT_AI.exceptions import BitAdmitAIException
from BIT_ADMIT_AI.logger import logging
from BIT_ADMIT_AI.utils.main_utils import load_object


@dataclass
class BitAdmitFeatures:
	"""Container for incoming application features."""

	program_category: str
	country: str
	bit_program_applied: str
	degree_language: str
	previous_gpa: float
	math_physics_background_score: float
	research_alignment_score: float
	publication_count: float
	recommendation_strength: float
	interview_score: float
	english_test_type: str
	english_score: float
	chinese_proficiency: str

	def to_dataframe(self) -> pd.DataFrame:
		return pd.DataFrame([asdict(self)])

	def to_dict(self) -> Dict[str, object]:
		return asdict(self)


class BitAdmitClassifier:
	def __init__(self, model_path: str = BEST_MODEL_PATH) -> None:
		try:
			self.model_bundle = load_object(model_path)
			self.preprocessor = self.model_bundle["preprocessor"]
			self.models = self.model_bundle["models"]
			self.target_columns = self.model_bundle.get("target_columns", TARGET_COLUMNS)
			self.target_value_map: TargetValueMap = self.model_bundle["target_value_map"]
		except Exception as exc:
			raise BitAdmitAIException(exc, sys) from exc

	def predict(self, features: BitAdmitFeatures) -> Dict[str, str]:
		try:
			input_df = features.to_dataframe()
			transformed_features = self.preprocessor.transform(input_df)

			predictions: Dict[str, str] = {}

			for target_name in self.target_columns:
				model = self.models[target_name]
				encoded_prediction = model.predict(transformed_features)
				decoded = self.target_value_map.decode_prediction(
					encoded_prediction, [target_name]
				)
				predictions[target_name] = decoded[target_name]

			return predictions

		except Exception as exc:
			raise BitAdmitAIException(exc, sys) from exc