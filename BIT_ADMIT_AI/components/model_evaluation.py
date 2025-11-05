"""Model evaluation component.

Compares the newly trained model against the current best using average F1 across
targets. Accepts and pushes the new model if the improvement meets the configured
threshold, and returns a ModelEvaluationArtifact summarizing the decision.
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from BIT_ADMIT_AI.entity.artifact import (
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from BIT_ADMIT_AI.entity.config import (
    ModelEvaluationConfig,
    ModelPusherConfig,
)
from BIT_ADMIT_AI.exceptions import BitAdmitAIException
from BIT_ADMIT_AI.logger import logging
from BIT_ADMIT_AI.utils.main_utils import read_yaml_file


class ModelEvaluation:
    """Evaluate a trained model against the current best and optionally promote it.

    Args:
        model_trainer_artifact: Outputs from training (path, per-target metrics).
        model_evaluation_config: Thresholds and best-model metadata paths.
        model_pusher_config: Destinations for promoted model and its metrics.

    Attributes:
        model_trainer_artifact: Training outputs for the candidate model.
        model_evaluation_config: Evaluation parameters and paths.
        model_pusher_config: Promotion target locations.
    """

    def __init__(
        self,
        model_trainer_artifact: ModelTrainerArtifact,
        model_evaluation_config: ModelEvaluationConfig,
        model_pusher_config: ModelPusherConfig,
    ) -> None:
        self.model_trainer_artifact = model_trainer_artifact
        self.model_evaluation_config = model_evaluation_config
        self.model_pusher_config = model_pusher_config

    @staticmethod
    def _average_f1(metrics: Dict[str, Dict[str, float]]) -> float:
        scores = [target.get("f1_score", 0.0) for target in metrics.values()]
        return float(np.mean(scores)) if scores else 0.0

    def _load_best_metrics(self) -> Optional[Dict[str, float]]:
        try:
            metrics_path = Path(self.model_evaluation_config.best_model_metrics_path)
            if metrics_path.is_file():
                return read_yaml_file(str(metrics_path))
            return None
        except Exception as exc:
            raise BitAdmitAIException(exc, sys) from exc

    def evaluate_model(self) -> ModelEvaluationArtifact:
        """Compare candidate to best model and optionally push the candidate.

        Logic:
        - compute candidate avg F1 over targets,
        - load prior best avg F1 if available,
        - accept if improvement >= change_threshold (or no prior best),
        - on acceptance, push the model(in our case save it to dir) and persist metrics.

        Returns:
            ModelEvaluationArtifact: Decision, improvement, and reference paths.

        Raises:
            BitAdmitAIException: If evaluation or push fails.
        """
        try:
            current_metrics = self.model_trainer_artifact.metrics_per_target
            current_avg_f1 = self._average_f1(current_metrics)

            previous_metrics = self._load_best_metrics()
            previous_avg_f1 = (
                float(previous_metrics.get("avg_f1_score", 0.0))
                if previous_metrics
                else None
            )

            improved_metric = (
                current_avg_f1 - previous_avg_f1
                if previous_avg_f1 is not None
                else current_avg_f1
            )

            is_model_accepted = False
            if previous_avg_f1 is None:
                logging.info("No prior best model, accepting current model")
                is_model_accepted = True
            elif improved_metric >= self.model_evaluation_config.change_threshold:
                logging.info(
                    "Current model improved F1 by %.4f (threshold %.4f)",
                    improved_metric,
                    self.model_evaluation_config.change_threshold,
                )
                is_model_accepted = True
            else:
                logging.info(
                    "Model improvement %.4f below threshold %.4f; retaining previous best",
                    improved_metric,
                    self.model_evaluation_config.change_threshold,
                )

            if is_model_accepted:
                from BIT_ADMIT_AI.components.model_pusher import ModelPusher

                pusher = ModelPusher(
                    model_trainer_artifact=self.model_trainer_artifact,
                    model_pusher_config=self.model_pusher_config,
                )
                pusher.push_model(current_metrics, current_avg_f1)

            return ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                improved_metric=improved_metric,
                current_metrics=current_metrics,
                best_model_path=self.model_pusher_config.best_model_path,
                best_model_metrics_path=self.model_pusher_config.best_model_metrics_path,
            )

        except Exception as exc:
            raise BitAdmitAIException(exc, sys) from exc
