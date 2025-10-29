import os
import shutil
import sys
from pathlib import Path
from typing import Dict

from BIT_ADMIT_AI.entity.artifact import ModelPusherArtifact, ModelTrainerArtifact
from BIT_ADMIT_AI.entity.config import ModelPusherConfig
from BIT_ADMIT_AI.exceptions import BitAdmitAIException
from BIT_ADMIT_AI.logger import logging
from BIT_ADMIT_AI.utils.main_utils import write_yaml_file


class ModelPusher:
    def __init__(
        self,
        model_trainer_artifact: ModelTrainerArtifact,
        model_pusher_config: ModelPusherConfig,
    ) -> None:
        self.model_trainer_artifact = model_trainer_artifact
        self.model_pusher_config = model_pusher_config

    def push_model(
        self,
        metrics_per_target: Dict[str, Dict[str, float]],
        avg_f1_score: float,
    ) -> ModelPusherArtifact:
        try:
            os.makedirs(self.model_pusher_config.best_model_dir, exist_ok=True)

            source_path = Path(self.model_trainer_artifact.trained_model_file_path)
            destination_path = Path(self.model_pusher_config.best_model_path)
            shutil.copy2(source_path, destination_path)

            metadata = {
                "avg_f1_score": avg_f1_score,
                "metrics_per_target": metrics_per_target,
            }
            write_yaml_file(
                file_path=self.model_pusher_config.best_model_metrics_path,
                content=metadata,
                replace=True,
            )

            logging.info(
                "Updated best model at %s with avg F1 %.4f",
                destination_path,
                avg_f1_score,
            )

            return ModelPusherArtifact(
                best_model_path=str(destination_path),
                best_model_metrics_path=self.model_pusher_config.best_model_metrics_path,
            )

        except Exception as exc:
            raise BitAdmitAIException(exc, sys) from exc
