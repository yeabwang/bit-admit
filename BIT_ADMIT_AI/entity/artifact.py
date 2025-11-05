"""Pipeline artifact dataclasses.

payloads passed between pipeline stages (ingestion, validation,
transformation, training, evaluation, pusher).
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class DAArtifacts:
    """Data ingestion artifacts.

    Attributes:
        training_file_path: Path to the persisted training split.
        test_file_path: Path to the persisted test split.
    """

    training_file_path: str = ""
    test_file_path: str = ""


@dataclass
class DataValidationArtifact:
    """Data validation outputs.

    Attributes:
        validation_status: True if validation passed.
        message: Summary of validation outcome.
        drift_report_file_path: Path to data drift report (YAML/HTML).
    """

    validation_status: bool
    message: str
    drift_report_file_path: str


@dataclass
class DataTransformationArtifact:
    """Data transformation outputs.

    Attributes:
        transformed_object_file_path: Path to fitted preprocessing object.
        transformed_train_file_path: Path to transformed training data.
        transformed_test_file_path: Path to transformed test data.
    """

    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str


@dataclass
class ClassMetricArtifact:
    """Classification metrics (macro or per-target).

    Attributes:
        f1_score: F1 score.
        precision_score: Precision.
        recall_score: Recall.
    """

    f1_score: float
    precision_score: float
    recall_score: float


@dataclass
class ModelTrainerArtifact:
    """Model training outputs.

    Attributes:
        trained_model_file_path: Path to the trained model artifact.
        metrics_per_target: Dict[target_name -> {metric_name: value}].
        metric_artifact: Optional aggregate ClassMetricArtifact.
    """

    trained_model_file_path: str
    metrics_per_target: Dict[str, Dict[str, float]]
    metric_artifact: Optional[ClassMetricArtifact] = None


@dataclass
class ModelEvaluationArtifact:
    """Model evaluation decision and references.

    Attributes:
        is_model_accepted: True if the new model beats the baseline.
        improved_metric: Absolute improvement vs. baseline threshold.
        current_metrics: Dict[target_name -> {metric_name: value}] for candidate.
        best_model_path: Path to the current best/production model.
        best_model_metrics_path: Path to metrics of the best model.
    """

    is_model_accepted: bool
    improved_metric: float
    current_metrics: Dict[str, Dict[str, float]]
    best_model_path: str
    best_model_metrics_path: str


@dataclass
class ModelPusherArtifact:
    """Model pusher/export outputs.

    Attributes:
        best_model_path: Path where the promoted model was saved.
        best_model_metrics_path: Path to the promoted model's metrics file.
    """

    best_model_path: str
    best_model_metrics_path: str
