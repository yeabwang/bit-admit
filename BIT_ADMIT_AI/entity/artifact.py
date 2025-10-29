from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class DAArtifacts:
    training_file_path: str = ""
    test_file_path: str = ""


@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str
    drift_report_file_path: str


@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str


@dataclass
class ClassMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    metrics_per_target: Dict[str, Dict[str, float]]
    metric_artifact: Optional[ClassMetricArtifact] = None


@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    improved_metric: float
    current_metrics: Dict[str, Dict[str, float]]
    best_model_path: str
    best_model_metrics_path: str


@dataclass
class ModelPusherArtifact:
    best_model_path: str
    best_model_metrics_path: str
