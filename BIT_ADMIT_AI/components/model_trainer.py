import importlib
import sys
from typing import Dict

import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_score

from BIT_ADMIT_AI.constant import TARGET_COLUMNS
from BIT_ADMIT_AI.entity.artifact import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from BIT_ADMIT_AI.entity.config import ModelTrainerConfig
from BIT_ADMIT_AI.entity.estimator import TargetValueMap
from BIT_ADMIT_AI.exceptions import BitAdmitAIException
from BIT_ADMIT_AI.logger import logging
from BIT_ADMIT_AI.utils.main_utils import (
    load_numpy_array_data,
    load_object,
    read_yaml_file,
    save_object,
)


class ModelTrainer:
    """Train per-target classifiers and persist the inference bundle."""

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ) -> None:
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
            self.model_config = read_yaml_file(
                self.model_trainer_config.model_config_file_path
            )
        except Exception as exc:
            raise BitAdmitAIException(exc, sys) from exc

    def _load_transformed_datasets(self):
        try:
            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )
            preprocessor_bundle = load_object(
                self.data_transformation_artifact.transformed_object_file_path
            )
            return train_arr, test_arr, preprocessor_bundle
        except Exception as exc:
            raise BitAdmitAIException(exc, sys) from exc

    @staticmethod
    def _split_features_targets(dataset, feature_count):
        features = dataset[:, :feature_count]
        targets = dataset[:, feature_count:]
        return features, targets

    @staticmethod
    def _instantiate_estimator(model_info: Dict, num_classes: int):
        module = importlib.import_module(model_info["module"])
        estimator_cls = getattr(module, model_info["class"])
        params = model_info.get("params", {}).copy()
        params.pop("use_label_encoder", None)  # Drop deprecated XGBoost flag

        if model_info["module"].startswith("xgboost") and num_classes > 2:
            params.setdefault("objective", "multi:softprob")
            params.setdefault("num_class", num_classes)
            params.setdefault("eval_metric", "mlogloss")

        return estimator_cls(**params)

    def _run_grid_search(self, estimator, search_params, X, y):
        grid_cfg = self.model_config.get("grid_search", {})
        if not grid_cfg or not search_params:
            estimator.fit(X, y)
            return estimator, {}

        grid_kwargs = grid_cfg.get("params", {}).copy()
        grid_kwargs.pop("verbose", None)

        cv_splits = int(grid_kwargs.pop("cv", 3))
        n_jobs = grid_kwargs.pop("n_jobs", None)
        scoring = grid_kwargs.pop("scoring", None)
        shuffle = grid_kwargs.pop("shuffle", True)
        random_state = grid_kwargs.pop("random_state", 42)
        snapshot_interval = float(grid_kwargs.pop("snapshot_interval", 0.25))
        snapshot_interval = min(max(snapshot_interval, 0.05), 1.0)

        if grid_kwargs:
            logging.debug("Unused grid params ignored: %s", grid_kwargs)

        param_grid = list(ParameterGrid(search_params))
        total_combos = len(param_grid)
        if total_combos == 0:
            estimator.fit(X, y)
            return estimator, {}

        cv_strategy = StratifiedKFold(
            n_splits=cv_splits,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )

        snapshot_steps = [
            round(step, 2)
            for step in np.arange(
                snapshot_interval, 1.0 + snapshot_interval, snapshot_interval
            )
            if step <= 1.0 + 1e-6
        ]
        snapshot_index = 0

        best_score = -np.inf
        best_params: Dict = {}
        best_estimator = None

        logging.info(
            "Grid search over %d combinations (cv=%d) for estimator %s",
            total_combos,
            cv_splits,
            estimator.__class__.__name__,
        )

        for combo_idx, param_set in enumerate(param_grid, start=1):
            estimator_clone = clone(estimator)
            estimator_clone.set_params(**param_set)
            scores = cross_val_score(
                estimator_clone,
                X,
                y,
                cv=cv_strategy,
                scoring=scoring,
                n_jobs=n_jobs,
            )
            mean_score = float(np.mean(scores))

            if mean_score > best_score:
                best_score = mean_score
                best_params = param_set
                best_estimator = estimator_clone

            progress_ratio = combo_idx / total_combos
            if (
                snapshot_index < len(snapshot_steps)
                and progress_ratio >= snapshot_steps[snapshot_index]
            ):
                logging.info(
                    "Grid search progress: %.0f%% complete (%d / %d)",
                    progress_ratio * 100,
                    combo_idx,
                    total_combos,
                )
                snapshot_index += 1

        if best_estimator is None:
            raise BitAdmitAIException(
                "Grid search failed to locate a valid estimator", sys
            )

        logging.info(
            "Grid search completed. Best CV score: %.4f with params: %s",
            best_score,
            best_params,
        )

        best_estimator.fit(X, y)

        return best_estimator, best_params

    @staticmethod
    def _calculate_metrics(y_true, y_pred) -> Dict[str, float]:
        return {
            "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "precision_score": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall_score": recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
        }

    def _train_for_target(self, target_name, X_train, y_train, X_test, y_test):
        num_classes = len(np.unique(y_train))
        best_model = None
        best_metrics: Dict[str, float] = {}
        best_score = -np.inf

        model_candidates = self.model_config.get("model_selection", {})
        if not model_candidates:
            raise BitAdmitAIException(
                f"No model candidates provided for target '{target_name}'",
                sys,
            )

        for model_key, model_info in model_candidates.items():
            estimator = self._instantiate_estimator(model_info, num_classes)
            search_grid = model_info.get("search_param_grid", {})
            trained_estimator, best_params = self._run_grid_search(
                estimator, search_grid, X_train, y_train
            )

            y_pred = trained_estimator.predict(X_test)
            metrics = self._calculate_metrics(y_test, y_pred)

            logging.info(
                "Model '%s' for target '%s' achieved metrics: %s",
                model_key,
                target_name,
                metrics,
            )
            if best_params:
                logging.info("Best params for %s: %s", model_key, best_params)

            if metrics["f1_score"] > best_score:
                best_score = metrics["f1_score"]
                best_model = trained_estimator
                best_metrics = metrics

        if best_model is None:
            raise BitAdmitAIException(
                f"Unable to train a model for target '{target_name}'",
                sys,
            )

        return best_model, best_metrics

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr, test_arr, preprocessor_bundle = self._load_transformed_datasets()

            feature_count = train_arr.shape[1] - len(TARGET_COLUMNS)
            X_train, y_train_matrix = self._split_features_targets(
                train_arr, feature_count
            )
            X_test, y_test_matrix = self._split_features_targets(
                test_arr, feature_count
            )

            best_models: Dict[str, object] = {}
            metrics_per_target: Dict[str, Dict[str, float]] = {}

            for idx, target_name in enumerate(TARGET_COLUMNS):
                y_train = y_train_matrix[:, idx].astype(int)
                y_test = y_test_matrix[:, idx].astype(int)

                best_model, target_metrics = self._train_for_target(
                    target_name, X_train, y_train, X_test, y_test
                )

                best_models[target_name] = best_model
                metrics_per_target[target_name] = target_metrics

            target_value_map = TargetValueMap(
                preprocessor_bundle.get("target_encoders", {})
            )

            model_package = {
                "preprocessor": preprocessor_bundle.get("preprocessor"),
                "models": best_models,
                "target_value_map": target_value_map,
                "target_columns": TARGET_COLUMNS,
                "feature_names": preprocessor_bundle.get(
                    "transformed_feature_names", []
                ),
            }

            save_object(
                self.model_trainer_config.trained_model_file_path,
                model_package,
            )

            logging.info("Model training metrics per target: %s", metrics_per_target)

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metrics_per_target=metrics_per_target,
            )

        except Exception as exc:
            raise BitAdmitAIException(exc, sys) from exc
