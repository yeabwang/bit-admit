"""Data transformation component.

Prepares model-ready arrays and persists transformation artifacts:
- reads train/test CSVs,
- engineers features and standardizes strings,
- builds/fits a ColumnTransformer (num: impute+scale, cat: impute+OHE),
- encodes target columns,
- saves preprocessor bundle and transformed arrays.

Relies on schema.yaml (columns, drops) and a passing DataValidationArtifact.
Outputs are written to paths from DataTransformationConfig.
"""

import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from BIT_ADMIT_AI.constant import SCHEMA_PATH, TARGET_COLUMNS
from BIT_ADMIT_AI.entity.artifact import (
    DAArtifacts,
    DataTransformationArtifact,
    DataValidationArtifact,
)
from BIT_ADMIT_AI.entity.config import DataTransformationConfig
from BIT_ADMIT_AI.exceptions import BitAdmitAIException
from BIT_ADMIT_AI.logger import logging
from BIT_ADMIT_AI.utils.main_utils import (
    drop_columns,
    read_yaml_file,
    save_numpy_array_data,
    save_object,
)


class DataTransformation:
    """Prepare feature/target arrays and persist preprocessing artifacts."""

    def __init__(
        self,
        data_ingestion_artifact: DAArtifacts,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact,
    ) -> None:
        """Initialize the component with artifacts and config.

        Args:
            data_ingestion_artifact: Train/test file paths from ingestion.
            data_transformation_config: Output locations for arrays/objects.
            data_validation_artifact: Validation result gating transformation.

        Raises:
            BitAdmitAIException: If schema/config loading fails.
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_PATH)
        except Exception as exc:  # pragma: no cover - setup failure
            raise BitAdmitAIException(exc, sys) from exc

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """Read a CSV into a DataFrame.

        Args:
            file_path: Path to the CSV.

        Returns:
            pandas.DataFrame: Loaded data.

        Raises:
            BitAdmitAIException: If read fails.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as exc:
            raise BitAdmitAIException(exc, sys) from exc

    @staticmethod
    def _standardize_strings(series: pd.Series) -> pd.Series:
        return series.astype(str).str.strip().str.lower().str.replace("-", "_")

    @staticmethod
    def _language_requirement_passed(row: pd.Series) -> int:
        degree_language = row.get("degree_language", "")
        english_type = row.get("english_test_type", "")
        english_score = row.get("english_score", 0.0)
        chinese_level = row.get("chinese_proficiency", 0)

        if degree_language == "english_taught":
            if english_type == "duolingo" and english_score >= 90:
                return 1
            if english_type == "toefl" and english_score >= 90:
                return 1
            if english_type == "ielts" and english_score >= 6.5:
                return 1
            return 0

        if degree_language == "chinese_taught":
            return 1 if chinese_level >= 4 else 0

        return 0

    @staticmethod
    def _weighted_score(row: pd.Series) -> float:
        category = row.get("program_category", "")
        gpa = row.get("previous_gpa", 0.0)
        math_phys = row.get("math_physics_background_score", 0.0)
        research_alignment = row.get("research_alignment_score", 0.0)
        publications = min(row.get("publication_count", 0.0), 5.0)
        recommendation = row.get("recommendation_strength", 0.0)
        interview = row.get("interview_score", 0.0)

        if category == "undergraduate":
            return (
                0.40 * gpa + 0.30 * math_phys + 0.10 * recommendation + 0.20 * interview
            )

        if category == "postgraduate":
            return (
                0.40 * gpa
                + 0.30 * research_alignment
                + 0.10 * publications
                + 0.10 * recommendation
                + 0.10 * interview
            )

        # Chinese language & dual degree programs share weighting
        return 0.50 * gpa + 0.20 * recommendation + 0.30 * interview

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        processed = df.copy()

        for column in ["program_category", "degree_language", "english_test_type"]:
            if column in processed.columns:
                processed[column] = self._standardize_strings(processed[column])

        if "publication_count" in processed.columns:
            processed["publication_count"] = np.log1p(
                processed["publication_count"].clip(lower=0)
            )

        if "chinese_proficiency" in processed.columns:
            processed["chinese_proficiency"] = (
                self._standardize_strings(processed["chinese_proficiency"])
                .str.replace("hsk", "", regex=False)
                .replace("", np.nan)
                .astype(float)
            )

        processed["language_requirement_passed"] = processed.apply(
            self._language_requirement_passed, axis=1
        )
        processed["weighted_score"] = processed.apply(self._weighted_score, axis=1)

        drop_candidate = set(self._schema_config.get("dropped_columns", []))
        protected = set(self._schema_config.get("target_columns", []))
        protected.update({"program_category", "degree_language", "english_test_type"})
        drop_list = [
            col
            for col in drop_candidate
            if col in processed.columns and col not in protected
        ]

        if drop_list:
            processed = drop_columns(processed, drop_list)

        return processed

    @staticmethod
    def _split_features_targets(
        df: pd.DataFrame, target_columns: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        target_df = df[target_columns].copy()
        feature_df = df.drop(columns=target_columns, errors="ignore")
        return feature_df, target_df

    @staticmethod
    def _build_preprocessor(
        numeric_columns: List[str], categorical_columns: List[str]
    ) -> ColumnTransformer:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("numerical", numeric_pipeline, numeric_columns),
                ("categorical", categorical_pipeline, categorical_columns),
            ],
            remainder="drop",
        )

    def _encode_targets(
        self,
        train_targets: pd.DataFrame,
        test_targets: pd.DataFrame,
        target_columns: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder]]:
        encoders: Dict[str, LabelEncoder] = {}

        encoded_train = train_targets.copy()
        encoded_test = test_targets.copy()

        for column in target_columns:
            encoder = LabelEncoder()
            encoded_train[column] = encoder.fit_transform(train_targets[column])
            encoded_test[column] = encoder.transform(test_targets[column])
            encoders[column] = encoder

        return encoded_train, encoded_test, encoders

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """Run the transformation pipeline and persist artifacts.

        Steps:
        - validate gate, load train/test,
        - engineer features, select columns from schema,
        - fit/transform with ColumnTransformer,
        - label-encode targets,
        - save preprocessor bundle and numpy arrays.

        Returns:
            DataTransformationArtifact: Paths to saved object and arrays.

        Raises:
            BitAdmitAIException: On validation failure or any processing/IO error.
        """
        try:
            if not self.data_validation_artifact.validation_status:
                raise BitAdmitAIException(self.data_validation_artifact.message, sys)

            logging.info("Starting data transformation stage")

            train_df = self.read_data(self.data_ingestion_artifact.training_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            feature_train_df, target_train_df = self._split_features_targets(
                train_df, TARGET_COLUMNS
            )
            feature_test_df, target_test_df = self._split_features_targets(
                test_df, TARGET_COLUMNS
            )

            feature_train_df = self._engineer_features(feature_train_df)
            feature_test_df = self._engineer_features(feature_test_df)

            numeric_cols = [
                col
                for col in self._schema_config.get("numerical_columns", [])
                if col in feature_train_df.columns
            ]
            numeric_cols.extend(
                [
                    column
                    for column in self._schema_config.get("engineered_columns", [])
                    if column in feature_train_df.columns
                ]
            )
            numeric_cols = sorted(set(numeric_cols))

            categorical_cols = [
                col
                for col in self._schema_config.get("categorical_columns", [])
                if col in feature_train_df.columns and col not in TARGET_COLUMNS
            ]
            categorical_cols = [
                col
                for col in categorical_cols
                if col not in set(self._schema_config.get("dropped_columns", []))
            ]

            preprocessor = self._build_preprocessor(numeric_cols, categorical_cols)

            logging.info("Fitting preprocessing pipeline on training features")
            input_feature_train_arr = preprocessor.fit_transform(feature_train_df)
            input_feature_test_arr = preprocessor.transform(feature_test_df)

            encoded_train_targets, encoded_test_targets, encoders = (
                self._encode_targets(target_train_df, target_test_df, TARGET_COLUMNS)
            )

            train_arr = np.concatenate(
                [input_feature_train_arr, encoded_train_targets.values], axis=1
            )
            test_arr = np.concatenate(
                [input_feature_test_arr, encoded_test_targets.values], axis=1
            )

            transformed_feature_names = []
            try:
                transformed_feature_names = (
                    preprocessor.get_feature_names_out().tolist()
                )
            except Exception:  # pragma: no cover - optional metadata
                pass

            preprocessor_bundle = {
                "preprocessor": preprocessor,
                "target_encoders": encoders,
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "transformed_feature_names": transformed_feature_names,
            }

            save_object(
                file_path=self.data_transformation_config.transformed_object_file_path,
                obj=preprocessor_bundle,
            )
            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_train_file_path,
                array=train_arr,
            )
            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_test_file_path,
                array=test_arr,
            )

            logging.info("Data transformation finished successfully")

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )

        except Exception as exc:
            raise BitAdmitAIException(exc, sys) from exc
