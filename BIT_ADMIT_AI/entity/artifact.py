from dataclasses import dataclass


@dataclass
class DAArtifacts:
    training_file_path: str = ""
    test_file_path: str = ""
