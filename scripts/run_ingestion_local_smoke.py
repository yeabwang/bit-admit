"""Smoke test for local CSV fallback ingestion.

Run this when you do NOT have a valid .env or MongoDB connection.
It will execute the DataIngestion pipeline and print artifact paths.
"""

from BIT_ADMIT_AI.components.data_ingestion import DataIngestion


def main():
    artifact = DataIngestion().init_data_ingestion()
    print("Train path:", artifact.training_file_path)
    print("Test path:", artifact.test_file_path)


if __name__ == "__main__":
    main()
