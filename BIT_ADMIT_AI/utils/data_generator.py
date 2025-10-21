import pandas as pd
import numpy as np
import random
from datetime import datetime
from BIT_ADMIT_AI.logger import logging

np.random.seed(42)
random.seed(42)
n = 2000

time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

program_categories = [
    "Undergraduate",
    "Postgraduate",
    "Chinese Language",
    "Dual Degree",
]
undergraduate_programs = [
    "Aerospace Engineering",
    "Art and Design",
    "Automation",
    "Chemistry and Chemical Engineering",
    "Computer Science and Technology",
    "Cyberspace Science and Technology",
    "Economics",
    "Foreign Languages",
    "Humanities and Social Sciences",
    "Information and Electronics",
    "Law",
    "Life Science",
    "Management",
    "Materials Science and Engineering",
    "Mathematics and Statistics",
    "Mechanical Engineering",
    "Mechatronical Engineering",
    "Medical Technology",
    "Optics and Photonics",
    "Physics",
]
postgraduate_programs = [
    "Aeronautical and Astronautical Science and Technology",
    "Applied Economics",
    "Biomedical Engineering",
    "Business Administration",
    "Chemical Engineering and Technology",
    "Computer Science and Technology",
    "Control Science and Engineering",
    "Design",
    "Education",
    "Electronics Science and Technology",
    "Information and Communication Engineering",
    "Instrument Science and Technology",
    "Law",
    "Management Science and Engineering",
    "Materials Science and Technology",
    "Mathematics",
    "Mechanical Engineering",
    "Optical Engineering",
    "Physics",
    "Statistics",
    "Cyberspace Science and Technology",
]
chinese_language_programs = ["Chinese Language and Culture"]
dual_degree_programs = [
    "Joint Mechanical Engineering",
    "Joint Computer Science Program",
    "Joint Business Administration Program",
]
countries = [
    "Pakistan",
    "India",
    "Zimbabwe",
    "Russia",
    "France",
    "Kazakhstan",
    "Mongolia",
    "Thailand",
    "Kenya",
    "Brazil",
    "Egypt",
    "Morocco",
    "Ethiopia",
    "Malaysia",
    "Indonesia",
]
english_tests = ["IELTS", "TOEFL", "DUOLINGO"]


def select_program(cat):
    if cat == "Undergraduate":
        return random.choice(undergraduate_programs)
    if cat == "Postgraduate":
        return random.choice(postgraduate_programs)
    if cat == "Chinese Language":
        return random.choice(chinese_language_programs)
    return random.choice(dual_degree_programs)


def language_logic(cat):
    if cat in ["Undergraduate", "Postgraduate", "Dual Degree"]:
        return np.random.choice(["English-taught", "Chinese-taught"], p=[0.7, 0.3])
    return "Chinese-taught"


def gen_lang_score(test, quality_class):
    if test == "TOEFL":
        mean = {"high": 105, "mid": 92, "low": 75}[quality_class]
        sd = {"high": 8, "mid": 12, "low": 20}[quality_class]
        return float(np.clip(np.random.normal(mean, sd), 0, 120))
    if test == "IELTS":
        mean = {"high": 7.5, "mid": 6.5, "low": 5.0}[quality_class]
        sd = {"high": 0.4, "mid": 0.7, "low": 1.0}[quality_class]
        return float(np.clip(np.random.normal(mean, sd), 0, 9))
    if test == "DUOLINGO":
        mean = {"high": 125, "mid": 95, "low": 70}[quality_class]
        sd = {"high": 10, "mid": 20, "low": 30}[quality_class]
        return float(np.clip(np.random.normal(mean, sd), 0, 160))
    return 0.0


def assign_chinese_proficiency(lang, quality_class):
    if lang != "Chinese-taught":
        return np.random.choice(["HSK1", "HSK2", "HSK3"], p=[0.5, 0.3, 0.2])
    if quality_class == "high":
        return np.random.choice(["HSK5", "HSK6"], p=[0.6, 0.4])
    if quality_class == "mid":
        return np.random.choice(["HSK4", "HSK5"], p=[0.6, 0.4])
    return np.random.choice(["HSK3", "HSK4"], p=[0.7, 0.3])


def assign_targets(row):
    if row.previous_gpa < 2.5:
        return pd.Series(["Rejected", "No Scholarship"])
    if row.recommendation_strength < 6 or row.interview_score < 60:
        return pd.Series(["Rejected", "No Scholarship"])
    if row.degree_language == "English-taught":
        if (
            (row.english_test_type == "TOEFL" and row.english_score < 90)
            or (row.english_test_type == "IELTS" and row.english_score < 6)
            or (row.english_test_type == "DUOLINGO" and row.english_score < 90)
        ):
            return pd.Series(["Rejected", "No Scholarship"])
    if row.degree_language == "Chinese-taught" and row.chinese_proficiency not in [
        "HSK4",
        "HSK5",
        "HSK6",
    ]:
        return pd.Series(["Rejected", "No Scholarship"])
    gpa = row.previous_gpa / 4
    rec = row.recommendation_strength / 10
    inter = row.interview_score / 100
    research = row.research_alignment_score / 10
    math = row.math_physics_background_score / 10
    pub = min(row.publication_count / 5, 1)
    if row.program_category == "Postgraduate":
        score = 10 * (0.4 * gpa + 0.3 * research + 0.1 * pub + 0.1 * rec + 0.1 * inter)
    elif row.program_category == "Undergraduate":
        score = 10 * (0.4 * gpa + 0.3 * math + 0.1 * rec + 0.2 * inter)
    else:
        score = 10 * (0.5 * gpa + 0.2 * rec + 0.3 * inter)
    if score < 6.5:
        return pd.Series(["Rejected", "No Scholarship"])
    if score < 7.5:
        return pd.Series(["Admitted", "No Scholarship"])
    if score < 8.5:
        if row.previous_gpa >= 3.2 and row.recommendation_strength >= 7:
            return pd.Series(["Admitted", "Partial Scholarship"])
        return pd.Series(["Admitted", "No Scholarship"])
    if (
        row.previous_gpa >= 3.6
        and row.recommendation_strength >= 8
        and row.interview_score >= 85
    ):
        return pd.Series(["Admitted", "Full Scholarship"])
    return pd.Series(["Admitted", "Partial Scholarship"])


def generate_dataset():
    df = pd.DataFrame(
        {
            "application_id": [f"BIT2025{str(i).zfill(4)}" for i in range(1, n + 1)],
            "program_category": np.random.choice(
                program_categories, n, p=[0.4, 0.35, 0.15, 0.1]
            ),
            "country": np.random.choice(countries, n),
            "quality_class": np.random.choice(
                ["low", "mid", "high"], n, p=[0.5, 0.35, 0.15]
            ),
        }
    )
    df["bit_program_applied"] = df["program_category"].apply(select_program)
    df["degree_language"] = df["program_category"].apply(language_logic)

    # GPA - quality correlated
    df["previous_gpa"] = np.round(
        np.where(
            df["quality_class"] == "high",
            np.random.normal(3.7, 0.15, n),
            np.where(
                df["quality_class"] == "mid",
                np.random.normal(3.2, 0.3, n),
                np.random.normal(2.6, 0.4, n),
            ),
        ),
        2,
    )

    # Math/Physics background - quality correlated
    df["math_physics_background_score"] = np.round(
        np.clip(
            np.where(
                df["quality_class"] == "high",
                np.random.normal(8.0, 1.0, n),
                np.where(
                    df["quality_class"] == "mid",
                    np.random.normal(6.0, 1.5, n),
                    np.random.normal(4.5, 1.8, n),
                ),
            ),
            0,
            10,
        ),
        1,
    )

    # Research alignment - quality correlated
    df["research_alignment_score"] = np.round(
        np.clip(
            np.where(
                df["quality_class"] == "high",
                np.random.normal(7.5, 1.2, n),
                np.where(
                    df["quality_class"] == "mid",
                    np.random.normal(5.5, 1.5, n),
                    np.random.normal(3.8, 1.8, n),
                ),
            ),
            0,
            10,
        ),
        1,
    )

    # Publication count - quality correlated
    df["publication_count"] = np.where(
        df["quality_class"] == "high",
        np.random.poisson(1.5, n),
        np.where(
            df["quality_class"] == "mid",
            np.random.poisson(0.5, n),
            np.random.poisson(0.1, n),
        ),
    )

    # Recommendation strength - quality correlated
    df["recommendation_strength"] = np.round(
        np.clip(
            np.where(
                df["quality_class"] == "high",
                np.random.normal(8.5, 0.8, n),
                np.where(
                    df["quality_class"] == "mid",
                    np.random.normal(7.2, 1.0, n),
                    np.random.normal(5.8, 1.2, n),
                ),
            ),
            0,
            10,
        ),
        1,
    )

    # Interview score - quality correlated
    df["interview_score"] = np.round(
        np.clip(
            np.where(
                df["quality_class"] == "high",
                np.random.normal(88, 8, n),
                np.where(
                    df["quality_class"] == "mid",
                    np.random.normal(78, 10, n),
                    np.random.normal(65, 12, n),
                ),
            ),
            0,
            100,
        ),
        1,
    )

    df["english_test_type"] = np.random.choice(english_tests, n, p=[0.4, 0.4, 0.2])
    df["english_score"] = [
        gen_lang_score(t, q)
        for t, q in zip(df["english_test_type"], df["quality_class"])
    ]
    df["chinese_proficiency"] = [
        assign_chinese_proficiency(l, q)
        for l, q in zip(df["degree_language"], df["quality_class"])
    ]

    target = df.apply(assign_targets, axis=1)
    df["admission_decision"] = target.iloc[:, 0]
    df["scholarship_tier"] = target.iloc[:, 1]

    # Drop quality_class as it's only a generation helper, not a real feature
    df = df.drop(columns=["quality_class"])

    file_path = f"./dataset/BIT_Admissions_{time_string}.csv"
    df.to_csv(file_path, index=False)
    return df


if __name__ == "__main__":
    logging.info("Data generation started...")
    df = generate_dataset()
    logging.info("Generation complete. No NaN present.")
