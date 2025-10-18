import pandas as pd
import numpy as np
import random
from datetime import datetime
from time import strftime

np.random.seed(42)
n = 2000

current_time = datetime.now()
time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")

# Application categories and programs
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
    "Armament Science and Technology",
    "Aviation Digital Economy and Management",
    "Biology",
    "Biomedical Engineering",
    "Business Administration",
    "Chemical Engineering and Technology",
    "Chemistry",
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
    "Master of Business Administration (MBA)",
    "Mechanical Engineering",
    "Mechanics",
    "Optical Engineering",
    "Physics",
    "Power Engineering and Engineering Thermophysics",
    "Safety Science and Technology",
    "Statistics",
    "Teaching Chinese to Speakers of Other Languages",
    "Theoretical Economics",
    "Cyberspace Science and Technology",
    "Integrated Circuit Science and Technology",
    "Master of Engineering Management",
    "National Economy Mobilization",
]
chinese_language_programs = ["Chinese Language and Culture"]
dual_degree_programs = [
    "Joint Mechanical Engineering",
    "Joint Computer Science Program",
    "Joint Business Administration Program",
]

# Demographic
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

# Helper functions
def select_program(category):
    if category == "Undergraduate":
        return random.choice(undergraduate_programs)
    elif category == "Postgraduate":
        return random.choice(postgraduate_programs)
    elif category == "Chinese Language":
        return random.choice(chinese_language_programs)
    else:
        return random.choice(dual_degree_programs)


# Chinese-taught logic: require HSK5+, else NO_EXAM or low HSK
def assign_chinese_proficiency(lang):
    if lang == "Chinese-taught":
        return np.random.choice(["HSK5", "HSK6"], p=[0.7, 0.3])
    else:
        return np.random.choice(
            ["NO_EXAM", "HSK1", "HSK2", "HSK3"], p=[0.7, 0.1, 0.1, 0.1]
        )


# Degree language logic
def language_logic(cat):
    if cat in ["Undergraduate", "Postgraduate", "Dual Degree"]:
        return np.random.choice(["English-taught", "Chinese-taught"], p=[0.7, 0.3])
    else:
        return "Chinese-taught"


# Target assignment logic simplified for demo
def assign_targets(row):
    # Auto-reject conditions
    if (
        row.document_authenticity_score < 0.7
        or (row.degree_language == "English-taught" and row.english_score < 60)
        or (
            row.degree_language == "Chinese-taught"
            and row.chinese_proficiency not in ["HSK5", "HSK6"]
        )
        or row.previous_visa_refusals >= 2
        or row.application_completeness_score < 0.85
    ):
        return pd.Series(["Rejected", "None", 0])

    # Scholarship logic
    if (
        row.belt_road_country == 1
        and row.previous_gpa >= 3.5
        and row.research_alignment_score >= 8
    ):
        return pd.Series(["Admitted", "Full", random.randint(45000, 60000)])
    if (
        row.technical_competitions >= 2
        and row.interview_score >= 90
        and row.family_income_tier == "Low"
    ):
        return pd.Series(["Admitted", "Full", random.randint(35000, 50000)])
    if row.previous_gpa >= 3.2 and row.developing_country == 1 and row.sop_score >= 7:
        return pd.Series(["Admitted", "Partial", random.randint(15000, 30000)])
    # Default admit without scholarship
    return pd.Series(["Admitted", "NO_SCHOLARSHIP", 0])


# Initialize dataframe
df = pd.DataFrame(
    {
        "application_id": [f"BIT2025{str(i).zfill(4)}" for i in range(1, n + 1)],
        "program_category": np.random.choice(
            program_categories, n, p=[0.4, 0.35, 0.15, 0.1]
        ),
    }
)

df["bit_program_applied"] = df["program_category"].apply(select_program)
df["degree_language"] = df["program_category"].apply(language_logic)

# Academic / profile numeric features
df["previous_gpa"] = np.round(np.random.normal(3.3, 0.4, n).clip(0, 4), 2)
df["academic_ranking_percentile"] = np.random.randint(1, 101, n)
df["math_physics_background_score"] = np.round(np.random.uniform(4, 10, n), 1)
df["research_alignment_score"] = np.round(np.random.uniform(0, 10, n), 1)
df["publication_count"] = np.random.poisson(0.5, n)
df["technical_competitions"] = np.random.poisson(1.5, n)

# Language proficiency
english_tests = ["IELTS", "TOEFL", "DUOLINGO", "NO_EXAM"]
chinese_levels = ["NO_EXAM", "HSK1", "HSK2", "HSK3", "HSK4", "HSK5", "HSK6"]

df["english_test_type"] = np.random.choice(english_tests, n, p=[0.5, 0.3, 0.1, 0.1])
df["english_score"] = np.round(np.random.normal(85, 10, n).clip(40, 100), 1)


df["chinese_proficiency"] = df["degree_language"].apply(assign_chinese_proficiency)
df["chinese_study_duration_months"] = np.random.randint(0, 12, n)
df["language_certificate_authenticity"] = np.round(np.random.uniform(0.7, 1, n), 2)

df["home_country"] = np.random.choice(countries, n)
df["belt_road_country"] = np.random.choice([0, 1], n, p=[0.3, 0.7])
df["developing_country"] = np.random.choice([0, 1], n, p=[0.4, 0.6])
df["previous_china_experience"] = np.random.choice([0, 1], n, p=[0.8, 0.2])
df["bit_partner_university"] = np.random.choice([0, 1], n, p=[0.7, 0.3])
df["age"] = np.random.randint(18, 35, n)
df["gender"] = np.random.choice(["Male", "Female"], n, p=[0.55, 0.45])

# Financial
df["financial_guarantee_available"] = np.random.choice([0, 1], n, p=[0.2, 0.8])
df["family_income_tier"] = np.random.choice(
    ["Low", "Medium", "High"], n, p=[0.4, 0.45, 0.15]
)
df["sponsorship_type"] = np.random.choice(
    ["Government", "Self-funded", "University", "Company"], n, p=[0.3, 0.4, 0.2, 0.1]
)
df["previous_scholarship"] = np.random.choice([0, 1], n, p=[0.75, 0.25])
df["scholarship_essay_score"] = np.round(np.random.uniform(4, 10, n), 1)
df["interview_score"] = np.round(np.random.normal(80, 10, n).clip(0, 100), 1)
df["sop_score"] = np.round(np.random.uniform(5, 10, n), 1)
df["recommendation_strength"] = np.round(np.random.uniform(5, 10, n), 1)
df["cultural_adaptability_score"] = np.round(np.random.uniform(5, 10, n), 1)
df["document_authenticity_score"] = np.round(np.random.uniform(0.7, 1, n), 2)
df["health_certificate_status"] = np.random.choice(
    ["Valid", "Invalid"], n, p=[0.95, 0.05]
)
df["previous_visa_refusals"] = np.random.choice([0, 1, 2], n, p=[0.85, 0.1, 0.05])
df["application_completeness_score"] = np.round(np.random.uniform(0.8, 1, n), 2)

df[["admission_decision", "scholarship_tier", "scholarship_amount_rmb"]] = df.apply(
    assign_targets, axis=1
)

# Save CSV
file_path = f"./dataset/BIT_International_Admissions_Synthetic_{time_string}.csv"
df.to_csv(file_path, index=False)
