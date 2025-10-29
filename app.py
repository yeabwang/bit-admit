from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from BIT_ADMIT_AI.components.data_transformation import DataTransformation
from BIT_ADMIT_AI.pipeline.prediction import BitAdmitClassifier, BitAdmitFeatures

app = FastAPI(title="BIT Admit AI")

static_dir = Path(__file__).parent / "static"
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

classifier = BitAdmitClassifier()


def _compute_language_pass_and_weight(features: BitAdmitFeatures) -> Dict[str, float]:
    row = pd.Series(features.to_dict())

    chinese_level = row.get("chinese_proficiency", 0)
    if isinstance(chinese_level, str):
        digits = "".join(ch for ch in chinese_level if ch.isdigit())
        chinese_level = float(digits) if digits else 0.0
        row["chinese_proficiency"] = chinese_level

    language_pass = float(DataTransformation._language_requirement_passed(row))  # type: ignore[attr-defined]
    weighted_score = float(DataTransformation._weighted_score(row))  # type: ignore[attr-defined]
    return {"language_pass": language_pass, "weighted_score": weighted_score}


def _calculate_radar_data(features: BitAdmitFeatures) -> list[float]:
    metrics = _compute_language_pass_and_weight(features)
    data_points = [
        min(float(features.previous_gpa) / 4.0, 1.0),
        min(float(features.math_physics_background_score) / 10.0, 1.0),
        min(float(features.research_alignment_score) / 10.0, 1.0),
        min(float(features.publication_count) / 5.0, 1.0),
        min(float(features.recommendation_strength) / 10.0, 1.0),
        min(float(features.interview_score) / 100.0, 1.0),
        metrics["language_pass"],
    ]
    return data_points


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "admission.html",
        {
            "request": request,
            "predictions": None,
            "input_data": {},
            "radar_data": None,
            "prediction_timestamp": "Awaiting input...",
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(  # pylint: disable=too-many-arguments
    request: Request,
    program_category: str = Form(...),
    country: str = Form(...),
    bit_program_applied: str = Form(...),
    degree_language: str = Form(...),
    previous_gpa: float = Form(...),
    math_physics_background_score: float = Form(...),
    research_alignment_score: float = Form(...),
    publication_count: float = Form(...),
    recommendation_strength: float = Form(...),
    interview_score: float = Form(...),
    english_test_type: str = Form(""),
    english_score: Optional[str] = Form(None),
    chinese_proficiency: str = Form(""),
) -> HTMLResponse:
    english_score_value = (
        float(english_score.strip()) if english_score and english_score.strip() else 0.0
    )

    features = BitAdmitFeatures(
        program_category=program_category,
        country=country,
        bit_program_applied=bit_program_applied,
        degree_language=degree_language,
        previous_gpa=previous_gpa,
        math_physics_background_score=math_physics_background_score,
        research_alignment_score=research_alignment_score,
        publication_count=publication_count,
        recommendation_strength=recommendation_strength,
        interview_score=interview_score,
        english_test_type=english_test_type,
        english_score=english_score_value,
        chinese_proficiency=chinese_proficiency,
    )

    predictions = classifier.predict(features)
    radar_data = _calculate_radar_data(features)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    input_snapshot = features.to_dict()
    if input_snapshot["degree_language"] != "english_taught":
        input_snapshot["english_score"] = ""
        input_snapshot["english_test_type"] = ""

    return templates.TemplateResponse(
        "admission.html",
        {
            "request": request,
            "predictions": predictions,
            "input_data": input_snapshot,
            "radar_data": radar_data,
            "prediction_timestamp": timestamp,
        },
    )


@app.post("/predict-json", response_class=JSONResponse)
async def predict_json(payload: Dict[str, Any]) -> JSONResponse:
    features = BitAdmitFeatures(**payload)
    predictions = classifier.predict(features)
    radar_data = _calculate_radar_data(features)
    timestamp = datetime.utcnow().isoformat()
    return JSONResponse(
        {
            "predictions": predictions,
            "radar_data": radar_data,
            "timestamp": timestamp,
        }
    )
