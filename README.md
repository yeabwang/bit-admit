# BIT Admit AI

**Machine Learning–powered admission and scholarship decision framework for international applicants to Chinese universities.**

---

## 🧩 Overview

Chinise Universities handle thousands of international applications yearly. Manual evaluation is slow, inconsistent, and contain human bias.  **BIT-ADMIT** propose a machine learning pipeline to automate admission and scholarship decisions for Chinise Universities while keeping fairness and transparency measurable.  

We generated 2,000 synthetic student profiles modeled on Beijing Institute of Technology’s application schema, featuring:  
- Academic data (GPA, subject scores)  
- Research and recommendation strength  
- Interview and language proficiency metrics  

No PII included. Data validated via entropy, drift, and bias checks.  
Six classifiers were trained — Random Forest, XGBoost, CatBoost, Gradient Boosting, Logistic Regression, and SVC.  
**XGBoost** and **Gradient Boosting** achieved the best performance:  
- Admission F1: **0.998**  
- Scholarship F1: **0.985**

Deployed as a FastAPI service for real-time inference.  
Although trained on synthetic data, distribution alignment tests indicate good generalization potential.  
**Future work:** validation on real anonymized admissions data + fairness metrics integration.  

[Read the paper](https://github.com/yeabwang/bit-admit/blob/main/documents/BIT_ADMIT__A_Machine_Learning_Framework_for_Automating_International_Student_Admission_and_Scholarship_Decisions.pdf)
---

## 🧠 Core Features

- Synthetic dataset generator + MongoDB ingestion  
- Full ML pipeline: ingestion → validation → transformation → training → evaluation → model push  
- Schema validation and drift detection via **Evidently**  
- Auto feature engineering with `ColumnTransformer`  
- Independent admission/scholarship classifiers with grid search  
- F1/Precision/Recall evaluation and model promotion tracking  
- FastAPI UI + JSON endpoint for real-time predictions  

---
## 🛠️ Setup  

```bash
conda create -n myenv python=3.12 -y
conda activate myenv
pip install -r requirements.txt
pip install -e .
```

### .env configuration(Optional)
```bash
cat > .env <<'EOF'
DATABASE_NAME="your_db"
COLLECTION_NAME="your_collection"
MONGODB_URL_KEY="your_connection_url"
EOF
```

### Run the ui
```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
# or with Docker
docker build -t bit-admit-ai .
docker run -p 8000:8000 bit-admit-ai
```
---

## Data Options  

**Option 1: Generate Synthetic Data**  
```bash
python -c 'from BIT_ADMIT_AI.utils.main_utils import generate_dataset; df=generate_dataset(); print(df.shape)'
```

**Option 2: Ingest from MongoDB**  
```bash
python scripts/run_ingestion.py
```

**Option 3: Local CSV fallback (default if no .env)**

If `.env` is not provided or MongoDB is unreachable, the pipeline will automatically load the most recent CSV from `original_dataset/` and proceed:

```bash
python -c "from BIT_ADMIT_AI.components.data_ingestion import DataIngestion; DataIngestion().init_data_ingestion()"
```
Ensure at least one CSV exists in `original_dataset/`.

## 🏋️ Train the Model

Full training pipeline:
```bash
python demo.py
```

**Outputs:**  
- Trained model: `bit_artifact/<timestamp>/model_trainer/trained_model/model.pkl`  
- Best model + metrics: `best_model/model.pkl`, `best_model/metrics.yaml`

---

## 🌐 Serve via FastAPI

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Endpoints:  
- `GET /` — web form  
- `POST /predict` — form submission  
- `POST /predict-json` — API  

**Example JSON:**
```bash
curl -s -X POST http://localhost:8000/predict-json   -H "Content-Type: application/json"   -d '{
    "program_category": "Undergraduate",
    "country": "India",
    "bit_program_applied": "Computer Science",
    "degree_language": "english_taught",
    "previous_gpa": 3.5,
    "math_physics_background_score": 8,
    "research_alignment_score": 6,
    "publication_count": 1,
    "recommendation_strength": 8,
    "interview_score": 85,
    "english_test_type": "IELTS",
    "english_score": 7.0,
    "chinese_proficiency": "HSK3"
  }' | jq .
```

---

## 🧑‍💻 Development

```bash
black .
ruff check --fix .
```

---

Built by 夏雨
