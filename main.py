from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from fastapi.staticfiles import StaticFiles # Add this
from fastapi.responses import FileResponse
import numpy as np
import os

app = FastAPI(title="Healthcare Fraud Protection API")

# --- ASSET LOADING ---
MODEL_PATH = "models/iforest_model.joblib"
SCALER_PATH = "models/scaler.joblib"

# Dictionary for Clinical Truth
MEDICAL_KNOWLEDGE = {
    "460": {"name": "Common Cold", "max_stay": 1, "avg_cost": 200},
    "410": {"name": "Heart Attack", "max_stay": 10, "avg_cost": 15000},
    "486": {"name": "Pneumonia", "max_stay": 7, "avg_cost": 5000}
}

# --- SCHEMAS ---
class ClaimInput(BaseModel):
    diag_code: str
    amt: float
    stay: int
    age: int
    deductible: float = 1068.0
    procedure_count: int = 1
app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')
# --- API ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "Operational", "engine": "FastAPI", "model": "Isolation Forest"}

@app.post("/predict")
async def analyze_claim(claim: ClaimInput):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=500, detail="AI Model not found. Run app.py.")

    # 1. Load Model & Scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # 2. Layer 1: Clinical Rule Validation
    is_clinical_fraud = False
    clinical_msg = "Medically plausible stay and cost."
    if claim.diag_code in MEDICAL_KNOWLEDGE:
        info = MEDICAL_KNOWLEDGE[claim.diag_code]
        if claim.stay > (info['max_stay'] * 3) or claim.amt > (info['avg_cost'] * 5):
            is_clinical_fraud = True
            clinical_msg = f"ALERT: {info['name']} intensity exceeds clinical norms."

    # 3. Layer 2: Statistical AI Validation
    input_vector = np.array([[claim.amt, claim.deductible, claim.stay, claim.age, claim.procedure_count]])
    scaled_vector = scaler.transform(input_vector)
    
    score = float(model.decision_function(scaled_vector)[0])
    is_anomaly = bool(model.predict(scaled_vector)[0])
    is_anomaly = score > 0.15
    return {
        "diagnosis": MEDICAL_KNOWLEDGE.get(claim.diag_code, {}).get("name", "Other"),
        "clinical_check": {"fraud_detected": is_clinical_fraud, "message": clinical_msg},
        "statistical_check": {"is_anomaly": is_anomaly, "anomaly_score": round(score, 4)},
        "final_decision": "FLAGGED" if (is_clinical_fraud or is_anomaly) else "APPROVED"
    }