#main.py
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import joblib, numpy as np, os, torch

app = FastAPI(title="FraudGuard Hybrid Engine")

# --- GLOBAL CONFIGURATION ---
MODELS = {
    "sc": "models/scaler.joblib", 
    "if": "models/iforest.joblib", 
    "ae": "models/autoenc.joblib",
    "norms": "models/clinical_norms.joblib" # Added the dynamic library
}
BRAIN = {}

@app.on_event("startup")
def load_models():
    """Remaps GPU weights to CPU if needed and loads clinical truths."""
    inference_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        BRAIN["sc"] = joblib.load(MODELS["sc"])
        BRAIN["if"] = joblib.load(MODELS["if"])
        BRAIN["norms"] = joblib.load(MODELS["norms"]) # Load the 95th percentile database
        
        # Load Autoencoder with Hardware Awareness
        BRAIN["ae"] = joblib.load(MODELS["ae"])
        BRAIN["ae"].device = inference_device
        BRAIN["ae"].model.to(inference_device)
        
        print(f"🧠 Engine Online: Running on {inference_device.upper()}")
        print(f"📊 Clinical Norms: {len(BRAIN['norms'])} diagnoses mapped.")
    except Exception as e:
        print(f"⚠️ Load Warning: {e}")

class Claim(BaseModel):
    diag_code: str; amt: float; stay: int; age: int

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def ui(): return FileResponse('static/index.html')

@app.post("/predict")
async def analyze(c: Claim):
    try:
        # 1. AI Score Logic (Numerical Patterns)
        # Order: Amt, Deductible, Stay, Age, ProcCount (using 1 as default ProcCount)
        vec = BRAIN["sc"].transform(np.array([[c.amt, 1068, c.stay, c.age, 1]]))
        s_if = float(BRAIN["if"].decision_function(vec)[0])
        s_ae = float(BRAIN["ae"].decision_function(vec)[0])
        
        # 2. Dynamic Clinical Logic (Domain Rules)
        clin_f = False
        diag_label = "Unknown Diagnosis"
        
        # Instead of a hardcoded dict, we check our learned clinical norms
        if c.diag_code in BRAIN["norms"]:
            truth = BRAIN["norms"][c.diag_code]
            # Flag if the claim exceeds the 95th percentile of historical data
            # We use a 1.2x buffer to allow for slight natural variance
            if c.amt > (truth['InscClaimAmtReimbursed'] * 1.2) or c.stay > (truth['StayDuration'] * 1.2):
                clin_f = True
            diag_label = f"ICD9: {c.diag_code}"
        
        # 3. Decision Boundary Calibration
        IF_THRESHOLD = 0.15
        AE_THRESHOLD = 5.0  # Set based on your 50-epoch baseline

        is_if_anomaly = s_if > IF_THRESHOLD
        is_ae_anomaly = s_ae > AE_THRESHOLD

        return {
            "diag_name": diag_label,
            "is_flagged": (clin_f or is_if_anomaly or is_ae_anomaly),
            "if_score": round(s_if, 4),
            "ae_score": round(s_ae, 4),
            "clin_msg": "Inside Clinical Norms" if not clin_f else "Statistical Outlier Alert",
            "detectors": {
                "clinical": clin_f,
                "iforest": is_if_anomaly,
                "autoencoder": is_ae_anomaly
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})