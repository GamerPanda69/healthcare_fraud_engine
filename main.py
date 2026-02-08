from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import joblib, numpy as np, os, torch  # Added torch for hardware detection

app = FastAPI(title="FraudGuard Hybrid Engine")

# Global containers for the models
MODELS = {"sc": "models/scaler.joblib", "if": "models/iforest.joblib", "ae": "models/autoenc.joblib"}
BRAIN = {}

@app.on_event("startup")
def load_models():
    """ 
    This block executes once when the server starts. 
    It handles the re-mapping of GPU weights to CPU for your partner.
    """
    inference_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Load standard models
        BRAIN["sc"] = joblib.load(MODELS["sc"])
        BRAIN["if"] = joblib.load(MODELS["if"])
        
        # Load the Autoencoder and force it onto the detected hardware
        # This prevents a 'RuntimeError' on non-NVIDIA laptops
        BRAIN["ae"] = joblib.load(MODELS["ae"])
        BRAIN["ae"].device = inference_device
        BRAIN["ae"].model.to(inference_device)
        
        print(f"🧠 Engine Online: Running on {inference_device.upper()}")
    except Exception as e:
        print(f"⚠️ Load Warning: Models not found or incompatible. Error: {e}")

# Medical domain knowledge base
KNOWLEDGE = {
    "460": {"n": "Cold", "s": 1, "c": 200}, 
    "410": {"n": "Heart Attack", "s": 10, "c": 15000}
}

class Claim(BaseModel):
    diag_code: str; amt: float; stay: int; age: int

# Mount UI static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def ui(): return FileResponse('static/index.html')

@app.post("/predict")
async def analyze(c: Claim):
    try:
        # 1. Feature Transformation
        # Order must match training: Amt, Deductible, Stay, Age, ProcCount
        vec = BRAIN["sc"].transform(np.array([[c.amt, 1068, c.stay, c.age, 1]]))
        
        # 2. Statistical Scores (Tree-based & Deep Learning)
        s_if = float(BRAIN["if"].decision_function(vec)[0])
        s_ae = float(BRAIN["ae"].decision_function(vec)[0])
        
        # 3. Dynamic Calibration Thresholds
        # Note: These may need adjustment after your 50-epoch run
        IF_THRESHOLD = 0.15
        AE_THRESHOLD = 5.0 # Lowered for a 50-epoch high-fidelity model

        is_if_anomaly = s_if > IF_THRESHOLD
        is_ae_anomaly = s_ae > AE_THRESHOLD
        
        # 4. Clinical Heuristics
        clin_f = False
        if c.diag_code in KNOWLEDGE:
            k = KNOWLEDGE[c.diag_code]
            if c.stay > k['s']*3 or c.amt > k['c']*5: clin_f = True

        return {
            "diag_name": KNOWLEDGE.get(c.diag_code, {}).get("n", "Other"),
            "is_flagged": (clin_f or is_if_anomaly or is_ae_anomaly),
            "if_score": round(s_if, 4),
            "ae_score": round(s_ae, 4),
            "clin_msg": "Medically Plausible" if not clin_f else "Clinical Intensity Alert",
            "detectors": {
                "clinical": clin_f,
                "iforest": is_if_anomaly,
                "autoencoder": is_ae_anomaly
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})