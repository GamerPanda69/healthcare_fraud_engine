import pandas as pd
import numpy as np
import os
import joblib
import torch
from pyod.models.iforest import IForest
from pyod.models.auto_encoder import AutoEncoder
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
DATA_DIR = "anomaly/"
MODEL_DIR = "models/"
MASTER_FILE = "master_trained_data.csv"
MODELS = {
    "sc": os.path.join(MODEL_DIR, "scaler.joblib"),
    "if": os.path.join(MODEL_DIR, "iforest.joblib"),
    "ae": os.path.join(MODEL_DIR, "autoenc.joblib")
}
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    print("🏗️  Synthesizing Relational Data...")
    ts = "1542865627584"
    prefix = "Train"
    
    f_in = f"{DATA_DIR}{prefix}_Inpatientdata-{ts}.csv"
    f_out = f"{DATA_DIR}{prefix}_Outpatientdata-{ts}.csv"
    f_bene = f"{DATA_DIR}{prefix}_Beneficiarydata-{ts}.csv"
    f_labels = f"{DATA_DIR}{prefix}-{ts}.csv"
    
    df_in = pd.read_csv(f_in).assign(Src='In')
    df_out = pd.read_csv(f_out).assign(Src='Out')
    df_bene = pd.read_csv(f_bene)
    df_labels = pd.read_csv(f_labels)
    
    df = pd.concat([df_in, df_out], ignore_index=True)
    df = df.merge(df_bene, on='BeneID', how='left').merge(df_labels, on='Provider', how='left')
    
    df['StayDuration'] = (pd.to_datetime(df['ClaimEndDt']) - pd.to_datetime(df['ClaimStartDt'])).dt.days + 1
    df['Age'] = pd.to_datetime(df['ClaimStartDt']).dt.year - pd.to_datetime(df['DOB']).dt.year
    df['ProcCount'] = df[[f'ClmProcedureCode_{i}' for i in range(1, 7)]].notnull().sum(axis=1)
    
    return df[['InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'StayDuration', 'Age', 'ProcCount']]

def train():
   # Check if models already exist
    if all(os.path.exists(v) for v in MODELS.values()):
        print("🛡️  Assets already exist. Skipping training to preserve the current brain.")
        return # This exits the function immediately
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Hardware Detection: {device.upper()}")
    
    df_features = load_data()
    X = df_features.fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. Isolation Forest (Tree-based Layer)
    print("🌳 Training Isolation Forest...")
    iforest = IForest(contamination=0.1, n_estimators=100, random_state=42).fit(X_scaled)
    
    # 2. Deep Autoencoder (Neural Layer)
    # Using the EXACT parameter names from your help command: 
    # 'hidden_neuron_list' and 'epoch_num'
    print(f"🚀 Training Deep Autoencoder on {device.upper()}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  System Check: {'NVIDIA GPU Detected' if device == 'cuda' else 'No GPU - Using CPU'}")

    autoenc = AutoEncoder(
    hidden_neuron_list=[5, 3, 2, 3, 5],
    epoch_num=50,
    batch_size=1024 if device == 'cuda' else 128, # Drop batch size for CPU to avoid lag
    contamination=0.1,
    device=device, # <--- THIS IS THE KEY
    preprocessing=False,
    verbose=1
)
    autoenc.fit(X_scaled)
    
    # Unified Saving
    joblib.dump(scaler, f"{MODEL_DIR}scaler.joblib")
    joblib.dump(iforest, f"{MODEL_DIR}iforest.joblib")
    joblib.dump(autoenc, f"{MODEL_DIR}autoenc.joblib")
    print("✅ Dual-Brain Ensemble successfully saved to /models.")

if __name__ == "__main__":
    train()