import pandas as pd
import numpy as np
import os
import joblib
from pyod.models.iforest import IForest
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
DATA_DIR = "anomaly/"
MODEL_DIR = "models/"
MASTER_FILE = "master_trained_data.csv"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def load_and_combine():
    print("🏗️  Step 1: Relational Data Synthesis...")
    ts = "1542865627584"
    prefix = "Train"
    
    # Check if files exist
    required_files = [
        f"{DATA_DIR}{prefix}_Inpatientdata-{ts}.csv",
        f"{DATA_DIR}{prefix}_Outpatientdata-{ts}.csv",
        f"{DATA_DIR}{prefix}_Beneficiarydata-{ts}.csv",
        f"{DATA_DIR}{prefix}-{ts}.csv"
    ]
    
    for f in required_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing file: {f}. Please place it in /anomaly")

    df_in = pd.read_csv(required_files[0])
    df_out = pd.read_csv(required_files[1])
    df_bene = pd.read_csv(required_files[2])
    df_labels = pd.read_csv(required_files[3])
    
    df_in['Source'] = 'Inpatient'
    df_out['Source'] = 'Outpatient'
    df = pd.concat([df_in, df_out], axis=0, ignore_index=True)
    df = pd.merge(df, df_bene, on='BeneID', how='left')
    df = pd.merge(df, df_labels, on='Provider', how='left')
    
    # Feature Engineering
    df['ClaimStartDt'] = pd.to_datetime(df['ClaimStartDt'])
    df['ClaimEndDt'] = pd.to_datetime(df['ClaimEndDt'])
    df['DOB'] = pd.to_datetime(df['DOB'])
    df['StayDuration'] = (df['ClaimEndDt'] - df['ClaimStartDt']).dt.days + 1
    df['Age'] = (df['ClaimStartDt'].dt.year - df['DOB'].dt.year)
    
    proc_cols = [f'ClmProcedureCode_{i}' for i in range(1, 7)]
    df['ProcedureCount'] = df[proc_cols].notnull().sum(axis=1)
    
    return df

def train_and_save():
    if os.path.exists(MASTER_FILE):
        print(f"📦 Loading cached data from {MASTER_FILE}...")
        df = pd.read_csv(MASTER_FILE, low_memory=False)
    else:
        df = load_and_combine()
        df.to_csv(MASTER_FILE, index=False)

    print("🧠 Step 2: Training Multi-Dimensional Isolation Forest...")
    features = ['InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'StayDuration', 'Age', 'ProcedureCount']
    X = df[features].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = IForest(contamination=0.1, n_estimators=100, random_state=42)
    clf.fit(X_scaled)
    
    # Save assets
    joblib.dump(clf, f"{MODEL_DIR}iforest_model.joblib")
    joblib.dump(scaler, f"{MODEL_DIR}scaler.joblib")
    print(f"✅ Assets saved in {MODEL_DIR}")

if __name__ == "__main__":
    train_and_save()