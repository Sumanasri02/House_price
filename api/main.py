from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

# -------------------------
# File paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models/best_model.pkl"

# -------------------------
# Load model
# -------------------------
print(f"Looking for model at: {MODEL_PATH}")
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading model: {e}")
    raise SystemExit("Exiting: Model file missing")

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="House Price Prediction API")

# -------------------------
# Input schema: main features only
# -------------------------
class HouseInput(BaseModel):
    area: int
    bedrooms: int
    bathrooms: int
    stories: int
    parking: int
    mainroad: str           # Yes / No
    furnishingstatus: str   # Furnished / Semi-furnished / Unfurnished

# -------------------------
# Helper: Process input like training
# -------------------------
def preprocess_input(input_df: pd.DataFrame):
    # 1. One-hot encode categorical columns
    input_df = pd.get_dummies(input_df, drop_first=True)
    
    # 2. Ensure all model features exist
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # 3. Reorder columns exactly as model
    input_df = input_df[model_features]
    
    return input_df

# -------------------------
# Routes
# -------------------------
@app.get("/")
def home():
    return {"message": "House Price Prediction API"}

@app.post("/predict")
def predict_price(data: HouseInput):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Preprocess
    processed_df = preprocess_input(input_df)
    
    # Predict
    prediction = model.predict(processed_df)
    
    return {"predicted_price": float(prediction[0] * 1000)}
