from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model

# Always build paths relative to this file.
# This prevents "File not found" errors when Uvicorn is started from another folder.
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "iris-model"

MODEL_PATH = MODEL_DIR / "my_model.keras"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
ENCODER_PATH = MODEL_DIR / "Label_Encoder.pkl"

# Check required files before loading them.
for file_path in (MODEL_PATH, SCALER_PATH, ENCODER_PATH):
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

app = FastAPI(
    title="Iris Flower Prediction API",
    description="Deep Learning based Iris flower classification API",
    version="1.0.0",
)

# Allows a separate frontend to call this API during local development/deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IrisData(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, description="Petal width in cm")


@app.get("/")
def home():
    return {
        "message": "Iris Flower Prediction API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: IrisData):
    input_data = np.array(
        [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width,
        ]],
        dtype=float,
    )

    # Use exactly the same scaler that was fitted during training.
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled, verbose=0)
    class_index = int(np.argmax(prediction, axis=1)[0])

    species = str(encoder.inverse_transform([class_index])[0])
    confidence = float(np.max(prediction) * 100)

    return {
        "prediction": species,
        "confidence": round(confidence, 2),
    }
