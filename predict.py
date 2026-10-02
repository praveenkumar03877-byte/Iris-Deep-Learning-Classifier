from pathlib import Path

import joblib
import numpy as np
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "iris-model"

model = load_model(MODEL_DIR / "my_model.keras")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
encoder = joblib.load(MODEL_DIR / "Label_Encoder.pkl")


def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    input_data = np.array(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        dtype=float,
    )

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled, verbose=0)

    class_index = int(np.argmax(prediction, axis=1)[0])
    species = str(encoder.inverse_transform([class_index])[0])
    confidence = float(np.max(prediction) * 100)

    return species, confidence


if __name__ == "__main__":
    species, confidence = predict_iris(5.1, 3.5, 1.4, 0.2)
    print("Predicted Species:", species)
    print("Confidence:", round(confidence, 2), "%")
