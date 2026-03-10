import os
import sys
import joblib
import pandas as pd

# Add scripts folder to path for imports (fixes ModuleNotFoundError)
sys.path.append(os.path.dirname(__file__))

# Import sibling scripts
import training
import predict
import generate

# Paths for data, model, and encoder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "bird_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "data", "bird_health_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "data", "region_encoder.pkl")


# 1️⃣ Load trained model
def load_model():
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        return model
    else:
        raise FileNotFoundError("Model not found. Train the model first!")


# 2️⃣ Predict bird health
def predict_bird_health(bird_features: dict):
    """
    bird_features: dict of feature names and values
    Example: {'region': 'north', 'weight': 2.5, 'age': 1.2}
    """
    model = load_model()
    df = pd.DataFrame([bird_features])
    result = predict.predict_health(df, model, ENCODER_PATH)
    return result


# 3️⃣ Add a new bird to CSV
def add_bird(bird_features: dict):
    """
    Adds bird data to CSV
    """
    df = pd.DataFrame([bird_features])
    if os.path.exists(DATA_PATH):
        df.to_csv(DATA_PATH, mode='a', header=False, index=False)
    else:
        df.to_csv(DATA_PATH, index=False)
    return "Bird added successfully!"


# 4️⃣ Train/retrain the model
def train_model():
    """
    Train model from CSV and save model & encoder
    """
    if os.path.exists(DATA_PATH):
        training.train_model(DATA_PATH, MODEL_PATH, ENCODER_PATH)
        return "Model trained successfully!"
    else:
        raise FileNotFoundError("CSV data not found. Generate data first!")


# 5️⃣ Get all birds from CSV
def get_all_birds():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        return df.to_dict(orient="records")
    else:
        return []
