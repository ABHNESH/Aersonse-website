from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import os

app = Flask(__name__)
CORS(app)

# ----------------------------
# FILE PATHS
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(BASE_DIR, "users.json")
PRED_FILE = os.path.join(BASE_DIR, "predictions.json")

MODEL_PATH = os.path.join(BASE_DIR, "data/bird_health_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "data/region_encoder.pkl")

# ----------------------------
# LOAD MODEL
# ----------------------------
model = joblib.load(MODEL_PATH)
region_encoder = joblib.load(ENCODER_PATH)

# ----------------------------
# JSON HELPERS
# ----------------------------
def read_json(path):
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump([], f)
    with open(path, "r") as f:
        return json.load(f)

def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

# ----------------------------
# SIGNUP API
# ----------------------------
@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    email = data.get("email")
    role = data.get("role")

    users = read_json(USERS_FILE)

    # Check duplicate
    for u in users:
        if u["email"] == email:
            return jsonify({"status": "error", "message": "User already exists"}), 400

    # Save new user
    users.append(data)
    write_json(USERS_FILE, users)

    return jsonify({"status": "success", "message": "User registered"})

# ----------------------------
# LOGIN API
# ----------------------------
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    role = data.get("role")

    users = read_json(USERS_FILE)

    for u in users:
        if u["email"] == email and u["password"] == password and u["role"] == role:
            return jsonify({"status": "success", "role": role})

    return jsonify({"status": "error", "message": "Invalid credentials"}), 401

# ----------------------------
# PREDICT API
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    bird_name = data.get("bird_name")
    region = data.get("region")
    frequency = data.get("frequency")

    if not bird_name or not region:
        return jsonify({"status": "error", "message": "Missing fields"}), 400

    try:
        region_encoded = region_encoder.transform([region])[0]
        X = [[bird_name, region_encoded, frequency]]

        prediction_proba = model.predict_proba(X)[0]
        prediction_label = model.predict(X)[0]
        confidence = round(max(prediction_proba) * 100, 2)

        # save prediction
        predictions = read_json(PRED_FILE)
        predictions.append({
            "bird_name": bird_name,
            "region": region,
            "frequency": frequency,
            "prediction": prediction_label,
            "confidence": confidence
        })
        write_json(PRED_FILE, predictions)

        return jsonify({
            "status": "success",
            "prediction": prediction_label,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ----------------------------
# RUN SERVER
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
