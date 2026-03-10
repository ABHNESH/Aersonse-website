import os
import pandas as pd
import joblib
import tempfile
import shutil
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np

DATA_CSV = r"C:/Users/sec23/Desktop/Aerosense/data/bird_data.csv"
MODEL_PATH = r"C:/Users/sec23/Desktop/Aerosense/data/bird_health_model.pkl"
ENCODER_PATH = r"C:/Users/sec23/Desktop/Aerosense/data/region_encoder.pkl"

os.makedirs(os.path.dirname(DATA_CSV), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def safe_to_csv(df, path):
    try:
        dirname = os.path.dirname(path) or "."
        with tempfile.NamedTemporaryFile(delete=False, mode="w", newline="", dir=dirname) as tmp:
            df.to_csv(tmp.name, index=False)
            tmpname = tmp.name
        shutil.move(tmpname, path)
    except Exception as e:
        print(f"Error saving CSV: {e}")
        raise

if not os.path.exists(DATA_CSV):
    raise FileNotFoundError("CSV file not found. Please create bird_data.csv first.")

df = pd.read_csv(DATA_CSV)

# Add helper lowercase columns
df["bird_name_lower"] = df["bird_name"].astype(str).str.strip().str.lower()
df["region_state_lower"] = df["region_state"].astype(str).str.strip().str.lower()

encoder = LabelEncoder()
df["region_encoded"] = encoder.fit_transform(df["region_state_lower"])
joblib.dump(encoder, ENCODER_PATH)


def train_rf_model(dataframe):
    rows = []
    for _, r in dataframe.iterrows():
        rows.append({
            "wingbeat": r["healthy_wingbeat_freq_hz"],
            "region_encoded": r["region_encoded"],
            "label": "Healthy"
        })
        rows.append({
            "wingbeat": r["unhealthy_wingbeat_freq_hz"],
            "region_encoded": r["region_encoded"],
            "label": "Unhealthy"
        })

    tdf = pd.DataFrame(rows)
    X = tdf[["wingbeat", "region_encoded"]]
    y = tdf["label"]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model


rf_model = train_rf_model(df)   # silent, no print

def distance_predict(obs, healthy, unhealthy):
    d_h = abs(obs - healthy)
    d_u = abs(obs - unhealthy)

    denom = d_h + d_u
    if denom == 0:
        return "Healthy", 0.5

    if d_h < d_u:
        conf = 1 - d_h / denom
        return "Healthy", round(conf, 3)
    else:
        conf = 1 - d_u / denom
        return "Unhealthy", round(conf, 3)

def predict_for_bird(bird_name, region_state, obs):
    b = bird_name.strip().lower()
    r = region_state.strip().lower()

    matches = df[df["bird_name_lower"] == b]

    if matches.empty:
        return None  

    same_region = matches[matches["region_state_lower"] == r]
    row = same_region.iloc[0] if not same_region.empty else matches.iloc[0]

    healthy = float(row["healthy_wingbeat_freq_hz"])
    unhealthy = float(row["unhealthy_wingbeat_freq_hz"])

    return distance_predict(obs, healthy, unhealthy)


def add_new_bird(bird_name, region_state):
    global df, rf_model, encoder

    print(f"\nAdding '{bird_name}' — please provide details.")

    wing_len = float(input("Wing length (cm): "))
    heartbeat_mv = float(input("Heartbeat micro vibrations: "))
    respiration_bpm = float(input("Respiration rate (bpm): "))
    healthy_w = float(input("Healthy wingbeat freq (Hz): "))
    unhealthy_w = float(input("Unhealthy wingbeat freq (Hz): "))

    new_row = {
        "bird_name": bird_name,
        "region_state": region_state,
        "wing_length_cm": wing_len,
        "heartbeat_micro_vibrations": heartbeat_mv,
        "respiration_bpm": respiration_bpm,
        "healthy_wingbeat_freq_hz": healthy_w,
        "unhealthy_wingbeat_freq_hz": unhealthy_w,
    }

  
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df["bird_name_lower"] = df["bird_name"].astype(str).str.strip().str.lower()
    df["region_state_lower"] = df["region_state"].astype(str).str.strip().str.lower()

    encoder = LabelEncoder()
    df["region_encoded"] = encoder.fit_transform(df["region_state_lower"])
    joblib.dump(encoder, ENCODER_PATH)

    safe_to_csv(
        df.drop(columns=["bird_name_lower", "region_state_lower"]),
        DATA_CSV
    )

    rf_model = train_rf_model(df)

    print("Bird added and model updated.\n")

if __name__ == "__main__":
    print("===Bird Health Predictor===")
   

    while True:
        bird = input("Enter bird name: ").strip()
        if bird.lower() == "exit":
            print("Exiting program.")
            break

        region = input("Enter region/state: ").strip()
        if region.lower() == "exit":
            print("Exiting program.")
            break

        obs_in = input("Enter observed wingbeat frequency (Hz): ").strip()
        if obs_in.lower() == "exit":
            print("Exiting program.")
            break

        try:
            obs = float(obs_in)
        except:
            print("Invalid frequency. Try again.\n")
            continue

        result = predict_for_bird(bird, region, obs)

        if result is None:
            print(f"\nBird '{bird}' not found in dataset.")
            ch = input("Do you want to add it? (y/n): ").strip().lower()

            if ch == "y":
                add_new_bird(bird, region)
                result = predict_for_bird(bird, region, obs)
            else:
                print("Skipping prediction.\n")
                continue

        label, confidence = result
        print("\n====== RESULT ======")
        print(f"Health Status : {label}")
        print(f"Confidence    : {confidence:.2f}")
        print("====================\n")
