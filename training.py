import os
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# --------------------------
# PATHS
# --------------------------
DATA_CSV = r"C:/Users/sec23/Desktop/Aerosense/data/bird_data.csv"
MODEL_PATH = r"C:/Users/sec23/Desktop/Aerosense/data/bird_health_model.pkl"
ENCODER_PATH = r"C:/Users/sec23/Desktop/Aerosense/data/region_encoder.pkl"

# --------------------------
# LOAD DATA
# --------------------------
if not os.path.exists(DATA_CSV):
    raise FileNotFoundError("bird_data.csv not found!")

df = pd.read_csv(DATA_CSV)

# --------------------------
# ENCODE REGION
# --------------------------
le = LabelEncoder()
df["region_encoded"] = le.fit_transform(df["region_state"])

# Save encoder
joblib.dump(le, ENCODER_PATH)

# --------------------------
# BUILD TRAINING DATA
# --------------------------
rows = []

for _, row in df.iterrows():
    # Healthy sample
    rows.append([
        row["healthy_wingbeat_freq_hz"],
        row["region_encoded"],
        "Healthy"
    ])

    # Unhealthy sample
    rows.append([
        row["unhealthy_wingbeat_freq_hz"],
        row["region_encoded"],
        "Unhealthy"
    ])

train_df = pd.DataFrame(rows, columns=["wingbeat", "region_encoded", "label"])

X = train_df[["wingbeat", "region_encoded"]]
y = train_df["label"]

# --------------------------
# TRAIN MODEL
# --------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, MODEL_PATH)

print("Model trained successfully.")
