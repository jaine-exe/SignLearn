import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================
# PATHS
# =========================
DATASET_PATH = "dataset/isl_landmarks.csv"
MODEL_PATH = "model/gesture_model.pkl"

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv(DATASET_PATH)

# =========================
# CLEAN DATA
# =========================
if "uses_two_hands" in df.columns:
    df = df.drop(columns=["uses_two_hands"])

# Keep only first 3 targets (for A/B/C project)
unique_targets = sorted(df["target"].unique())
allowed_targets = unique_targets[:3]
df = df[df["target"].isin(allowed_targets)]

# =========================
# FEATURES / LABELS
# =========================
X = df.drop(columns=["target"])
y = df["target"]

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# TRAIN MODEL
# =========================
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# =========================
# EVALUATE
# =========================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"✅ Model Accuracy: {acc * 100:.2f}%")

# =========================
# SAVE MODEL
# =========================
os.makedirs("model", exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"✅ Model saved at: {MODEL_PATH}")
print(f"✅ Number of features used: {X.shape[1]}")
print(f"✅ Labels used: {sorted(y.unique())}")