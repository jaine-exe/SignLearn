import joblib
import numpy as np
import cv2
import mediapipe as mp

MODEL_PATH = "model/gesture_model.pkl"

# Map model labels to letters
LABEL_MAP = {
    0: "A",
    1: "B",
    2: "C"
}

model = joblib.load(MODEL_PATH)

mp_hands = mp.solutions.hands


def extract_landmarks_from_image(image):
    """
    Extract 126 features:
    - 63 for left hand
    - 63 for right hand
    If only one hand is detected, other hand is filled with zeros.
    """

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    ) as hands:
        results = hands.process(image_rgb)

        left_hand = np.zeros(63)
        right_hand = np.zeros(63)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])

                label = handedness.classification[0].label  # Left or Right

                if label == "Left":
                    left_hand = np.array(coords)
                else:
                    right_hand = np.array(coords)

        features = np.concatenate([left_hand, right_hand])
        return features, results

def predict_from_image(image):
    """
    Takes BGR image and returns prediction + confidence
    """
    features, results = extract_landmarks_from_image(image)

    if np.all(features == 0):
        return None, 0.0, features, results

    features = features.reshape(1, -1)
    raw_prediction = model.predict(features)[0]

    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba(features)) * 100)
    else:
        confidence = 80.0

    prediction = LABEL_MAP.get(int(raw_prediction), str(raw_prediction))

    return prediction, round(confidence, 2), features.flatten(), results