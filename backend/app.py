from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import mediapipe as mp
from tensorflow import keras
import pandas as pd
import copy
import itertools
import string

app = Flask(__name__)
CORS(app)

# 🔥 Load model
model = keras.models.load_model("Indian-Sign-Language-Detection/model.h5")

# 🔤 Labels
alphabet = ['1','2','3','4','5','6','7','8','9']
alphabet += list(string.ascii_uppercase)

mp_hands = mp.solutions.hands

# --------------------------
# FUNCTIONS
# --------------------------

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for landmark in landmarks.landmark:
        x = min(int(landmark.x * image_width), image_width - 1)
        y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([x, y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp = copy.deepcopy(landmark_list)

    base_x, base_y = temp[0][0], temp[0][1]

    for i in range(len(temp)):
        temp[i][0] -= base_x
        temp[i][1] -= base_y

    temp = list(itertools.chain.from_iterable(temp))

    max_value = max(list(map(abs, temp)))
    if max_value == 0:
        return temp

    return [n / max_value for n in temp]


# --------------------------
# 🔥 FEEDBACK SYSTEM
# --------------------------

THRESHOLD = 15

def generate_feedback(expected, landmarks):
    feedback = []

    tips = {
        "Index": 8,
        "Middle": 12,
        "Ring": 16,
        "Little": 20
    }

    def is_straight(tip):
        return landmarks[tip][1] < landmarks[tip - 2][1] - THRESHOLD

    def is_folded(tip):
        return landmarks[tip][1] > landmarks[tip - 2][1] + THRESHOLD

    # ✋ A → all fingers folded
    if expected == "A":
        for name, tip in tips.items():
            if not is_folded(tip):
                feedback.append(f"{name} finger should be folded")

    # ✋ B → all fingers straight
    elif expected == "B":
        for name, tip in tips.items():
            if not is_straight(tip):
                feedback.append(f"{name} finger should be straight")

    # ✋ Default fallback
    if len(feedback) == 0:
        return "Adjust your hand posture"

    return ", ".join(feedback)


# --------------------------
# API ROUTE
# --------------------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]
        expected = request.form.get("expected")  # 🔥 IMPORTANT

        img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image"})

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        ) as hands:

            results = hands.process(img_rgb)

            if not results.multi_hand_landmarks:
                return jsonify({"error": "No hand detected"})

            hand_landmarks = results.multi_hand_landmarks[0]

            landmark_list = calc_landmark_list(img, hand_landmarks)
            processed = pre_process_landmark(landmark_list)

            df = pd.DataFrame(processed).transpose()

            predictions = model.predict(df, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)

            label = alphabet[predicted_classes[0]]
            confidence = float(np.max(predictions))

            # 🔥 FINAL CORRECT LOGIC
            if label == expected:
                feedback = "Correct Gesture ✅"
            else:
                feedback = generate_feedback(expected, landmark_list)

            return jsonify({
                "prediction": label,
                "confidence": confidence,
                "feedback": feedback
            })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)})


# --------------------------
# RUN SERVER
# --------------------------

if __name__ == "__main__":
    app.run(debug=True)