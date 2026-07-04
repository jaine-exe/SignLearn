import cv2
import mediapipe as mp
import copy
import itertools
from tensorflow import keras
import numpy as np
import pandas as pd
import string

# load model
model = keras.models.load_model("model.h5")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

alphabet = ['1','2','3','4','5','6','7','8','9']
alphabet += list(string.ascii_uppercase)

# ---------------- FUNCTIONS ---------------- #

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        x = min(int(landmark.x * image_width), image_width - 1)
        y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([x, y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = temp_landmark_list[0]

    for i in range(len(temp_landmark_list)):
        temp_landmark_list[i][0] -= base_x
        temp_landmark_list[i][1] -= base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(list(map(abs, temp_landmark_list)))

    if max_value != 0:
        temp_landmark_list = [n / max_value for n in temp_landmark_list]

    return temp_landmark_list


# ---------------- MAIN ---------------- #

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        image = cv2.flip(image, 1)

        if not success:
            continue

        image.flags.writeable = False
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        image.flags.writeable = True
        image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        feedback_text = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                landmark_list = calc_landmark_list(image, hand_landmarks)
                processed = pre_process_landmark(landmark_list)

                df = pd.DataFrame(processed).transpose()

                # ---------- PREDICTION ---------- #
                predictions = model.predict(df, verbose=0)
                predicted_class = np.argmax(predictions)
                confidence = np.max(predictions)

                label = alphabet[predicted_class]

                # ---------- FEEDBACK ---------- #

                # Confidence feedback
                if confidence < 0.7:
                    feedback_text = "Adjust hand position"

                else:
                    feedback_text = f"{label} ({confidence:.2f})"

                # Finger checks

                # Index finger (tip=8, joint=6)
                if landmark_list[8][1] > landmark_list[6][1]:
                    feedback_text = "Lift your index finger"

                # Thumb (tip=4, joint=2)
                if landmark_list[4][0] < landmark_list[2][0]:
                    feedback_text = "Adjust thumb position"

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # ---------- DISPLAY ---------- #
        cv2.putText(image, feedback_text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255), 2)

        cv2.imshow('ISL Detection with Feedback', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()