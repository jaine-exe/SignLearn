import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import os

# =============================
# CONFIG
# =============================
VIDEO_PATH = "videos/alphabet.mp4"
OUTPUT_CSV = "dataset/isl_landmarks.csv"
ASSETS_FOLDER = "assets"

# Exact image timestamps (seconds)
LETTER_IMAGE_TIMES = {
    "A": 10,
    "B": 13,
    "C": 17,
    "D": 20,
    "E": 23,
    "F": 26,
    "G": 29,
    "H": 33,
    "I": 36,
    "J": 38,
    "K": 42,
    "L": 45,
    "M": 48,
    "N": 50,
    "O": 54,
    "P": 57,
    "Q": 60,
    "R": 63,
    "S": 66,
    "T": 70,
    "U": 73,
    "V": 76,
    "W": 78,
    "X": 81,
    "Y": 85,
    "Z": 88
}

LETTERS = list(LETTER_IMAGE_TIMES.keys())
SAMPLE_EVERY_N_FRAMES = 10

os.makedirs("dataset", exist_ok=True)
os.makedirs(ASSETS_FOLDER, exist_ok=True)

mp_hands = mp.solutions.hands

def extract_landmarks(frame, hands):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    left_hand = np.zeros(63)
    right_hand = np.zeros(63)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])

            label = handedness.classification[0].label

            if label == "Left":
                left_hand = np.array(coords)
            else:
                right_hand = np.array(coords)

    features = np.concatenate([left_hand, right_hand])

    # Safety: force exactly 126 features
    if len(features) > 126:
        features = features[:126]
    elif len(features) < 126:
        padded = np.zeros(126)
        padded[:len(features)] = features
        features = padded

    return features, results

# =============================
# OPEN VIDEO
# =============================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"❌ Could not open video: {VIDEO_PATH}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps if fps > 0 else 0

print("====================================")
print(f"✅ Video opened: {VIDEO_PATH}")
print(f"FPS: {fps}")
print(f"Total Frames: {total_frames}")
print(f"Duration: {duration:.2f} seconds")
print("====================================")

data = []

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
) as hands:

    for i, letter in enumerate(LETTERS):
        image_sec = LETTER_IMAGE_TIMES[letter]

        # Save reference image
        image_frame = int(image_sec * fps)
        if image_frame < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, image_frame)
            ret, exact_img = cap.read()

            if ret:
                image_path = os.path.join(ASSETS_FOLDER, f"{letter}.png")
                cv2.imwrite(image_path, exact_img)
                print(f"📸 Saved reference image: {image_path} @ {image_sec}s")

        # Dataset collection interval
        start_sec = image_sec
        if i < len(LETTERS) - 1:
            end_sec = LETTER_IMAGE_TIMES[LETTERS[i + 1]]
        else:
            end_sec = min(start_sec + 3, duration)

        print(f"🔤 Collecting dataset for {letter}: {start_sec:.2f}s to {end_sec:.2f}s")

        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)

        if start_frame >= total_frames:
            print(f"⚠ Skipping {letter}")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame
        saved_count = 0

        while frame_idx <= end_frame and frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_idx - start_frame) % SAMPLE_EVERY_N_FRAMES == 0:
                features, results = extract_landmarks(frame, hands)

                if not np.all(features == 0):
                    row = list(features) + [letter]
                    data.append(row)
                    saved_count += 1

            frame_idx += 1

        print(f"   ✅ Samples saved for {letter}: {saved_count}")

cap.release()

# Save CSV properly
columns = [f"f{i}" for i in range(126)] + ["label"]
df = pd.DataFrame(data, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)

print("\n====================================")
print(f"✅ Dataset saved to: {OUTPUT_CSV}")
print(f"✅ Total samples collected: {len(df)}")
print("====================================")