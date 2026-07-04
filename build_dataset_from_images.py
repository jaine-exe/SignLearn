import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# =============================
# CONFIG
# =============================
DATASET_FOLDER = "external_dataset"           # your A-Z image folders
OUTPUT_CSV = "dataset/isl_landmarks.csv"

# valid image extensions
VALID_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# create output folder if missing
os.makedirs("dataset", exist_ok=True)

# MediaPipe Hands
mp_hands = mp.solutions.hands


# =============================
# HELPER: normalize folder label
# =============================
def normalize_label(folder_name):
    name = str(folder_name).strip()

    # If folder names are lowercase like a, b, c
    if len(name) == 1 and name.isalpha():
        return name.upper()

    # If folder names are numeric like 0,1,2 -> A,B,C
    if name.isdigit():
        num = int(name)
        if 0 <= num <= 25:
            return chr(ord("A") + num)

    # If folder names are already A-Z
    return name.upper()


# =============================
# HELPER: extract 126 landmarks
# =============================
def extract_landmarks_from_image(image, hands):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    left_hand = np.zeros(63)
    right_hand = np.zeros(63)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])

            label = handedness.classification[0].label  # Left or Right

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

    return features


# =============================
# MAIN
# =============================
def main():
    if not os.path.exists(DATASET_FOLDER):
        print(f"❌ Folder not found: {DATASET_FOLDER}")
        return

    rows = []
    total_images = 0
    total_used = 0

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    ) as hands:

        for folder in sorted(os.listdir(DATASET_FOLDER)):
            folder_path = os.path.join(DATASET_FOLDER, folder)

            if not os.path.isdir(folder_path):
                continue

            label = normalize_label(folder)

            # keep only A-Z
            if label not in [chr(ord("A") + i) for i in range(26)]:
                print(f"⚠ Skipping unknown folder: {folder}")
                continue

            print(f"\n🔤 Processing folder: {folder} → label {label}")

            used_count = 0
            file_count = 0

            for filename in os.listdir(folder_path):
                if not filename.lower().endswith(VALID_EXTS):
                    continue

                file_count += 1
                total_images += 1

                img_path = os.path.join(folder_path, filename)
                image = cv2.imread(img_path)

                if image is None:
                    continue

                features = extract_landmarks_from_image(image, hands)

                # only keep rows where at least some landmarks were detected
                if not np.all(features == 0):
                    row = list(features) + [label]
                    rows.append(row)
                    used_count += 1
                    total_used += 1

            print(f"   Total images: {file_count}")
            print(f"   Usable samples: {used_count}")

    # Save CSV
    columns = [f"f{i}" for i in range(126)] + ["label"]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n======================================")
    print(f"✅ Dataset saved to: {OUTPUT_CSV}")
    print(f"📦 Total images scanned: {total_images}")
    print(f"✅ Total usable samples: {total_used}")
    print("======================================")

    if len(df) > 0:
        print("\nSamples per label:")
        print(df["label"].value_counts())
    else:
        print("❌ No usable landmarks extracted.")


if __name__ == "__main__":
    main()