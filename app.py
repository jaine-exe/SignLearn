import streamlit as st
import json
import os
import numpy as np
import cv2
from PIL import Image
import mediapipe as mp
import time

from database import (
    save_result,
    get_results,
    get_average_score,
    get_completed_gestures,
    register_user,
    login_user,
    get_all_users,
    get_all_results
)
from predict_model import predict_from_image

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="ISL Learning System",
    page_icon="🤟",
    layout="wide"
)

# =============================
# LOAD REFERENCE DATA
# =============================
with open("reference_data.json", "r", encoding="utf-8") as f:
    reference_data = json.load(f)

# =============================
# SESSION STATE
# =============================
if "page" not in st.session_state:
    st.session_state.page = "login"

if "selected_module" not in st.session_state:
    st.session_state.selected_module = None

if "selected_gesture" not in st.session_state:
    st.session_state.selected_gesture = None

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "username" not in st.session_state:
    st.session_state.username = None

if "role" not in st.session_state:
    st.session_state.role = None

# =============================
# CUSTOM CSS
# =============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #020617, #0f172a, #1e293b);
    color: #f8fafc;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
h1 {
    font-size: 3rem !important;
    font-weight: 800 !important;
    color: #f8fafc !important;
}
h2 {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #f8fafc !important;
}
h3 {
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: #f8fafc !important;
}
p, label, .stMarkdown {
    color: #cbd5e1 !important;
    font-size: 1.05rem;
}
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #3b82f6);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.75rem 1rem;
    font-size: 1rem;
    font-weight: 700;
    width: 100%;
    transition: 0.25s ease;
    box-shadow: 0 8px 18px rgba(37, 99, 235, 0.35);
}
.stButton > button:hover {
    transform: translateY(-2px);
    background: linear-gradient(135deg, #3b82f6, #60a5fa);
    box-shadow: 0 12px 24px rgba(59, 130, 246, 0.45);
}
.card {
    background: rgba(255, 255, 255, 0.06);
    border-radius: 22px;
    padding: 24px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.28);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 20px;
}
.small-card {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.25);
    border: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 16px;
}
.center-text {
    text-align: center;
}
.home-card {
    background: rgba(255, 255, 255, 0.06);
    border-radius: 22px;
    padding: 30px 24px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.28);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 16px;
    min-height: 220px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    text-align: center;
}
.result-box {
    background: rgba(255, 255, 255, 0.06);
    border-radius: 20px;
    padding: 22px;
    min-height: 170px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
    box-shadow: 0 8px 20px rgba(0,0,0,0.25);
    border: 1px solid rgba(255,255,255,0.08);
}
.result-title {
    font-size: 1.6rem;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 20px;
}
.result-value {
    width: 100%;
    padding: 18px 20px;
    border-radius: 14px;
    font-size: 1.25rem;
    font-weight: 700;
    text-align: left;
    color: white;
}
.prediction-bg {
    background: linear-gradient(135deg, #14532d, #166534, #047857);
}
.score-bg {
    background: linear-gradient(135deg, #4d4f0f, #5b5f12, #6b7280);
}
.correct-bg {
    background: linear-gradient(135deg, #14532d, #166534, #047857);
}
.incorrect-bg {
    background: linear-gradient(135deg, #7f1d1d, #991b1b, #7c2d12);
}
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04);
    padding: 14px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
}
.stSuccess, .stWarning, .stError, .stInfo {
    border-radius: 14px !important;
}
img {
    border-radius: 18px;
}
</style>
""", unsafe_allow_html=True)

# =============================
# FEEDBACK LOGIC
# =============================
def generate_feedback(score, predicted, expected):
    if str(predicted) == str(expected) and score >= 85:
        return "Correct", "Excellent! Your gesture is very accurate."
    elif str(predicted) == str(expected) and score >= 70:
        return "Correct", "Good attempt! Slight improvement needed."
    else:
        return "Incorrect", "Gesture does not match well. Recheck the instruction and try again."

# =============================
# MEDIAPIPE DRAW
# =============================
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def draw_landmarks_on_image(image, results):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = image_rgb.copy()

    if results and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                output,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    return output

# =============================
# PROCESS IMAGE
# =============================
def process_gesture_image(image_bgr, expected_gesture):
    predicted, confidence, features, results = predict_from_image(image_bgr)

    if predicted is None:
        st.error("❌ No hand detected.")
        st.info("""
### Try this:
- Keep your full hand visible in the frame
- Move your hand slightly away from the camera
- Avoid cutting fingers out of the image
- Use better lighting
- Keep a plain background if possible
- Hold the gesture steady for 2–3 seconds
""")
        return

    landmark_image = draw_landmarks_on_image(image_bgr, results)

    st.markdown("## ✨ Result")
    st.image(landmark_image, caption="Detected Hand Landmarks", use_container_width=True)

    score = confidence if str(predicted) == str(expected_gesture) else max(40, confidence - 30)
    correctness, feedback = generate_feedback(score, str(predicted), str(expected_gesture))

    save_result(st.session_state.user_id, expected_gesture, str(predicted), confidence, score, correctness)

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown(f"""
        <div class="result-box">
            <div class="result-title">📌 Prediction</div>
            <div class="result-value prediction-bg">{predicted}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="result-box">
            <div class="result-title">📊 Score</div>
            <div class="result-value score-bg">{score:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        correctness_class = "correct-bg" if correctness == "Correct" else "incorrect-bg"
        st.markdown(f"""
        <div class="result-box">
            <div class="result-title">✅ Correctness</div>
            <div class="result-value {correctness_class}">{correctness}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 💡 Feedback")
    st.info(feedback)

    with st.expander("Show Landmark Coordinates"):
        st.write(features.tolist())

# =============================
# IN-PAGE CAMERA AUTO CAPTURE
# =============================
def streamlit_live_capture_with_timer(seconds=5):
    frame_placeholder = st.empty()
    timer_placeholder = st.empty()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return None, "❌ Could not open webcam."

    captured_frame = None
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None, "❌ Failed to read from webcam."

        frame = cv2.flip(frame, 1)
        elapsed = int(time.time() - start_time)
        remaining = seconds - elapsed

        display_frame = frame.copy()

        cv2.putText(
            display_frame,
            f"Capturing in {max(remaining, 0)}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        timer_placeholder.info(f"⏳ Auto-capturing in {max(remaining, 0)} seconds...")

        if remaining <= 0:
            captured_frame = frame.copy()
            break

        time.sleep(0.03)

    cap.release()
    frame_placeholder.empty()
    timer_placeholder.empty()

    return captured_frame, None

# =============================
# LOGIN PAGE
# =============================
def login_page():
    st.title("🔐 Login to ISL Learning System")
    st.write("Login as a user or admin to continue.")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("👤 User / Admin Login")

        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login"):
            user = login_user(username.strip(), password.strip())

            if user:
                st.success(f"Login successful! Welcome {user[1]}")
                st.session_state.logged_in = True
                st.session_state.user_id = user[0]
                st.session_state.username = user[1]
                st.session_state.role = user[2]

                if user[2] == "admin":
                    st.session_state.page = "admin_dashboard"
                else:
                    st.session_state.page = "home"

                st.rerun()
            else:
                st.error("Invalid username or password.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📝 New User Registration")

        new_username = st.text_input("Choose Username", key="register_username")
        new_password = st.text_input("Choose Password", type="password", key="register_password")

        if st.button("Register"):
            if not new_username or not new_password:
                st.warning("Please enter both username and password.")
            else:
                success, message = register_user(new_username.strip(), new_password.strip())
                if success:
                    st.success(message)
                    st.info("Now login using the same username and password.")
                else:
                    st.error(message)
        st.markdown('</div>', unsafe_allow_html=True)

# =============================
# SIDEBAR
# =============================
def sidebar_menu():
    with st.sidebar:
        st.markdown(f"### 👋 Welcome, {st.session_state.username}")
        st.markdown(f"**Role:** {st.session_state.role.capitalize()}")

        if st.session_state.role == "user":
            if st.button("🏠 Home"):
                st.session_state.page = "home"
            if st.button("📊 Progress"):
                st.session_state.page = "progress"

        if st.session_state.role == "admin":
            if st.button("🛠 Admin Dashboard"):
                st.session_state.page = "admin_dashboard"

        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.username = None
            st.session_state.role = None
            st.session_state.page = "login"
            st.rerun()

# =============================
# HOME PAGE
# =============================
def home_page():
    sidebar_menu()

    st.title("🤟Indian Sign Language Learning System")
    st.subheader("Learn • Practice • Improve")
    st.write("An interactive system to help learners study, practice, and evaluate Indian Sign Language gestures.")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="home-card">
            <h2>🚀 Start Learning</h2>
            <p>Begin your ISL learning journey with guided practice and reference gestures.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Learning", key="start_learning_btn", use_container_width=True):
            st.session_state.page = "modules"

    with col2:
        st.markdown("""
        <div class="home-card">
            <h2>📊 View Progress</h2>
            <p>Track your completed gestures, score, and learning history.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("View Progress", key="view_progress_btn", use_container_width=True):
            st.session_state.page = "progress"

# =============================
# MODULE PAGE
# =============================
def module_page():
    sidebar_menu()

    st.title("🗂️ Select Learning Module")
    st.write("Choose the stage you want to begin with.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card center-text">
            <h2>🔤 Alphabets</h2>
            <p>Learn and practice basic ISL alphabet gestures.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Alphabets", key="module_alphabets", use_container_width=True):
            st.session_state.selected_module = "alphabets"
            st.session_state.page = "gesture_list"

    with col2:
        st.markdown("""
        <div class="card center-text">
            <h2>📝 Words</h2>
            <p>Learn and practice commonly used ISL words.</p>
        </div>
        """, unsafe_allow_html=True)
        st.info("Coming Soon")

    with col3:
        st.markdown("""
        <div class="card center-text">
            <h2>💬 Phrases</h2>
            <p>Learn simple ISL phrases and short expressions.</p>
        </div>
        """, unsafe_allow_html=True)
        st.info("Coming Soon")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⬅ Back to Home"):
        st.session_state.page = "home"

# =============================
# GESTURE LIST PAGE
# =============================
def gesture_list_page():
    sidebar_menu()

    module = st.session_state.selected_module
    gestures = list(reference_data[module].keys())

    st.title(f"📚 {module.capitalize()}")
    st.write("Select a gesture to practice")

    cols = st.columns(3)

    for i, gesture in enumerate(gestures):
        with cols[i % 3]:
            if st.button(gesture, key=f"gesture_{gesture}", use_container_width=True):
                st.session_state.selected_gesture = gesture
                st.session_state.page = "practice"

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⬅ Back"):
        st.session_state.page = "modules"

# =============================
# PRACTICE PAGE
# =============================
def practice_page():
    sidebar_menu()

    module = st.session_state.selected_module
    gesture = st.session_state.selected_gesture
    data = reference_data[module][gesture]

    st.title(f"🖐 Gesture: {gesture}")

    st.markdown("## 📝 Instruction")
    st.info(data["instruction"])

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("## 📌 Reference")
        if "image" in data and data["image"]:
            image_path = data["image"]
            if os.path.exists(image_path):
                st.image(image_path, caption=f"Correct gesture for {gesture}", use_container_width=True)
            else:
                st.warning(f"Image not found: {image_path}")
        else:
            st.info("No reference image added yet.")

    with col2:
        st.markdown("## 🎯 Try Your Gesture")

        input_mode = st.radio(
            "Choose input method:",
            ["Upload Image", "Live Camera (In-App Auto Capture)"],
            horizontal=True
        )

        if input_mode == "Upload Image":
            uploaded = st.file_uploader("Upload your gesture", type=["jpg", "jpeg", "png"])

            if uploaded:
                pil_image = Image.open(uploaded).convert("RGB")
                image_np = np.array(pil_image)
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                st.image(image_np, caption="Your Uploaded Gesture", use_container_width=True)
                process_gesture_image(image_bgr, gesture)

        elif input_mode == "Live Camera (In-App Auto Capture)":
            st.info("Click below to start live camera preview inside the page. After the countdown, your gesture will be captured automatically.")

            countdown = st.selectbox("Select countdown time (seconds)", [3, 5, 7], index=1)

            if st.button("🎥 Start Auto Capture"):
                with st.spinner("Opening webcam..."):
                    captured_frame, error = streamlit_live_capture_with_timer(seconds=countdown)

                if error:
                    st.error(error)

                elif captured_frame is not None:
                    st.success("✅ Gesture captured successfully!")

                    image_rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
                    st.image(image_rgb, caption="Captured Gesture", use_container_width=True)

                    process_gesture_image(captured_frame, gesture)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("⬅ Back"):
        st.session_state.page = "gesture_list"

# =============================
# PROGRESS PAGE
# =============================
def progress_page():
    sidebar_menu()

    st.title("📊 Learning Progress")

    avg = get_average_score(st.session_state.user_id)
    completed = get_completed_gestures(st.session_state.user_id)
    results = get_results(st.session_state.user_id)

    st.subheader(f"Average Score: {avg}%")
    st.progress(avg / 100 if avg else 0)

    st.markdown("## ✅ Completed Gestures")
    st.write(completed if completed else "No gestures completed yet.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## 📜 Practice History")

    if not results:
        st.info("No practice results yet.")
    else:
        for r in results:
            _, user_id, g, p, c, s, cor, t = r
            st.markdown(f"""
            <div class="small-card">
                <b>Gesture:</b> {g}<br>
                <b>Predicted:</b> {p}<br>
                <b>Confidence:</b> {c:.2f}%<br>
                <b>Score:</b> {s:.2f}%<br>
                <b>Result:</b> {cor}<br>
                <b>Time:</b> {t}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⬅ Back to Home"):
        st.session_state.page = "home"

# =============================
# ADMIN DASHBOARD
# =============================
def admin_dashboard():
    sidebar_menu()

    st.title("🛠 Admin Dashboard")

    st.markdown("## 👥 Registered Users")
    users = get_all_users()

    if not users:
        st.info("No users found.")
    else:
        for u in users:
            uid, uname, role = u
            st.markdown(f"""
            <div class="small-card">
                <b>User ID:</b> {uid}<br>
                <b>Username:</b> {uname}<br>
                <b>Role:</b> {role}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## 📜 All Practice Results")

    all_results = get_all_results()

    if not all_results:
        st.info("No practice results available.")
    else:
        for r in all_results:
            _, username, g, p, c, s, cor, t = r
            st.markdown(f"""
            <div class="small-card">
                <b>User:</b> {username}<br>
                <b>Gesture:</b> {g}<br>
                <b>Predicted:</b> {p}<br>
                <b>Confidence:</b> {c:.2f}%<br>
                <b>Score:</b> {s:.2f}%<br>
                <b>Result:</b> {cor}<br>
                <b>Time:</b> {t}
            </div>
            """, unsafe_allow_html=True)

# =============================
# ROUTER
# =============================
if not st.session_state.logged_in:
    login_page()
else:
    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "modules":
        module_page()
    elif st.session_state.page == "gesture_list":
        gesture_list_page()
    elif st.session_state.page == "practice":
        practice_page()
    elif st.session_state.page == "progress":
        progress_page()
    elif st.session_state.page == "admin_dashboard":
        admin_dashboard()