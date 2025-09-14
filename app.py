import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os

# ========================
# Haar Cascade Setup
# ========================
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ========================
# Helper Functions
# ========================
def detect_faces_local(image: Image.Image):
    """Detect faces and return list of cropped face arrays"""
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    detected_faces = []
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (100, 100))
        detected_faces.append(face_resized)

    return detected_faces

def mark_attendance(name):
    """Mark attendance with date and time"""
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    if not any((entry["Name"] == name and entry["Date"] == date) for entry in st.session_state.attendance):
        st.session_state.attendance.append({"Name": name, "Date": date, "Time": time})

# ========================
# Initialize
# ========================
st.title("🧑‍💼 Facial Recognition Attendance System ")

menu = [
    "Register Face (Upload)",
    "Register Face (Webcam)",
    "Mark Attendance (Upload)",
    "Mark Attendance (Webcam)",
    "View Attendance Report",
    "📊 Daily Summary & Analytics"
]
choice = st.sidebar.selectbox("Menu", menu)

# In-memory DB
if "face_db" not in st.session_state:
    st.session_state.face_db = {}  # {name: [face_arrays]}
if "attendance" not in st.session_state:
    st.session_state.attendance = []
if "label_map" not in st.session_state:
    st.session_state.label_map = {}  # numeric_id -> name
if "recognizer" not in st.session_state:
    st.session_state.recognizer = cv2.face.LBPHFaceRecognizer_create()

# Function to retrain recognizer
def retrain_recognizer():
    labels, faces = [], []
    label_map = {}
    label_counter = 0

    for name, face_list in st.session_state.face_db.items():
        if not face_list:
            continue
        label_counter += 1
        label_map[label_counter] = name
        for face in face_list:
            faces.append(face)
            labels.append(label_counter)

    if faces:
        st.session_state.recognizer.train(faces, np.array(labels))
        st.session_state.label_map = label_map

# ========================
# REGISTER FACE (Upload)
# ========================
if choice == "Register Face (Upload)":
    st.subheader("Register multiple faces from uploaded images")

    num_faces = st.number_input("How many people do you want to register?", min_value=1, max_value=10, value=1)

    for i in range(num_faces):
        with st.expander(f"Person {i+1}"):
            name = st.text_input(f"Enter name for person {i+1}", key=f"name_{i}")
            img_files = st.file_uploader(
                f"Upload image(s) for {name if name else 'person ' + str(i+1)}",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key=f"upload_{i}"
            )

            if st.button(f"Register {name}", key=f"register_btn_{i}"):
                if name and img_files:
                    for img_file in img_files:
                        image = Image.open(img_file)
                        faces = detect_faces_local(image)

                        if faces:
                            if name not in st.session_state.face_db:
                                st.session_state.face_db[name] = []
                            st.session_state.face_db[name].extend(faces)
                            st.success(f"✅ {name} registered from {img_file.name}")
                        else:
                            st.warning(f"⚠ No face detected in {img_file.name}")
                    retrain_recognizer()
                else:
                    st.warning("Please enter a name and upload at least one image.")

# ========================
# REGISTER FACE (Webcam)
# ========================
elif choice == "Register Face (Webcam)":
    st.subheader("Register a new face via Webcam")
    name = st.text_input("Enter Name")
    captured_img = st.camera_input("Capture photo")

    if st.button("Register"):
        if name and captured_img:
            image = Image.open(captured_img)
            faces = detect_faces_local(image)

            if faces:
                if name not in st.session_state.face_db:
                    st.session_state.face_db[name] = []
                st.session_state.face_db[name].extend(faces)
                st.success(f"{name} registered successfully with webcam image!")
                retrain_recognizer()
            else:
                st.warning("No face detected. Try again.")
        else:
            st.warning("Please enter name and capture an image.")

# ========================
# MARK ATTENDANCE (Upload)
# ========================
elif choice == "Mark Attendance (Upload)":
    st.subheader("Mark Attendance via Uploaded Images")
    img_files = st.file_uploader(
        "Upload Images for Attendance", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )

    if st.button("Check Attendance"):
        if img_files:
            if not st.session_state.label_map:
                st.error("⚠ No registered faces found. Please register first.")
            else:
                for img_file in img_files:
                    image = Image.open(img_file)
                    detected_faces = detect_faces_local(image)

                    if detected_faces:
                        for detected_face in detected_faces:
                            label, confidence = st.session_state.recognizer.predict(detected_face)
                            if confidence < 70:  # lower = better match
                                name = st.session_state.label_map[label]
                                mark_attendance(name)
                                st.success(f"✅ Attendance marked for {name} (conf={confidence:.2f})")
                            else:
                                st.warning(f"⚠ Unknown face in {img_file.name}")
                    else:
                        st.warning(f"⚠ No face detected in {img_file.name}")
        else:
            st.warning("Please upload at least one image.")

# ========================
# MARK ATTENDANCE (Webcam)
# ========================
elif choice == "Mark Attendance (Webcam)":
    st.subheader("Mark Attendance via Webcam")
    captured_img = st.camera_input("Capture photo for attendance")

    if st.button("Check Attendance"):
        if captured_img:
            if not st.session_state.label_map:
                st.error("⚠ No registered faces found. Please register first.")
            else:
                image = Image.open(captured_img)
                detected_faces = detect_faces_local(image)

                if detected_faces:
                    for detected_face in detected_faces:
                        label, confidence = st.session_state.recognizer.predict(detected_face)
                        if confidence < 70:
                            name = st.session_state.label_map[label]
                            mark_attendance(name)
                            st.success(f"✅ Attendance marked for {name} (conf={confidence:.2f})")
                        else:
                            st.warning("⚠ Unknown face in capture")
                else:
                    st.warning("No face detected. Try again.")
        else:
            st.warning("Please capture an image.")

# ========================
# VIEW ATTENDANCE REPORT
# ========================
elif choice == "View Attendance Report":
    st.subheader("📊 Attendance Report")

    if st.session_state.attendance:
        df = pd.DataFrame(st.session_state.attendance)
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Attendance Report (CSV)",
            data=csv,
            file_name="attendance_report.csv",
            mime="text/csv"
        )
    else:
        st.info("No attendance records yet.")

# ========================
# DAILY SUMMARY & ANALYTICS
# ========================
elif choice == "📊 Daily Summary & Analytics":
    st.markdown("## 📊 Daily Attendance Summary & Analytics")

    if st.session_state.attendance:
        df = pd.DataFrame(st.session_state.attendance)
        today = datetime.now().strftime("%Y-%m-%d")
        today_df = df[df["Date"] == today]

        total_registered = len(st.session_state.face_db)
        present_today = today_df["Name"].nunique()
        absentees = total_registered - present_today

        col1, col2, col3 = st.columns(3)
        col1.metric("👨‍🎓 Total Registered", total_registered)
        col2.metric("✅ Present Today", present_today)
        col3.metric("❌ Absent Today", absentees)

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🥧 Attendance Distribution (Today)")
            fig1, ax1 = plt.subplots(figsize=(3,3))
            ax1.pie([present_today, absentees], labels=["Present", "Absent"],
                    autopct="%1.1f%%", startangle=90, shadow=True,
                    explode=(0.05, 0.05), textprops={"fontsize":10})
            ax1.axis("equal")
            st.pyplot(fig1)

        with col2:
            st.markdown("### 📊 Attendance Count by Student")
            fig2, ax2 = plt.subplots(figsize=(3.5,3))
            bars = df["Name"].value_counts().plot(kind="bar", ax=ax2, color="#2196F3")
            for bar in bars.patches:
                bar.set_linewidth(1)
                bar.set_edgecolor("black")
            ax2.set_ylabel("Count")
            ax2.set_xlabel("Student")
            ax2.set_title("Attendance per Student")
            plt.tight_layout()
            st.pyplot(fig2)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("### 📈 Attendance Trend Over Time")
            trend_df = df.groupby("Date")["Name"].nunique().reset_index()
            fig3, ax3 = plt.subplots(figsize=(3.5,3))
            ax3.plot(trend_df["Date"], trend_df["Name"], marker="o",
                     color="#9C27B0", linewidth=2, linestyle="-")
            ax3.grid(True, linestyle="--", alpha=0.6)
            ax3.set_xlabel("Date")
            ax3.set_ylabel("Present")
            ax3.set_title("Daily Trend")
            plt.xticks(rotation=45, fontsize=8)
            plt.yticks(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig3)
    else:
        st.info("No attendance records yet.")