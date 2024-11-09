import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque
from PIL import Image, ImageDraw, ImageFont

# Load pre-trained exercise classification model
model_path = 'model\model.keras'
classification_model = tf.keras.models.load_model(model_path)

# BlazePose model initialization
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=2, static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define exercise labels
unique_labels = ['스텝 백워드 다이나믹 런지', '스탠딩 니업', '바벨 로우', 
                 '버피 테스트', '플랭크', '시저크로스', '힙쓰러스트', '푸시업', 
                 '업라이트로우', '스텝 포워드 다이나믹 런지', '굿모닝', 
                 '바벨 스티프 데드리프트', 'Y - Exercise', '라잉 레그 레이즈', 
                 '스탠딩 사이드 크런치', '사이드 런지', '니푸쉬업', '크런치', 
                 '바이시클 크런치', '크로스 런지', '프런트 레이즈']

# Font path for Korean text (adjust based on your system)
font_path = 'NanumGothic.ttf'

# Function to extract keypoints from BlazePose results
def extract_keypoints(results):
    keypoints = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            keypoints.append(landmark.x)
            keypoints.append(landmark.y)
    return [np.mean(keypoints)] if keypoints else [0]

# Function to predict the exercise based on the sequence of keypoints
def predict_exercise(sequence):
    sequence = np.array(sequence).reshape(1, len(sequence), 1)  # Reshape for Conv1D model
    prediction = classification_model.predict(sequence)
    exercise_idx = np.argmax(prediction)
    confidence = prediction[0][exercise_idx] * 100
    return exercise_idx, confidence

# Function to draw Korean text on an image frame
def draw_text_korean(image, text, position, font_path, font_size=30, color=(255, 255, 255)):
    font = ImageFont.truetype(font_path, font_size)
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

# Streamlit app setup
st.title("Exercise Video Analyzer")
st.write("Upload a video file to analyze exercises.")

uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi"])

if uploaded_video is not None:
    # Save uploaded video temporarily for processing
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_video.getbuffer())

    # Display uploaded video in Streamlit UI
    st.video(uploaded_video)

    # Video capture and analysis setup using OpenCV and BlazePose
    cap = cv2.VideoCapture("temp_video.mp4")
    sequence_length = 17  # Number of frames to consider for each prediction
    frame_sequence = deque(maxlen=sequence_length)

    # Prepare output video writer (processed video will be saved)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('processed_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            keypoints = extract_keypoints(results)
            frame_sequence.append(keypoints)

            if len(frame_sequence) == sequence_length:
                exercise_idx, confidence = predict_exercise(frame_sequence)
                exercise_label = unique_labels[exercise_idx]
                exercise_text = f"운동: {exercise_label} ({confidence:.2f}%)"
                color = (0, 255, 0) if confidence >= 75 else (0, 0, 255)
                frame = draw_text_korean(frame, exercise_text, (10, 50), font_path, color=color)

        # Write processed frame to output video file
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    cap.release()
    out.release()

    st.success("Video processed successfully!")

    # Provide download link for processed video
    with open("processed_video.mp4", "rb") as f:
        st.download_button(
            label="Download Processed Video",
            data=f,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )
