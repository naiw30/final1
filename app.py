import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os

# 전체 레이아웃을 넓게 설정
st.set_page_config(layout="wide")

# 제목 설정
st.title("운동 분류 및 자세 검출 앱")

# BlazePose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=2, 
                    static_image_mode=False, 
                    min_detection_confidence=0.5, 
                    min_tracking_confidence=0.5)

# 운동 분류 모델 로드 (경로는 필요에 맞게 수정하세요)
model_path = 'model/model.keras'
classification_model = tf.keras.models.load_model(model_path)
st.success("운동 분류 모델이 성공적으로 로드되었습니다.")

# 운동 라벨 정의
unique_labels = ['스텝 백워드 다이나믹 런지', '스탠딩 니업', '바벨 로우', '버피 테스트', '플랭크', '시저크로스', 
                 '힙쓰러스트', '푸시업', '업라이트로우', '스텝 포워드 다이나믹 런지', '굿모닝', 
                 '바벨 스티프 데드리프트', 'Y - Exercise', '라잉 레그 레이즈', '스탠딩 사이드 크런치', 
                 '사이드 런지', '니푸쉬업', '크런치', '바이시클 크런치', '크로스 런지', '프런트 레이즈']

# 폰트 파일 경로 설정 (필요에 맞게 수정하세요)
font_path = 'NanumGothic.ttf'

# 비디오 파일 업로드
uploaded_file = st.file_uploader("비디오 파일을 업로드하세요", type=["mp4", "mov", "avi"])

# 전체 레이아웃을 컨테이너로 감싸기
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.header("원본 영상")
        if uploaded_file is not None:
            st.video(uploaded_file)
        else:
            st.write("원본 영상을 표시하려면 비디오 파일을 업로드하세요.")

    with col2:
        st.header("운동 분류 결과 영상")
        result_placeholder = st.empty()

# 입력 데이터 전처리 함수 (BlazePose에서 추출한 좌표를 평균하여 단일 특징 값으로 변환)
def extract_keypoints(results):
    keypoints = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            keypoints.append(landmark.x)
            keypoints.append(landmark.y)
    if len(keypoints) > 0:
        keypoints = [np.mean(keypoints)]  # 단일 채널로 변환
    else:
        keypoints = [0]  # 데이터가 없는 경우 0으로 채워줌
    return keypoints

# 운동 예측 함수
def predict_exercise(sequence):
    sequence = np.array(sequence).reshape(1, len(sequence), 1)  # Conv1D 모델에 맞는 형식으로 맞춤
    prediction = classification_model.predict(sequence)
    exercise_idx = np.argmax(prediction)
    confidence = prediction[0][exercise_idx] * 100
    return exercise_idx, confidence

# 이미지에 한글 텍스트 추가 함수
def draw_text_korean(image, text, position, font_path, font_size=30, color=(255, 255, 255)):
    font = ImageFont.truetype(font_path, font_size)
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

if uploaded_file is not None:
    # 비디오 처리 버튼 클릭 이벤트 처리
    if st.button("운동 분류 실행"):
        with tempfile.NamedTemporaryFile(delete=False) as temp_input:
            temp_input.write(uploaded_file.read())
            temp_input_path = temp_input.name
        
        cap = cv2.VideoCapture(temp_input_path)
        sequence_length = 17  # 시퀀스 프레임 설정
        frame_sequence = deque(maxlen=sequence_length)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
            output_path = temp_output.name

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                keypoints = extract_keypoints(results)
                frame_sequence.append(keypoints)

                # 운동 예측 수행
                if len(frame_sequence) == sequence_length:
                    exercise_idx, confidence = predict_exercise(frame_sequence)
                    exercise_label = unique_labels[exercise_idx]
                    exercise_text = f"운동: {exercise_label} ({confidence:.2f}%)"
                    color = (0, 255, 0) if confidence >= 75 else (0, 0, 255)

                    # 한글 텍스트를 프레임에 추가
                    frame_with_text = draw_text_korean(frame_rgb, exercise_text, (10, 50), font_path)

                    # 결과 프레임을 저장 (BGR로 변환 후 저장해야 함)
                    out.write(cv2.cvtColor(frame_with_text, cv2.COLOR_RGB2BGR))

        cap.release()
        out.release()

        # 결과 비디오를 스트림릿에 표시 및 다운로드 링크 제공
        result_placeholder.video(output_path)

        with open(output_path, "rb") as file:
            st.download_button(
                label="결과 영상 다운로드",
                data=file,
                file_name="classified_video.mp4",
                mime="video/mp4"
            )
