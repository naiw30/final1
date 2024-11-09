import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque
from PIL import Image, ImageDraw, ImageFont

# 폰트 파일 경로 설정
font_path = 'C:\\Users\\itwill\\Desktop\\나눔 글꼴\\나눔고딕\\NanumFontSetup_TTF_GOTHIC\\NanumGothic.ttf'

# BlazePose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=2, static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 학습된 운동 분류 모델 로드
model_path = 'D:\\찐막모델\\model.keras'
classification_model = tf.keras.models.load_model(model_path)
print("운동 분류 모델이 성공적으로 로드되었습니다.")

# 운동 라벨 정의
unique_labels = ['스텝 백워드 다이나믹 런지', '스탠딩 니업', '바벨 로우', 
								 '버피 테스트', '플랭크', '시저크로스', '힙쓰러스트', '푸시업', 
								 '업라이트로우', '스텝 포워드 다이나믹 런지', '굿모닝', 
								 '바벨 스티프 데드리프트', 'Y - Exercise', '라잉 레그 레이즈', 
								 '스탠딩 사이드 크런치', '사이드 런지', '니푸쉬업', '크런치', 
								 '바이시클 크런치', '크로스 런지', '프런트 레이즈']

# 비디오 파일 경로
video_path = 'D:\\테스트영상\\test3.mp4'
cap = cv2.VideoCapture(video_path)

# 시퀀스 프레임 설정
sequence_length = 17
frame_sequence = deque(maxlen=sequence_length)

# 입력 데이터 전처리 함수 (BlazePose에서 추출한 좌표를 평균하여 단일 특징 값으로 변환)
def extract_keypoints(results):
    keypoints = []
    
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            keypoints.append(landmark.x)
            keypoints.append(landmark.y)

    # 평균값을 계산하여 단일 채널로 변환
    if len(keypoints) > 0:
        keypoints = [np.mean(keypoints)]  # 단일 채널로 변환
    else:
        keypoints = [0]  # 데이터가 없는 경우 0으로 채워줌

    return keypoints

# 운동 예측 함수
def predict_exercise(sequence):
    sequence = np.array(sequence).reshape(1, sequence_length, 1)  # Conv1D 모델에 맞는 형식으로 맞춤
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

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width, _ = frame.shape
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
            frame = draw_text_korean(frame, exercise_text, (10, 50), font_path, color=color)

    cv2.imshow('Exercise and Posture Detection', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
