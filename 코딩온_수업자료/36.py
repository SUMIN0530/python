import cv2
import numpy as np
import matplotlib.pyplot as plt

# 객체 탐지
# 이미지에서 객체의 위치(바운딩 박스)와 클래스를 예측
# - 분류 : 이미지 전체가 무엇인가?
# - 탐지 : 어디에 무엇이 있는가?
# - 분할 : 픽셀 단위로 무엇인가?

# 모델 불러오기
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

images = [
    cv2.imread('./image1.jpg'),
    cv2.imread('./image2.jpg'),
    cv2.imread('./image3.jpg')
]

results = []

# 객체 탐지 알고리즘 실행
# 이미지 그리이스케일로 변환
for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1,
        minNeighbors=3, # 값이 높을수록 확실한 값만 검출 => 깐깐해짐 ('5' 였을 때 3번 사진 얼굴인식 x)
        minSize=(20, 20)
    )

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    img = cv2.resize(img, (300, 300))
    results.append(img)

# cv2.imshow("Image 1", results[0])
# cv2.imshow("Image 2", results[1])
# cv2.imshow("Image 3", results[2])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

from ultralytics import YOLO
# import cv2

model = YOLO('yolo11n.pt')

img = cv2.imread('./image2.jpg')

# 신뢰도 0.5로 객체 탐지 
results = model.predict(img, conf=0.5)

# 탐지 결과를 이미지 위에 그림
annotated_frame = results[0].plot()

# cv2.imshow('annotated_frame', annotated_frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 웹 캠 + YOLO
cap = cv2.VideoCapture(0)

if not cap.isOpened:
    raise RuntimeError('카메라를 열 수 없습니다')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 추론
    results = model(frame, verbose=False)

    # 바운딩 박스 + 라벨 그린 프레임
    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            label = f"{model.names[cls]} {conf:.2f}"

            # 박스
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)

            # 라벨 배경
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(frame,
                          (x1, y1 - th - 6),
                          (x1 + tw, y1),
                          (0, 255, 0), -1)

            # 라벨 텍스트
            cv2.putText(frame, label,
                        (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1)

    cv2.imshow('YOLO Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
