# Open CV (Open Source Computer Vision Library)
# 실시간 컴퓨터 비전을 위한 오픈소스 라이브러리
# 
# 특징 2500+ 최적화 된 알고리즘
# C++, Python, Java, MATLAB 지원
# Windows, Linux, maxOS, Android, IOS 지원
# GPU 가속 지원 (CUDA, OpenCL)

# 이미지 처리
# - 필터링, 변환, 색상 처리
# - 형태학적 연산
# - 엣지 검출
# 
# 객체 탐지/인식
# - 얼굴 검출
# - 물체 추적
# - 특징점 매칭
# 
# 비디오 분석
# - 모션 검출
# - 배경 제거
# - 광학 흐름
# 
# 딥러닝 추론 
# - DNN 모듈
# - 사전 학습 모델 로그
# - ONNX, TensorFlow, PyTorch 모델 지원
# 
# 산업 응용
# - 자율 주행
# - 의료 영상
# - 보안/감시
# - AR/VR 

import cv2
import numpy as np
print(f'OpenCV 버전 : {cv2.__version__}')

# OpenCV에서 이미지 = Numpy 배열

# 흑백 이미지 : (높이, 너비)
gray_img = np.zeros((100, 200), dtype=np.uint8) # uint8 : 8비트의 정수만 뽑아오겠다 / int와 동일하나 두 범위가 다름.
print(f'흑백 이미지 : {gray_img.shape}')        # C++ 에서 unsigned char와 동일 0 ~ 255(가장 일반적인 수치)

# 컬러 이미지 : (높이, 너비, 채널)
color_img = np.zeros((100, 200, 3), dtype=np.uint8)
print(f'컬러 이미지 : {color_img.shape}')

# OpenCV : RGB X => BGR (Blue, Green, Red)
# 빨간색 생성
bgr_red = np.zeros((100, 100, 3), dtype=np.uint8)
bgr_red[:,:,2] = 255 # R 채널 (인덱스 2)

# BGR -> RGB 변환
rgb_red = cv2.cvtColor(bgr_red, cv2.COLOR_BGR2RGB)
print(f'BGR 순서 : {bgr_red}')
print(f'BGR 순서 : {rgb_red}')

# 컬러 이미지 생성
img = np.zeros((300, 300, 3), dtype=np.uint8)
img[:,:,0] = 100 # Blue
img[:,:,1] = 150 # Green
img[:,:,2] = 200 # Red

# 채널 분리
b, g, r = cv2.split(img)
print(f'Blue 채널 : {b.shape}')
print(f'Green 채널 : {g.shape}')
print(f'Red 채널 : {r.shape}')

# 채널 병합
merged = cv2.merge([b, g, r])
print(f'병합 결과 : {merged}')

# 개별 채널 접근 (더 효율적)
blue_channel = img[:,:,0]
green_channel = img[:,:,1]
red_channel = img[:,:,2]

# 이미지 생성
# 검은색 이미지 생성
black = np.zeros((200, 300, 3), dtype=np.uint8)

# 흰색 이미지
white = np.ones((300, 300, 3), dtype=np.uint8) * 255

# 특정 색상 이미지
blue = np.zeros((200, 300, 3), dtype=np.uint8)
blue[:,:] = (255, 0, 0) # BGR

Green = np.zeros((200, 300, 3), dtype=np.uint8)
Green[:,:] = (0, 255, 0) # BGR

Red = np.zeros((200, 300, 3), dtype=np.uint8)
Red[:,:] = (0, 0, 255) # BGR

# 랜덤 이미지
random_img = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)

# 그라데이션 이미지
# np.linspace(0, 255, w) : 0 ~ 255까지 w개로 균등하게 분배
# np.tile(..., (h, 1)) : ... 줄을 h번 복사해서 아래로 쌓기
h, w = 200, 300
gradient_h = np.tile(np.linspace(0, 255, w), (h, 1)).astype(np.uint8)

# 수직 그라데이션
gradient_w = np.tile(np.linspace(0, 255, h), (w, 1)).T.astype(np.uint8)

# 컬러 그라데이션
h, w = 200, 300
gradient_color = np.zeros((h, w, 3), dtype=np.uint8)
gradient_color[:,:,0] = gradient_h # Blue
gradient_color[:,:,2] = gradient_w # Red

# 체크보드--------------------------------------------
h, w = 5, 5
square = 1 # 한 칸씩 띄워 만들겠다.

y = np.arange(h) // square # [0, 1, 2, 3, 4]
x = np.arange(w) // square # [0, 1, 2, 3, 4]

board = (y[:, None] + x[None, :]) % 2
#         y 부분       x 부분         
#       [               [
#          [0],            [0],
#          [1],            [1],
#          [2],            [2],
#          [3],            [3],
#          [4]             [4]    
#        ]               ]

[
    [ ], [1], [ ], [1], [ ],
    [1], [ ], [1], [ ], [1],
    [ ], [1], [ ], [1], [ ],
    [1], [ ], [1], [ ], [1],
    [ ], [1], [ ], [1], [ ]
]

# Numpy 출력 생략 끄기
# np.set_printoptions(threshold=np.inf)

print(board)

checkerboard = (board * 255).astype(np.uint8)
cv2.imshow('window_name', checkerboard)


# 체크보드------------------------------------------
checker = np.zeros((300, 300), dtype=np.uint8)

checker[::2, ::2] = 255
checker[1::2, 1::2] = 255

cv2.imshow("checker", checker)
cv2.waitKey(0)

# 체크무늬 확대
h, w = 200, 300
block = 40   # 🔥 이 값이 체크무늬 크기

y, x = np.indices((h, w))
mask = ((x // block + y // block) % 2) == 0

checker = np.zeros((h, w), dtype=np.uint8)
checker[mask] = 255

cv2.imshow("checker", checker)
cv2.waitKey(0)
# ----------------------------------------------------


# 윈도우 생성 및 표시
# cv2.namedWindow('My window', cv2.WINDOW_NORMAL) # 창 크기 조절 가능
cv2.imshow('window_name', gradient_color)

# 키 입력 대기
key = cv2.waitKey(0)

# 모든 윈도우 닫기
cv2.destroyAllWindows()