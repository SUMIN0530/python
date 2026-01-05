import cv2
import numpy as np

# 실습 1. 커스텀 커널 만들기
# 3x3 커널을 만들어 이미지에 적용
# - 중앙 가중치가 높은 커널 생성
# - 랜덤 이미지에 적용
# - 원본과 결과 비교

# 1️⃣ 랜덤 이미지 생성 (300x300, grayscale)
img = np.random.randint(50, 200, (300, 300, 3), dtype=np.uint8)

# 2️⃣ 3x3 커스텀 커널 (중앙 가중치 큼)
kernel = np.array([
    [1,  2, 1],
    [2, 10, 2],
    [1,  2, 1]
], dtype=np.float32)

# 커널 정규화 (밝기 유지)
kernel /= kernel.sum()

# 3️⃣ 커널 적용 (컨볼루션)
filtered = cv2.filter2D(img, -1, kernel)

# 4️⃣ 비교용 이미지 생성 (가로로 붙이기)
comparison = np.hstack((img, filtered))

# 5️⃣ 출력
cv2.imshow('Original (Left) | Filtered (Right)', comparison)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ===========================================================
# 실습 2. 블러 효과 비교
# - 같은 이미지 다른 블러(평균, 가우디안, 미디언)
# - 3x3 그리드로 결과 비교

# 2️⃣ 블러 커널 크기
ksizes = [3, 7, 15]

# 3️⃣ 결과 저장용 리스트
results = []

# 평균 블러
for k in ksizes:
    results.append(cv2.blur(img, (k, k)))

# 가우시안 블러
for k in ksizes:
    results.append(cv2.GaussianBlur(img, (k, k), 0))

# 미디안 블러
for k in ksizes:
    results.append(cv2.medianBlur(img, k))

# 4️⃣ 3x3 그리드 구성
row1 = np.hstack(results[0:3])  # 평균 3,7,15
row2 = np.hstack(results[3:6])  # 가우시안 3,7,15
row3 = np.hstack(results[6:9])  # 미디안 3,7,15

grid = np.vstack((row1, row2, row3))

# 5️⃣ 출력
cv2.imshow('Mean | Gaussian | Median  (3x3 Grid)', grid)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 실습 3. 샤프닝 강도 조절
# - 기본 샤프닝 커넬 사용
# - 강도를 약, 중, 강으로 조절
# - 슬라이더로 실시간 조절

# 1️⃣ 이미지 로드 (흑백)
img = cv2.imread('./AI_base/7. opencv/bird.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError("이미지를 불러올 수 없습니다.")

# 2️⃣ 기본 샤프닝 커널
base_kernel = np.array([
    [0, -1,  0],
    [-1, 5, -1],
    [0, -1,  0]
], dtype=np.float32)

# 3️⃣ 콜백 함수 (트랙바용)
def on_trackbar(value):
    # value : 0 ~ 100
    alpha = value / 50.0   # 강도 조절 (0 ~ 2)

    # 샤프닝 커널 강도 조절
    kernel = np.array([
        [0, -1,  0],
        [-1, 5 + alpha, -1],
        [0, -1,  0]
    ], dtype=np.float32)

    sharpened = cv2.filter2D(img, -1, kernel)
    cv2.imshow('Sharpening', sharpened)

# 4️⃣ 윈도우 & 트랙바 생성
cv2.namedWindow('Sharpening')
cv2.createTrackbar(
    'Strength',      # 트랙바 이름
    'Sharpening',    # 연결할 윈도우
    50,              # 초기값 (중간)
    100,             # 최대값
    on_trackbar
)

# 초기 화면 표시
on_trackbar(50)

cv2.waitKey(0)
cv2.destroyAllWindows()

# ====================================================
# 35.py
# 실습 1. Sobel 필터 비교
# - 수평(x), 수직(y), 크기(magnitude) 각각 계산
# - 3개를 나란히 표시
# - 어떤 엣지가 강조되는지 관찰 

img = cv2.imread('./AI_base/7. opencv/Cameraman.png')

# 2️⃣ Sobel 필터 적용
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # 수직 경계
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # 수평 경계

# # 절댓값 + uint8 변환
# sobel_x_abs = cv2.convertScaleAbs(sobel_x)
# sobel_y_abs = cv2.convertScaleAbs(sobel_y)

# 3️⃣ Magnitude 계산
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
magnitude = cv2.convertScaleAbs(magnitude)

# 4️⃣ 나란히 붙이기
# combined = np.hstack((sobel_x_abs, sobel_y_abs, magnitude))
combined = np.hstack((sobel_x, sobel_y, magnitude))

# 5️⃣ 출력
cv2.imshow('Sobel X | Sobel Y | Magnitude', combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 실습 2. Canny 파라미터 조정
# Canny 엣지 임계값을 조절하며 결과 관찰
# - 낮은 임계값 (50, 100) 
# - 중간 임계값 (100, 200) 
# - 높은 임계값 (150, 300) 
# - 트렉바로 실시간 조절

# Canny 엣지
low = cv2.Canny(img, 50, 100)
mid = cv2.Canny(img, 100, 200)
high = cv2.Canny(img, 150, 300)

# 나란히 배치
combined = np.hstack((low, mid, high))

cv2.imshow('Low (50,100) | Mid (100,200) | High (150,300)', combined)
cv2.waitKey(0)
cv2.destroyAllWindows()