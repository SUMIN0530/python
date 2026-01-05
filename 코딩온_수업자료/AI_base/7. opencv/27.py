# 좌표계와 인덱싱
# 죄표계의 이해

# OpenCV 좌표계
# (0, 0) -------------------------> x (width)
#   | 
#   | 
#   | 
#   | 
#   |           이미지 영역 
#   | 
#   | 
#   | 
#   | 
#   ▽
#   y (height)

# NumPy 인덱싱
# - img[y, x] 또는 img[row, col]
# - img [y1:y2, x1:x2]

# 사라져써...

# 픽셀 접근
# 이미지 생성
import numpy as np
import cv2
img = np.zeros((100, 200, 3), dtype=np.uint8)

# 단일 픽셀 접근 (읽기)
pixel = img[50, 100] # [y, x] = [row, col]
print(f'픽셀 값 (BGR) : {pixel}')

# 단일 픽셀 설정 (쓰기)
img[50, 100] = [200, 0, 0] 
print(f'픽셀 값 (BGR) : {pixel}')

# 영역 접근
roi = img[20:80, 50:150] # [y1:y2, x1:x2]
print(f' ROI 크기 : {roi.shape}')

# 영역 설정
roi = img[20:80, 50:150] = [0, 255, 0]

# ROI(Region of Interest)
imp = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)

# ROI 추출
x, y, w, h = 10, 50, 200, 150
roi = img[y:y+h, x:x+w]

# roi 복사 (독리적인 복사본)
roi_copy = roi.copy()

# ROI 수정 (원본에 영향)
roi[:] = [255, 0, 0]

# ROI 붙여넣기
target = np.zeros((300, 400, 4), dtype=np.uint8)
target[y:y+h, x:x+w] = roi_copy

cv2.imshow('target', target)