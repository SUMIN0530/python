import cv2
import numpy as np

# 실습 1 ========================================
# 컬러 이미지 생성
#
# 크기 : 300*900
# 각 정사각형 크기 : 300*300

# 1️⃣ 빈 이미지 생성 (300 x 900, 컬러)
img = np.zeros((300, 900, 3), dtype=np.uint8)

# 2️⃣ 영역별 색상 지정 (BGR 순서!)
# 빨강
img[:, 0:300] = [0, 0, 255]

# 초록
img[:, 300:600] = [0, 255, 0]

# 파랑
img[:, 600:900] = [255, 0, 0]

# 3️⃣ 출력
cv2.imshow('RGB Squares', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 실습 2 ========================================
# 채널 조작(응용)
#
# 이미지 생성 후 각 채널 분리, 특정 채널만 0으로 변경
# 원본 이미지 : 노란색RGB(0, 255, 255) 
# B채널을 0으로 만들면?

# 1️⃣ 이미지 생성 (300x300, 컬러)
img = np.zeros((300, 300, 3), dtype=np.uint8)

# 노란색 (OpenCV는 BGR)
img[:] = [0, 255, 255]   # B=0, G=255, R=255

# 2️⃣ 채널 분리
b, g, r = cv2.split(img)
'''
b = img[:,:,0]
g = img[:,:,1]
r = img[:,:,2]
'''

# 3️⃣ B 채널을 0으로 변경
b[:] = 0 # img[:,:,0]

# 4️⃣ 다시 합치기
result = cv2.merge([b, g, r])

# 5️⃣ 출력
cv2.imshow('Original (Yellow)', img)
cv2.imshow('B channel set to 0', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 실습 3  ========================================
# ROI 복사(응용)
#
# 두 개의 이미지 생성, 한 이미지 다른 이미지에 복사
# 이미지 1 : 400*400 파랑 
# 이미지 2 : 200*200 빨강 이미지 2를 1의중앙에 배치

# 1️⃣ 이미지 1 생성: 400x400 파랑 (BGR)
img1 = np.zeros((400, 400, 3), dtype=np.uint8)
img1[:] = [255, 0, 0] # Blue

# 2️⃣ 이미지 2 생성: 200x200 빨강 (BGR)
img2 = np.zeros((200, 200, 3), dtype=np.uint8)
img2[:] = [0, 0, 255] # Red

# 3️⃣ img1 중앙 좌표 계산
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

cx = w1 // 2
cy = h1 // 2

# 4️⃣ img2를 img1 중앙에 배치할 좌표 계산
x1 = cx - w2 // 2
y1 = cy - h2 // 2
x2 = x1 + w2
y2 = y1 + h2

# 5️⃣ 복사 (슬라이싱)
img1[y1:y2, x1:x2] = img2

'''
좌표 계산
y = (h1 - h2) // 2
x = (w1 - w2) // 2

복사
img1[y1:y+h2, x1:x+w2] = img2
'''

# 6️⃣ 출력
cv2.imshow('Result', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 실습 4  ========================================
# 그라데이션 이미지
#
# 수평과 수직이 동시에 적용된 이미지
# 크기 : 300*300
# 왼->오, 위->아래 : 0-255

h_grad = np.tile(np.linspace(0, 255, 300), (300, 1)).astype(np.uint8)
v_grad = np.tile(np.linspace(0, 255, 300), (300, 1)).T.astype(np.uint8)

# 두 그라데이션 합성 (평균)
combined = ((h_grad.astype(np.float32)
            + v_grad.astype(np.float32)) / 2).astype(np.uint8)

cv2.imshow('Horizonttal Gradient', h_grad)
cv2.imshow('Vertical Gradient', v_grad)
cv2.imshow('Combimed Gradient', combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

# =======================이미지, 동영상, 캠===================================
import os
# 실습 1  ========================================
# 이미지 저장 및 비교
#
# 랜덤 이미지 생성 후 파일 크기 비교
# 크기 640*480 컬러 이미지
# 형식 : JPG(품질 50), JPG(품질 95), PNG
# 각 파일 크기 출력 



# 실습 2  ========================================
# 이미지 읽기 실패 처리
#
# 이미지가 있으면 크기 출력
# 없으면 "이미지를 찾을 수 없습니다" 출력
# 기본 이미지(검은색  300*300) 반환 



# 실습 3  ========================================
# 웹 캠 캡처 및 저장
#
# 웹 캠에서 's'키를 누르면 사진을 저장
# 저장 형식 : photo_001.jpg, photo_002.jpg, ...
# 'q'키로 종료 
# 저장 시 "사진 저장됨" 메시지 출력 



# 실습 4  ========================================
# 비디오 재생속도 조절
# 
# 원본 비디오 생성(5초, 30fps, 컬러 프레임)
# 2배속 비디오 생성(프레임 간격 2)
# 0.5배속 비디오 생성(프레임 중복) 
