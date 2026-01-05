import cv2
import numpy as np
# 뭐가 있음

# Scharr 연산자
# 연산자의 개선 버전, 회전 불변성을 향상시켜 더 정확한 기울기를 계산
# 
# 문제점 
# 특정 각도(0, 45, 90, 135)의 엣지에만 최적화
# -> 다른 각도의 엣지는 정확도가 떨어짐
# Scharr X : Scharr Y :
# 
# 
# 
# 
# 정확도
# 회전 불변성
# 대신 3x3만
# 
# 언제 Scharr을 사용하는가?
# 정밀한 엣지 방향 계산이 필요
# 작은 각도 차이가 중요한 경유
# 회전 불변 특징 추출
# 
# 언제 Sobel을 상요하나
# 일반적인 엣지 검출
# 더 큰 커널이 비료한 경우
# 속도보다 유연성

img = np.random.randint(50, 200, (200, 300), dtype=np.uint8)

# cv2.Scharr(src, ddepth, dx, dy)
scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)

# 크기
magnitude = np.sqrt(scharr_x ** 2 + scharr_y ** 2)
magnitude = np.clip(magnitude, 0, 255).astype(np.uint8) # 두 모델 비교 시 주석처리

# Sobel과 비교
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
spbel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

print(f'Scharr 최대 : {magnitude.max()}')
print(f'Sobel 최대 : {spbel_mag.max():.2f}')

# Laplacian

def laplacian_of_gaussian(img, sigma=1.0, ksize=5):
    '''Log 적용'''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # 가우시안 블러
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)

    # Laplacian
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

    # 정규화
    log_result = cv2.normalize(np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX)

    return log_result.astype(np.uint8)

img = np.random.randint(50, 200, (200, 300, 3), dtype=np.uint8)

sigmas = [0.5, 1.0, 2.0, 3.0]
results = []

for sigma in [0.5, 1.0, 2.0, 3.0]:
    result = laplacian_of_gaussian(img, sigma=sigma)
    print(f'sigma={sigma} : 엣지 픽셀 수 = {np.sum(result > 50)}')

# 시각화
display = np.hstack(results)   # 왜 에러 뜨지
cv2.imshow('display', display)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Canny Edge Detection
# Canny 알고리즘 단계
# 최적의 엣지 검출 알고리즘
# 
# 낮은 오류율 : 실제 엣지만 검출, 가짜 엣지 최소화
# 정확한 위치 : 엣지의 정확한 위치 찾기
# 단일 응밥 : 하나의 엣지에 하나의 응답만
# 
# 5단계 알고리즘
# 1. 가우시안 블러(Gaussian Blur)
# - 목적 : 노이즈 제거
# - 이유 : 노이즈가 있으면 엣지로 오판
# 
# 2. 그라디언트 계산(Gradent Calculation)
# - 목적 : 엣지 크기와 방향 계산
# - 방법 : Sovel로 X, Y 방향 기울기
# 
# 3. 비최대 억제(Non-Maximum Suppression, NMS)
# - 목적 : 얇은 엣지 생성
# - 방법 : 엣지 방향에서 최대값만 유지
# 
# 4. 이중 임계값
# - 목적 : 엣지 분류
# - 결과
#   - 강한 엣지 : high threshold 이상 (확실한 엣지)
#   - 양한 엣지 : low - high threshold (애매한 엣지)
#   - 비엣지 : low threshold 이하
# 
# 5. 히스테리시스(Hysteresis / Edge Trackig)
# - 목적 : 연결된 엣지만 유지
# - 방법 : 강한 엣지에 연결된 약한 엣지만 최종 엣지로 채택
# 
# Sobel : 두꺼운 엣지, 노이즈 많음
# Laplacian : 노이즈 민감
# Canny : 얇고 정확하며 연속된 엣지

# cv2.Canny(src, threshold1, threshold2, apertureSize, L2gradient)
# threshold1 : low threshold
# threshold2 : high threshold
# apertureSize : Sobel 커널 크기(기본 3)
# L2gradient : True 이면 L2 norm 사용 
edges = cv2.Canny(img, 50, 150)

# 다양한 임계값
edges_low = cv2.Canny(img, 30, 100)
edges_high = cv2.Canny(img, 100, 200)

# cv2.imshow('edges', edges)
# cv2.imshow('edges_low', edges_low)
# cv2.imshow('edges_high', edges_high)

# 자동 임계값
def auto_canny(img, sigma=0.33):
    '''자동 임계값 Canny'''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # 중간값 계산
    median = np.median(gray)

    # 임계값 설정
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))

    edges = cv2.Canny(gray, lower, upper)

    return edges, lower, upper

# 테스트
edges, low, high = auto_canny(img)
print(f'')


cv2.waitKey(0)
cv2.destroyAllWindows()

 