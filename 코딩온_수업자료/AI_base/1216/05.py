# 선형 회귀의 원리
# 선형회귀(Linear Regression)

# 데이터를 가장 잘 설명하는 직선을 찾는 것

# 예측 : y = wx + b
# - w : 기울기 (가중치, weight)
# - b : 절편 (편향, bias)
# - x : 입력 
# - y : 출력 (예측값)
#
# 실생활 예시
# 공부 시간(x) -> 시험 점수 (y)
# 집 면적(x) -> 집 가격(y)
# 광고비(x) -> 매출(y)
# 온도(x) -> 아이스크림 판매량(y)

# 손실 함수
# 오차 측정
# 예측이 얼마나 틀렸는지 측정 

# 오차값 = 실제값 - 예측값 ==> y - ŷ

# 문제 : 오차가 양수/음수 섞여 있으면 상쇄됨
# 해결 : 오차를 제곱!

# MSE(Mean Squared Error)
# MSE = 평균((실제값 - 예측값)²) ==> (1/n) x ∑(yi - ŷi)²

# 특징
# - 오차가 클수록 큰 페널티 (제곱효과)
# - 항상 양수
# - 미분 가능 (경사하강법에 유리)
# 
# 수식으로 표현
# 예측 : ŷ = wx + b
# 
# 손실 함수
# L(w, b) =  (1/n) x ∑(yi - (wxi + b))²
#
# 목표 : L을 최소화 하는 w, b 찾기

# 최소 제곱법
# 정규 방정식
# 손실 함수를 w, b로 미분하고 0으로 놓으면 최적의 w, b를 구하는 공식 유도 가능
# w = ∑ 

import numpy as np

# 데이터
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# 평균
x_mean = np.mean(x)
y_mean = np.mean(y)

# 최소 제곱법
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean) ** 2)

w = numerator / denominator
b = y_mean - w * x_mean

print(f'기울기(w) : {w:.4f}')
print(f'절편(b) : {b:.4f}')
print(f'예측식 y : {w:.2f}x + {b:.2f}')