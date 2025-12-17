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
print(f'예측식 y : {w:.2f}x + {b:.2f}') # ---값이 달라

# Scikit-liarn으로 선형 회귀
from sklearn.linear_model import LinearRegression

x = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 2])

# 모델 생성 및 학습
model = LinearRegression()
model.fit(x, y)

# 파라미터 확인
print(f'기울기 : {model.coef_[0]:.4f}')
print(f'절편 : {model.intercept_:.4f}')

# 예측
y_pred = model.predict(x)
print(f'예측값 : {y_pred}')

'''# 시각화
import matplotlib.pyplot  as plt

# 폰트 꺠짐 해결
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['ax'] # 놓침

plt.figure(figsize=(10, 6))

# 데이터 점
plt.scatter(x, y, color='blue', s=100, label='실제데이터')

# 회귀선
plt.plot(x, y_pred, color='red', linewidth=2, label='회귀선')

# 오차 시각화
for i in range(len(x)) :
    plt.plot([x[i],x[i]], [y[i],y_pred[i]], 'g--', alpha=0.5 )


plt.xlabel('x')
plt.ylabel('y')
plt.title('선형 회귀')
plt.legend()
plt.grid(True)
plt.show()'''
print()

# 선형 회귀 가정
# 주요 가정
# 1. 선형성
# - x와 y 사이에 선형 관계 존재
# 
# 2. 독립성
# - 오차들이 서로 독립적
# 
# 3. 등분산성
# - 오차의 분산이 일정
# 
# 4. 정규성
# - 오차가 정규분포를 따름

# 가정 위반시
# 선형성 위반
# - 데이터가 곡선 패턴
# -> 다향 외귀, 비선형 모델
# 
# 이상치 존재
# - 극단적인 값, 결과 왜곡
# -> 이상치 저거 또는 robot 효과?? 

# ============== 실습 =================
print('============= 실습 =============')
# 데이터
ad = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1) # 왜 하는거라고?
sell = np.array([3, 5, 6, 8, 11, 13, 14, 16, 17, 20])

# 모델 생성 및 학습
model = LinearRegression()
model.fit(ad, sell)

print(f'기울기 : {model.coef_[0]:.4f}') # [0] : 배열을 의미 배열 첫 번째(0번째)
print(f'절편 : {model.intercept_:.4f}')
print(f'예측식 : 매출 = {model.coef_[0]:.2f} x 광고비 + {model.intercept_:.2f}')

# 예측 - 광고비 1,500(15)만원 이상일 때 예상 매출액
new_ad = np.array([[15]])
pred_sell = model.predict(new_ad)
print(f'1,500만원 이상 매출액 : {pred_sell[0]:.2f}천만원')

from sklearn.metrics import r2_score
r = model.predict(ad)
r2 = r2_score(sell, r)
print(f'R² : {r2}')

# R² Score 1에 가까우면
# 모델이 데이터를 잘 설명한다.
# 1 : 완벽한 예측(오차 X)
# 0 : 평균으로 예측하는 것과 동일
# < 0 :  평균보다 못한 예측
# 
# R² = 0.95 => 정확도 95%

# 머신러닝 지표는 몯레을 평가하기 위한 내부 언어
# 업무를 움직이는 언어 x
# 
# 한식 뷔페 발주
# 요일, 메뉴, 날씨
#
# 나쁜 예
# R² = 0.81, MSE = 23
# 다음주 n요일 예측방문 138명입니다.
# 
# 핵심 메시지(한 문장)
# 다음주 화요일은 평균보다 손님이 15~20% 많을 가능성이 높다
# 재료를 기준 대비 10% 추가 발주가 안전하다.

# 영양사
# 고기류가 메뉴에 있을 때 방문객 평균 18% 증가하는 패턴 반복
# 이번 주 메뉴 구성 기준으로 보면 고기 반찬 기준량을 1.2배 준비하는 것이 적절
# 
# 조리 담당자
# 피크 시간대(12:10 ~ 12:40)에 집중될 확률이 높음
# 이 시간 전에 주력 메뉴 1차 준비를 끝내는 것 추천

# 태양광 발전소
# 나쁜 예
# 태양광 발전량 예측 모델의 MAE 3.2mwh이고 정확도 87%이다.
# 
# 운영 담당자
# 내일 오후 1시 ~ 4시는 발전량이 평소 대비 30% 낮을 가능성이 높아
# ESS 방전 또는 외부 전력 보완이 필요하다. 