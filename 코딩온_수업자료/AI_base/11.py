# 분류 문제와 로지스틱 회귀

# 회귀(Regression)
# 연속적인 숫자 예측
# 예 : 집 값, 온도, 매출
# 
# 분류(Classification)
# 범주(카테고리) 예측
# 예 : 스팸/정상, 암/정상, 개/고양이 

# 분류의 종류
# 이진 분류(Binary)
# - 두 개의 클래스
# - 예 : 합격/불합격, 양성/음성
# 
# 다중 클래스 분류(Multi-class)
# - 세 개 이상의 클래스
# - 예 : 개/고양이/새, 등급 A/B/C
# 
# 다중 레이블 분류(Multi-label)
# - 여러 레이블 동시에 가능
# - 예 : 영화 장르 (액션+로맨스+코미디)

# 선형 회귀를 안 쓰는 이유
# 문제 : 합격(1)/불합격(0) 예측
# 
# 선형 회귀 사용
# y = wx + b
# 
# 문제점
# 1. 출력이 0~1 범위를 벗어남 (예 : -0.5, 1.7)
# 2. 이상치에 민감
# 3. 확률로 해석 불가

# 로지스틱 회귀
# 선형 회귀 출력을 0~1 범위로 변환
# 선형 : z = wx + b (범위 : -∞ ~ +∞)
# ↓
# 시그모이드 함수 적용
# ↓
# 출력 : σ(z) = 
# 출력값 = 확률로 계산 가능 

import numpy as np
import matplotlib.pyplot as plt


# 한글 폰트 설정 추가
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
# plt.rcParams['font.family'] = 'AppleGothic'  # Mac
plt.rcParams['axes.unicode_minus'] = False

def sigmod(z):
    return 1 / (1 + np.exp(-z))

# 시각화
z = np.linspace(-10, 10, 100)
y = sigmod(z)

plt.figure(figsize=(10, 6))
plt.plot(z, y, 'b-', linewidth=2)
plt.axhline(y=0.5, color='r', linestyle='--', label='경계 (0.5)')
plt.axvline(x=0, color='gray', linestyle='-' ,alpha=0.3)
plt.xlabel('z (wx + b)')
plt.ylabel('σ(z)')
plt.title('시그모이드 함수')
plt.legend()
plt.grid(True)
plt.show()

# 로지스틱 회귀 수식
# P(y=1|x) = σ(z) = 1 / (1 + e^(-z))
# P(y=1|x): x가 주어졌을 때 y = 1일 확률

# 손실 함수(Binary Cross Entropy)
# MSE 사용시 문제
# - 볼록하지 않은 형태 -> 경사하강법 어려움
# 
# 대안 : Log Loss (Cross Entropy)
# 

# 직관
# - y = 1일 때 : -log(ŷ) -> ŷ이 1에 가까울수록 손실 작음  
# - y = 0일 때 : -log(1-ŷ) -> ŷ이 0에 가까울수록 손실 작음

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 데이터 생성
np.random.seed(42)
x = np.random.randn(200,2)
y = (x[:, 0] + x[:, 1] > 0).astype(int)

print(x)
print(y)

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# 모델 학습
model = LogisticRegression()
model.fit(x_train, y_train)

# 예측
y_pred = model.predict(x_test)

# 평가
print(f'정확도 : {accuracy_score(y_test, y_pred):.2f}')
print(classification_report(y_test, y_pred))

# 확률 예측
# 클래스 확률
proba = model.predict_proba(x_test)
print('확률 예측 (처음 5개)')
print(proba[:5])

