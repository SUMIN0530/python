# 과적합

# 과적합(Overfitting)
# 훈련 데이터를 너무 잘 외워버림

# 특징
# - 훈련데이터 성능 : 매우 높음
# - 테스트 데이터 성능 : 낮음
# - '시험 문제는 잘 맞추는데 응용이 안됨'

# 시험 공부 비교
# - 기출문제 암기
# - 기출 100점
# - 새로운 문제 50점
# - '이거 안배웠는데요?'
# 
# 좋은 학생
# - 개념 이해
# - 기출 90점
# - 새로운 문제 85점
# - 응용력 있음!

# 과소적합(Underfitting)
# 데이터의 패턴을 충분히 학습하지 못함
# 
# 특징
# - 훈련 데이터 성능 : 낮음
# - 테스트 데이터 성능 : 낮음
# - 기본도 모르는 상태
# 
# 과소적합 학생
# - 공부 거의 X
# - 기출 30점
# - 새로운 문제 30점
# - 다 모르는 문제

# 과적합의 원인과 해결
# 원인
# 1. 모델이 너무 복잡함    ->       
# - 파라미터가 너무 많음
# - 신경망 층이 너무 깊음
# 
# 2. 데이터가 너무 적음    ->       더 만은 데이터 수집
# - 학습할 예시 부족               - 가장 효과적인 방법!
# - 모델이 데이터를 외움           - 데이터 증강 (Data Augementation)
# 
# 3. 노이즈까지 학습       ->       정규화(Regularization)
# - 의미 없는 패턴까지 학습          - L1, L2 정규화
#                                   - Dropout(딥러닝)
#  
# 4. 학습을 너무 오래함    ->       조기 종료 (Early Stopping))
# - 훈련 데이터에 점점 맞춰감        - 검증 성능이 떨어지기

# 과소적합 원인과 결과
# 1. 모델이 너무 단순함     ->      더 복합한 
# - 복잡한 패턴 표현 못함 
# 
# 2. 특성(Feature)이 부족   ->      특성 추가
# - 중요한 정보가 없음               - 새로운 특성 만들기
#                                   - 다항 특성 추가
# 
# 3. 학습 부족              ->      더 오래 학습
# - 에폭 수가 너무 적음              - 에폭 수 증가
# 
# 정규화 줄이기
# - 정규화가 너무 강하면 과소적합

# 과적합 예시
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 

matplotlib.rcParams['font.family'] = 'Malgun Gothic'

# 데이터 생성
np.random.seed(42)
x = np.linspace(0, 1, 20).reshape(-1, 1)
y = np.sin(2 * np.pi * x).ravel() + np.random.randn(20) * 0.3

print(f'x: {x}')
print()
print(f'y: {y}')

# 훈련/테스트 데이터 분할
x_train, x_test = x[:15], x[15:]
y_train, y_test = y[:15], y[15:]

# 다양한 복잡도로 모델 학습
degrees = [1, 4, 15]
plt.figure(figsize=(15, 4))

for i, degree in enumerate(degrees):
    plt.subplot(1, 3, i+1)

    # 다양한 특성 생성
    poly = PolynomialFeatures(degree=degree)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.transform(x_test)

    # 모델 학습
    model = LinearRegression()
    model.fit(x_train_poly, y_train)

    # 예측
    y_train_pred = model.predict(x_train_poly)
    y_test_pred = model.predict(x_test_poly)

    # 성능 계산
    train_error = mean_squared_error(y_train, y_train_pred)
    test_error = mean_squared_error(y_test, y_test_pred)
    
    # 시각화
    x_plot = np.linspace(0, 1, 100).reshape(-1, 1)
    x_plot_poly = poly.transform(x_plot)
    y_plot = model.predict(x_plot_poly)

    plt.scatter(x_train, y_train, label='훈련')
    plt.scatter(x_test, y_test, label='테스트', marker='s')
    plt.plot(x_plot, y_plot, 'r-', label='예측')
    plt.title(f'차수={degree}/n훈련 MSE={train_error:.3f}, 테스트 MSE={test_error:.3f}')
    plt.legend()

plt.tight_layout()
plt.show()

