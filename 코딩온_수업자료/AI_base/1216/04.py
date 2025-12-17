# Scikit-learn
# Python 머신러닝 라이브러리의 표준
# 
# 특징
# 쉬운 API(fit, predict, transform)
# 다양한 알고리즘 제공
# 전처리, 평가 도구 포함
# 풍부한 문서와 예제
# 무료, 오픈 소스

# 제공 기능
# 분류(Classification)
# LogisticRegression, SVC 등등
# 
# 회귀(Regression)
# LinearRegression, Ridge 등등
# 
# 군집화
# 
# 차원 축소
# 
# 전처리
# 
# 모델 선택 
'''
from sklearn.어디서든 import 모델

# 1. 모델 생성
model = 모델(하이퍼파라미터)

# 2. 학습
model.fit(x_train, y_traini)

# 3. 예측
predictions = model.predict(x_test)

# 4. 평가
score = model.score(x_test, y_test)
'''
from sklearn.datasets import load_iris

iris = load_iris()
print(f'특성:', iris.feature_names)
print(f'타겟:', iris.target_names)

x = iris.data
y = iris. target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
print(f'훈련: {x_train.shape} 테스트: {x_test.shape}')

from sklearn.neighbors import KNeighborsClassifier

# 1. 모델 생성
model = KNeighborsClassifier(n_neighbors=3)

# 2. fit 학습
model.fit(x_train, y_train)

# 3. 예측
y_pred = model.predict(x_test)
print('예측 결과 : ', y_pred[:10])
print('실제 결과 : ', y_test[:10])

# 모델 평가
from sklearn.metrics import accuracy_score, classification_report

# 정확도
accracy = accuracy_score(y_test, y_pred)
print(f'정확도 : {accracy:.2%}')

# 간단히
print(f'정확도 : {model.score(x_test, y_test):.2%}')

# 상세 리포트
print(classification_report(
    y_test, 
    y_pred, 
    target_names=iris.target_names
))

from sklearn.linear_model import LogisticRegression
# model = LogisticRegression(max_iter=200)
# # model = LogisticRegression() # 괄호 안채워도 정확도 동일
# model.fit(x_train, y_train)
# print(f'로지스틱 회귀 정확도 : {model.score(x_test, y_test):.2%}')

from sklearn.tree import DecisionTreeClassifier
# model = DecisionTreeClassifier()
# model.fit(x_train, y_train)
# print(f'결정 트리 정확도 : {model.score(x_test, y_test):.2%}')

from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier()
# model.fit(x_train, y_train)
# print(f'랜덤 포레스트 정확도 : {model.score(x_test, y_test):.2%}')

# 함축
models = {
    'Logistic' : LogisticRegression(max_iter=200),
    'Dec' : DecisionTreeClassifier(),
    'Random' : RandomForestClassifier()
}

for name, model in models.items():
    model.fit(x_train, y_train)
    print(f'{name} 정확도 : {model.score(x_test, y_test):.2%}')

# 데이터 전처리
# 스케일링 중요성
# 특성별 스케일이 다르면 문제!!
print('원본 데이터 범위 : ')
print(f'특성1 : {x[:,0].min():.1f} ~ {x[:,0].max():.1f}')
print(f'특성2 : {x[:,0].min():.1f} ~ {x[:,0].max():.1f}')
# 범위가 다르면 일부 특성이 과도한 영향 

from sklearn.preprocessing import StandardScaler

# 스케일링 생성 및 학습
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test) # fit은 하지 않음!

print('스케일링 후')
print(f'평균 : {x_train_scaled.mean(axis=0)}')
print(f'표준편차 : {x_train_scaled.std(axis=0)}')

from sklearn.svm import SVC

# 스케일링 없이
model = SVC()
model.fit(x_train, y_train)
print(f'스케일링 전 : {model.score(x_test, y_test):.2%}')

# 스케일링 후
model = SVC()
model.fit(x_train_scaled, y_train)
print(f'스케일링 전 : {model.score(x_test_scaled, y_test):.2%}')
print()

# ====================== 실습 ==========================
from sklearn.datasets import load_wine
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
print('============= 실습 ===============')
wine = load_wine()
x, y = wine.data, wine.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 3가지 이상의 모델 비교
models = {
    'Knn' : KNeighborsClassifier(n_neighbors=3),
    'Logistic' : LogisticRegression(),
    'Dec' : DecisionTreeClassifier(),
    'Random' : RandomForestClassifier(),
    'svc' : SVC()
}
'''
# 훈련/테스트 데이터 분할
x_train, x_test = x[:15], x[15:]
y_train, y_test = y[:15], y[15:]'''

print('스케일링 전')
for name, model in models.items():
    model.fit(x_train, y_train)
    print(f'{name} : {model.score(x_test, y_test):.2%}')

# 스케일링 생성 및 학습
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test) # fit은 하지 않음!

print('스케일링 후')
for name, model in models.items():
    model.fit(x_train, y_train)
    print(f'{name} : {model.score(x_test, y_test):.2%}')
# 성능이 다르게 떠야되는데 나는 왜 같이 뜨는가??

