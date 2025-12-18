# 다중 선형 회귀
# 단순 선형 회귀
# y = wx + b
# - 입력(x)이 1개
# - 예 : 면적 -> 집값 

# 다중 선형 회귀
# y = w1x1 + w2x2 + ... + wnxn + b
# - 입력(x)이 여러개
# - 예 : 면적, 방 수, 역까지 거리 -> 집값
# 
# 행렬 표현
# 벡터 표기 :
# y= wx + b = wTx + b
# 
# 행렬 표기(여러 데이터)
# Y = XW + b
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2

# 집값 테이터 생성
np.random.seed(42)
n_sample = '이게 몰까'
data = {
    '면적'
}

# 모델 학습
# 특성과 타겟 분리

print()

# ======================= 실습 ==========================
print('=============== 실습 ================')
from sklearn.datasets import fetch_california_housing

# 1. 데이터 준비
housing = fetch_california_housing()

x = housing.data
y = housing.target
feature_names = housing.feature_names

# 2. 데이터 확인
df = pd.DataFrame(x, columns=feature_names)
df['Targer'] = y

print('데이터 크기 : ', x.shape)
print(df.describe())

from sklearn.model_selection import train_test_split 
# 이게 스케일링 하려고 필요한건가? 회귀분석에서 쓸 자료 노누는거 아닌가??
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# n. 스케일링
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 3. 학습/테스트 분할 -> 어떻게 하는건데 할 때 마다 형태가 바뀌는거 같음... 물론 내가 제대로 이해를 못해서 그렇게 보이는 걸 수도 있지.

# 4. 다중 선형 회귀 모델 학습
model = LinearRegression()
model.fit()
# 5. R² Score 확인
from sklearn.metrics import r2_score
r = model.predict(x)
r2 = r2_score(y, r)
print(f'R² : {r2}')

# 6. 각 특성의 중요도 분석 -> 각 특성도 생성해서 만들어야 되는 거겠지?? 아무래도 그렇지 않을까?