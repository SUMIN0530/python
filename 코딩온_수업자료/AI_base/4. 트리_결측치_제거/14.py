# 실습
# 실전 데이터 전처리 기법
# 실제 데이터는 항상 지저분하다. (읽기 불편)
# 
# 결측치 처리 전략
# 결측치(Missing Value)
# - 데이터가 비어있는 경우
# - pandas에서는 Nan으로 표시
# - 예 : 나이 정보가 없음, 설문 응답 안함
# 
# 결측치 처리 방법
import pandas as pd
import numpy as np

# 샘플 데이터
df = pd.DataFrame({
    'age': [25, 30, np.nan, 45, np.nan, 35],
    'income': [50000, np.nan, 60000, 80000, 55000, np.nan],
    'city': ['서울', '부산', np.nan, '대구', '서울', '인천']
})

print('=== 결측치 확인 ===')
print(df.isnull().sum())

# 방법 1 : 삭제
# 결측치가 있는 행 전체 삭제
df_drop = df.drop()
print(f'원본 : {len(df)}행 => 삭제 후 : {len(df_drop)}행')

# 문제점 : 데이터가 많이 사라짐

# 방법 2 : 채우기(Imputation) <-- 가장 많이 사용
# 수치형 : 평균/중앙값/최빈값으로 채우기
df['age'].fillan(df['age'].median(), inplace=True) # 중앙값
df['income'].fillan(df['income'].mean(), inplace=True) # 평균

# 범주형 : 최빈값으로 채우기
df['city'].fillan(df['city'].mode(), inplace=True) # 최빈값
print(df.isnull().sum())

# 평균(mean)
# - 정규분포에 가까운 데이터
# - 이상치가 많으면 부적합
# 
# 중앙값(median)
# - 이상치에 강함
# - 대부분 수치형 데이터에 적합
# 
# 최빈값(mode)
# - 범주형 데이터
# - 가장 흔한 값으로 채우기
# 
# 범주형 데이터 인코딩
# 문제 : 머신러닝 모델은 숫자만 이해
# -> '남성', '여성' 이해 불가
# -> 0, 1 이해함
# 
# 방법 1. 수동 매핑(map)
df = pd.DataFrame({
    'sex' : ['male', 'female', 'male', 'female']
}) 

df['sex_encoded'] = df['sex'].map({'male'} ) # 아놔

# 방법 2 : LabelEncoder
from sklearn.preprocessing import LabelEncoder

# 여러 범주가 있는 경우
df = pd.DataFrame({
    'city': ['서울', '부산', np.nan, '대구', '서울', '인천']
})

le = LabelEncoder()
df['city_encoded'] = le.fit_transform(df['city'])

print(df)

# LabelEncoder는 순서를 부여한다
# 서울 = 3, 부산 = 2, 대전 = 1
#
# 문제 : 모델이 '서울' > '부산' > '대구'로 오해 가능
# 트리 기반 모델(랜덤 포레스트 등)은 괜찮음 
# 회귀 모델은 One-Hot Encoding 권장
df = pd.DataFrame({
 'city': ['서울', '부산' '대전']
})

df_encoded = pd.get_dummies(df, columns=['city'], prefix='city')
print(df_encoded)

# 하이퍼파라미터 튜닝
# 하이퍼파라미터
# - 학습 전에 사람이 정해주는 설정값
# - 예 : 랜덤 포레스트의 트리 개수, 최대 깊이
# 
# 파라미터 vs 하이퍼파라미터
# 파라미터 : 모델이 학습으로 찾는 값(가중치)
# 하이퍼파라미터 : 사람이 설정하는 값
#
# 방법 1 : 수동
from sklearn.ensemble import RandomForestClassifier
model1 = RandomForestClassifier(n_estimators=100) 
model2 = RandomForestClassifier(n_estimators=200) 
model3 = RandomForestClassifier(n_estimators=300)
# 너무 오래 걸림

# 방법 2 : GridSearchCV 자동으로 최적값 찾기
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris

# 데이터 로드
iris = load_iris()
x, y = iris.data, iris.target

# 시도할 하이퍼파라미터 조합
param_grid = {
    'n_estimator' : [50, 100, 200],
    'max_depth' : [3, 5, 7],
    'min_sampleㄴ_split' : [2, 5]
}
# 총 3 x 3 x 2 = 18가지 조합

model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    model,                  # 모델
    param_grid,             # 시도할 파라미터
    cv = 5,                 # 5-fold 교차 검증
    scoring='accuracy',     # 평가 지표
    n_jpbs=-1               # 모델 CPU 사용
)

# 뭐가 되게 많이 날아감

# GridSearchCV가 하는 일
# 1. param_grid의 모든 조합 생성(18가지)
# 2. 각 조합마다
# - 5-fold 교차 검증 수행 ===> 총 90번 함 (18 x 5)
# - 평균 점수 계산
# 3. 가장 높은 점수의 조합 선택
# 4. 전체 데이터로 최적 모델 재학습
# 
# 주의사상
# 장점 : 자동으로 최적값 찾기
# 단점 : 시간이 오래 걸림
# 
# 팁
# - 처음엔 넓은 범위로 탐색
# - 그 다음 좁은 범위로 세밀하게 조정  

