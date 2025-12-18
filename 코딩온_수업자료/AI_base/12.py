# 결정 트리
# 스무고개 게임!

# 과일 맞추기
# Q1. 빨간색인가?
# => yes -> Q2. 작나?
#               => yes -> A1. 체리 
#               => no -> A2. 사과
# => no -> Q2. 노란색인가? 

# 컴퓨터가 하는 일은 어떤 질문을 어떤 순서로 할지 데이터에서 자동으로 찾는 것

# 트리구조 용어 
#           [루트 노드]             <- 맨 처음 질문
#            빨간색인가?
#           /       \
#          Yes      No
#          /         \
#     [내부 노드]  [내부 노드]
#        작나?      노란색인가?     <- 중간 질문들
#       /    \        /    \
#     Yes     No    Yes     No
#      |      |      |      | 
#   [리프]  [리프] [리프]  [리프]   <- 최종 결정(잎사귀)
#    승인    검토   승인    거절

# 깊이(Depth) : 루트에서 리프까지 거치는 질문 수 

# 좋은 질문 vs 나쁜 질문
# 핵심은 잘 나누는 질문을 찾는 것

# 데이터 : 사과 10개, 오렌지 10개를 구분
# 나쁜 질문 : 무게가 100g 이상인가?
# 좋은 질문 : 빨간색인가?

# 각 그룹이 순수해지도록 나누기

# 순수도
# 지니 불순도(Gini Impurity) 
# Gini = 1 - (각 클래스 비율의 제곱의 합)
#  
# 이진 분류 범위 : 0 ~ 0.5까지
# 다중 클래스 : 0.6666
#
# 숫자가 클수록 불순도 있음. 
# 
# 직관적 의미 : 랜덤을 뽑아서 랜덤으로 라벨 붙이면 틀릴 확률
# 예1 : 상자에 [사과 10개]
# - 사과 비율 = 10/10 = 1.0
# - Gini = 1 - (1.0)² = 1 - 1 = 0
# - 완전히 순수! (틀릴 일 없음)
# 
# 예2 : 상자에 [사과 5개, 오렌지 5개]
# - 사과 비율 0.5, 오렌지 비율 0.5
# - Gini = 1 - (0.5² + 0.5²) = 1 - 0.5 = 0.5
# - 최대로 불순! (반반이라 가장 헤깔림)
# 
# 예3. 상자에 [사과 9개, 오렌지 1개]
# - 사과 비율 0.9, 오렌지 비율 0.1
# - Gini = 1 - (0.9² + 0.1²) = 1 - 0.82 = 0.18
# - 꽤 순수함

# 엔트로피(Entropy)
# Entropy = -
# 직관적 의미 : 얼마나 혼란스러운가? (정보이론 개념)
# 범위 : 0 ~ 1까지
# 
# 예1. [사과 10개]
# - Entropy = -(1.0 x log₂(1.0)) = 0
# - 전혀 혼란스럽지 않음
#  
# 예2. [사과 5개, 오렌지 5개]
# - Entropy = -(0.5 x log₂(0.5)) x 2 = 1
# - 최대로 혼란스러움
#
# 예3. [사과 9개, 오렌지 1개]
# - Entropy = -(0.9 x log₂(0.1)) = 0.47
# - 약간 혼란스러움

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# 데이터 로드
iris = load_iris()
x, y = iris.data, iris.target

# 분할
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# 모델 학습
model = DecisionTreeClassifier(
    random_state=42,
    criterion='gini',       # 분할 기준 : 'gini' 또는 'entropy'
    max_depth=5,            # 최대 깊이
    min_samples_split=10,   # 분할을 위한 최소 샘플 수
    min_samples_leaf=5,     # 리프 노드 최소 샘플 수
    max_features=None       # 분할에 사용할 특성 수
)
model.fit(x_train, y_train)

# 예측
y_pred = model.predict(x_test)

# 평가
print(f'정확도 : {accuracy_score(y_test, y_pred):.2f}')

# 한글 폰트 설정 추가
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
# plt.rcParams['font.family'] = 'AppleGothic'  # Mac
plt.rcParams['axes.unicode_minus'] = False

# 시각화
plt.figure(figsize=(20, 10))
plot_tree(model,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('붓꽃 분류 결정 트리')
plt.show()

# 텍스트로 출력
from sklearn.tree import export_text

tree_rules = export_text(
    model, feature_names=list(iris.feature_names)
)
print(tree_rules)

# ======================= 실습 ============================
print('=============== 실습 ================')
# kaggle : 데이터 불러오는 사이트
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# 1. 데이터 로드(csv 파일 경로)
Titanic = r"C:\Users\alsl0\Documents\python\코딩온_수업자료\AI_base\Titanic.csv"
df = pd.read_csv(Titanic, dtype=str)
# df = pd.read_csv('Titanic.cvs')

# 2. 필요한 특성만 선택(dropna())
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].dropna()

# 3. 성별을 숫자로 변환
df['Sex'] = df['Sex'].map({'male' : 0, 'female' : 1})
print(df.head())

# 4. 특성과 타겟 분리
x = df.drop(['Survived'], axis=1) # x = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = df['Survived']

# 5. 분할 
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# 6. 모델 학습
# 모델 학습
model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)

# 7. 평가
# 예측
y_pred = model.predict(x_test)
print(f'정확도 : {accuracy_score(y_test, y_pred):.2%}')

# 8. 시각화
# 한글 폰트 설정 추가
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
# plt.rcParams['font.family'] = 'AppleGothic'  # Mac
plt.rcParams['axes.unicode_minus'] = False

# 시각화
plt.figure(figsize=(20, 12))
plot_tree(model,
          feature_names= x.columns,
          class_names= ['Deth', 'Survived'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('타이타닉 생존 예측 결정 트리')
plt.show()