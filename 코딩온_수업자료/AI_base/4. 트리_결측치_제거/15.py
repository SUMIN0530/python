# 실습
# 누가 생존 했는가? 예측가는 분류 모델을 처음부터 끝까지 만들어보는 것
# 1단계 : 데이터 탐색(EDA)
# 데이터 불러오기
# 결측치가 어디에 얼마나 있는지 확인
# 생존/사망 비율 파악
# 성별, 객실, 등급별 생존율 시각화
# 특성 간 상관관계 히트맵 그리기
import pandas as pd
df1 = pd.read_csv('Titanic1.csv')
df2 = pd.read_csv('Titanic2.csv')

df = pd.concat([df1, df2], ignore_index=True) # concat : (같은 컬럼)상하 병합, marge : 컬럼 병합

missing_count = df.isnull().sum()
print('결측치 분포 확인\n', missing_count)

count = df['Survived'].value_counts().reset_index()
count['rate'] = count["count"] / count["count"].sum() * 100
count['rate'] = count['rate'].round(2)
print(count)

# 성별, 나이대, 객실 등급별 생존율 시각화
import matplotlib.pyplot as plt
bins = [0, 10, 20, 30, 40, 50, 60, 100] # 나이대 구간 나누기
labels = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60+"]
df["Age"] = pd.cut(df["Age"], bins=bins, labels=labels)

# pd.Categorical : 나이 구간이 단순 문자열 리스트가 아닌 순서가 있는 법주형 데이터(위치 고정)임을 나타냄
# categories : 현재 순서가 정답이다.
# ordered : 크기 비교, 정렬 가능
df["Age"] = pd.Categorical(
    df["Age"], categories=labels, ordered=True
)

cols = ["Sex", "Pclass", "Age"]
for col in cols:
    survival = df.groupby(col)["Survived"].mean() * 100
    survival.plot(kind="bar")
    plt.ylabel("Survival Rate (%)")
    plt.title(f"Survival Rate by {col}")
    plt.show()

# 히트맵 그리기
pairs = [
    ("Sex", "Age"),
    ("Sex", "Pclass"),
    ("Age", "Pclass")
]

titles = [
    "Survival Rate Heatmap (Sex × Age Group)",
    "Survival Rate Heatmap (Sex × Cabin Grade)",
    "Survival Rate Heatmap (Age Group × Cabin Grade)"
]

for (a, b), title in zip(pairs, titles):

    pivot = (
        df.groupby([a, b])["Survived"]
        .mean()
        .unstack()
        * 100
    )

    plt.figure()  # 🔴 히트맵 개별 그림 이 줄이 핵심
    plt.imshow(pivot, aspect="auto")
    plt.colorbar(label="Survival Rate (%)")

    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)

    plt.title(title)
    plt.xlabel(b)
    plt.ylabel(a)

    # 🔹 여기부터 수치 표시 핵심
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if not pd.isna(value):
                plt.text(
                    j, i,
                    f"{value:.1f}",
                    ha="center",
                    va="center"
                )

    plt.show()     # 🔴 이것도 for문 안에 있어야 함

# 2단계 : 데이터 전처리
# 필요한 특성만 선택
# 결측치 채우기
# 범주형 데이터를 숫자로 변환
# 훈련/테스트 세트 분할

# Age를 카테고리화 시켜놨기 때문에 결측치 처리 불가 (숫자형일 때만 처리 가능)
df1 = pd.concat([df1, df2], ignore_index=True) # 새로 생성
# df = df[['Survived', 'Pclass', 'Sex', 'Age']].dropna()  --> 필요 열 제외 나머지 버림 + 결측값이 있는 행 삭제(.dropna())
df1 = df1[['Survived', 'Pclass', 'Sex', 'Age']] #.fillna()  --> 결측값이 있는 행 대체값 삽입
df1['Age'] = df1.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))

missing_count = df1.isnull().sum()
print('결측치 분포 확인\n', missing_count)

df1['Sex'] = df1['Sex'].map({'male' : 0, 'female' : 1})
print(df1.head(10))

x = df.drop(['Survived'], axis=1) # x = df[['Pclass', 'Sex', 'Age']]
y = df['Survived']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)






# 3단계 : 모델 학습 및 비교
# 3가지 모델 훈련 : 로지스틱 회귀, 결정 트리, 랜덤 포레스트
# 교차 겁증으로 성능 비교
# GridSearchCV로 랜덤 포레스트 하이퍼파라미터 튜닝





# 4단계 : 평가
# 테스트 세트로 최종 정확도 측정
# 분류 모델이 에측을 어마나 맞췄는지를 정합 평가로 활용?
# 혼동 행렬이 어디서 틀렸는지 확인
# 특성 중요도 확인 (어떤 특성이 생존 예측에 중요했나)  