# 데이터 셋 분할

# 왜 데이터를 나누는가?
# 학교 시험 비유
# 나쁜 방법 (컨닝)
# 시험 문제로 공부
# 시험 문제로 시험
# 결과: 100점!
# 
# 좋은 방법
# 연습 문제로 송부
# 다른 문제로 시험
# 결과 : 진짜 실력!

# ML에서의 의미
# 문제 : 
# 

# 세 가지 데이터 셋
# 훈련 / 검증 / 테스트 
# 전체 데이터
# |
# |- 훈련 세트(Training Set) 60~80%
# |    -- 모델 학습에 사용
# | 
# |- 검증 세트(Validation Set) 10~20%
# |    -- 하이퍼 파라미터 튜닝에 사용
# | 
# |- 테스트 세트(Test Set) 10~20%
# |    -- 최종 성능 평가에 사용
# | 

# 각 세트의 역할
# 훈련 세트
# - 모델이 패턴을 학습
# - '공부 할 때 보는 교재'
# 
# 검증 세트
# - 학습 중간에 성능 확인
# - 하이퍼 파라미터 조정
# - '모의고사'
# 
# 테스트 세트
# - 최종 성능만 측정
# - 한 번 만 사용!
# - '실제 시험'
from sklearn.model_selection import train_test_split
# pip install scikit-learn
import numpy as np

# 훈련 / 테스트 (기본 분할)

# 샘플 데이터
# 훈련 데이터 : 8
# 테스트 데이터 : 2

# 훈련 / 검증 / 테스트 분할

# 1단계 : 훈련 + 검증 vs 테스트 (80 : 20)
# 내용 있음
# 2단걔 : 훈런 vs 검증 (75 : 25)
# 훈련 데이터 : 6
# 검증 데이터 : 2
# 테스트 데이터 : 2

# 계층 분할(Stratified)
# 
# 클래스 비율을 유지하면서 분할
# 불균형 데이터 예시
y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])   # 7 : 3

# stratified 옵션 사용
# 전체 [7 3] 
# 훈련 [6 2] 
# 테스트 [1 1]
#
# shuffle

# 기본적으로 shuffle=True (데이터 섞음) 
# 시계열 데이터는 셔플 금지 

# 적절한 비율
# 데이터 양에 따른 권장 비율
# 
# 대용량 (10만 이상)
# - 훈련 98% : 테스트 2%
# - 검증은 별도로 1~2%
# 
# 중간 (1000 ~ 10만)
# - 훈련 80% : 테스트 20%
# - 60 : 20 : 20
# 
# 소량 (1000미만)
# - 교차 검증 권장 

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# 붓꽃 데이터 로드
iris = load_iris()
X = iris.data       # 특성 (꽃잎, 꽃받침 크기)
y = iris.target     # 라벨 (품종)

print(f"전체 데이터: {X.shape}")
print(f"클래스: {np.unique(y)}")  # [0, 1, 2]

# 분할 (80 : 20)

# =======================실습=======================
# from sklearn.datasets import load_wine
from sklearn.datasets import load_wine
wine = load_wine()
X, Y = wine.data, wine.target

