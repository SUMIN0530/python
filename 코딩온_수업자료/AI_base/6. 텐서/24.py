import torch
'''# 텐서 생성하기
# 실습 
# 1. 3x3 크기의 0으로 채워진 텐서
zeros = torch.zeros(3, 3)
print(zeros)

# 2. 2x4 크기의 1로 채워진 텐서
ones = torch.ones(2, 4)
print(ones)

# 3. 0부터 9까지의 숫자가 들어있는 1차원 텐서
numbers = torch.arange(10)
print(numbers)

# 4. 평균 0, 표준편차 1인 정규 분포에서 샘플링한 5x5 텐서
random_normal = torch.randn(5, 5)
print(random_normal)

# 5. 3x3 단위행렬 (대각선이 1인 행렬)
identity = torch.eye(5, 5)
print(identity)

# ============================================================
# 실습 2
# 주어진 텐서
x = torch. arange(24)
print(f'원본 : {x}\n')
# 다음 변환을 수행하세요

# 1. 2x12 행태로 변환
shape_2_12 = x.reshape(2, 12)
print(shape_2_12)

shape_3_8 = shape_2_12.reshape(3, 8)
print(shape_3_8)

shape_2_3_4 = shape_3_8.reshape(2, 3, 4)
print(shape_2_3_4)

shape_4_2_3 = shape_2_3_4.reshape(4, 2, 3)
print(shape_4_2_3)

flattened = shape_4_2_3.reshape(24,)
print(flattened)

# =============================================================
# 실습 3
# 다음 연산을 수행하세요
A = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
], dtype=torch.float32)

B = torch.tensor([
    [2, 0, 1],
    [1, 3, 2]
], dtype=torch.float32)

print(f'A : \n {A}\n')
print(f'B : \n {B}\n')

# 1. A와 B의 원소별 합
element_sum = A + B
print(element_sum)
print()

# 2. A와 B의 원소별 곱
element_mul = A @ B.T
print(element_mul)
print()

# 3. A의 모든 원소 제곱
squared1 = torch.tensor(A ** 2)
squared2 = torch.pow(A, 2)
print(squared1)
print(squared2)
print()

# 4. A의 각 행의 합
row_sum_A = A.sum(dim=1)
print(row_sum_A)
print()

# 5. A의 각 열의 평균
torch.set_printoptions(precision=3) # 출력 표시 제한하기 (소수점 3자리)
col_mean = A.mean(dim=0)
print(col_mean)
print()

# 6. A에서 3보다 큰 원소들만 추출
greater_than_3 = A[A > 3]
print(greater_than_3)
print()'''

# ================================================================
# 실습 4
# 배치 데이터 (10개 샘플, 각 5개 특성)
batch = torch.randn(10, 5)
print(f'배치 데이터 크기 : {batch.shape}\n')

# 브로드캐스팅을 이용하여 다음을 수행
# 1. 모든 샘플에 벡터 [1, 2, 3, 4, 5]를 더하기
vector = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
result1 = batch + vector
print(result1)
print()

# 2. 각 특성(열)의 평균을 구하고, 각 샘플에서 해당 평균을 빼기 (중심화)
col_mean = batch.mean(dim=0) # 평균 계산
centered = batch - col_mean # 중심화
print(col_mean)
print(centered)
print()

# 3. 각 특성의 최솟값과 최댓값을 구하고, 0 ~ 1 범위로 정규화
# 공식 : (x - min) / (max - min)
min_vals = batch.min(dim=0).values
max_vals = batch.max(dim=0).values
normalized = (batch - min_vals) / (max_vals - min_vals)
print(min_vals)
print(max_vals)
print(normalized)
