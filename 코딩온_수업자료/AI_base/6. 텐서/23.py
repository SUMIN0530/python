# 행렬 연산

# 행렬 곱셈
# 2D 행렬
import torch
A = torch.tensor([
    [1, 2],
    [3, 4],
],dtype=torch.float32)

B = torch.tensor([
    [5, 6],
    [7, 8],
],dtype=torch.float32)

# 행렬 곱셈 (Matrix Multiplication)
print('=== 행렬 곱셈 ===')
print(f'A @ B :\n {A @ B}')
print(f'torch.mm(A, B) : \n {torch.mm(A, B)}')
print(f'torch.matmul(A, B) : \n {torch.matmul(A, B)}')

# 행렬 - 벡터 곱 (1차원)
matrix = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
], dtype=torch.float32)

vector = torch.tensor([1, 2, 3], dtype=torch.float32)

print(f'행렬 크기 : {matrix.shape}')
print(f'벡터 크기 : {vector.shape}')

result = torch.mv(matrix, vector)

print(f'결과 : {result}')
print(f'결과 크기 : {result.shape}')

# 배치 행렬 곱셈
# 3D 텐서 (배치 행렬)
#(batch, m, n) @ (batch, n, p) => (batch, m ,p)
batch_A = torch.randn(10, 3, 4) # 10개의 3 X 4 행렬
batch_B = torch.randn(10, 4, 5) # 10개의 4 X 5 행렬

result = torch.bmm(batch_A, batch_B)
print(f'결과 : {result}')
print(f'결과 크기 : {result.shape}')

# matmul은 자동으로 배치 처리
result = torch.matmul(batch_A, batch_B)
print(f'결과 크기 : {result.shape}')

B = torch.randn(4, 5) # (2, 4, 5) 브로딩캐스팅 문제로 RuntimeError  
result = torch.matmul(batch_A, B) # bmm 사용 => RuntimeError
print(f'결과 크기 : {result.shape}')

# 전치 행렬
A = torch.tensor([
    [1, 2],
    [3, 4]
], dtype=torch.float32)

# 전치 : 행렬의 행(row)과 열(column)을 서로 바꾼다
print(f'원본 : \n {A}')
print(f'전치 (A.T) : \n {A.T}')
print(f'전치 (transpose) : \n {A.transpose(0, 1)}')

# 역행렬
A_inv = torch.inverse(A)
print(f'역행렬 : \n {A_inv}')
print(f'A @ A_inv : \n {A @ A_inv}')

# 브로드캐스팅
# 크기가 다른 텐서 간의 연산 규칙
#
# 규칙
# 1. 차원이 다르면 작은 쪽에 1을 앞에 추가
# 2. 크기 1인 차원은 다른 텐서에 맞춰 확장
# 3. 크기가 다르고 둘 다 1이 아니라면 에러
# 
# 예시 
# (3, 4) + (4, ) => (3, 4) + (1, 4) => (3, 4)
# (3, 1) + (1, 4) => (3, 4)
# (3, 4) + (3, ) => 에러 발생

# 1D와 2D
matrix = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])  

vector = torch.tensor([10, 20, 30])
print(f'행렬 크기 : {matrix.shape}') # (2, 3)
print(f'벡터 크기 : {vector.shape}') # (3, ) => (1, 3) 변경
# [
#  [10, 20, 30],
#  [10, 20, 30]
# ]


result = matrix + vector

print(f'결과 : {result}')

# 열 벡터와 행 벡터
col_vec = torch.tensor([[1], [2], [3]])
row_vec = torch.tensor([10, 20, 30, 40])

print(f'열 벡터 크기 : {col_vec.shape}') # (3, 1) => (3, 1) => (3, 4)
print(f'행 벡터 크기 : {row_vec.shape}') # (4, ) => (1, 4) => (3, 4)
# [
#  [1, 1, 1, 1],
#  [2, 2 ,2, 2],
#  [3, 3, 3, 3] 
# ]

# [
#  [10, 20, 30, 40],
#  [10, 20, 30, 40],
#  [10, 20, 30, 40] 
# ]

result = col_vec + row_vec
print(f'결과 : {result}')

# 배치 정규화 개념
# 한 배치에 32명이 있고, 각 사람마다 10개의 숫자(특성)가 있는 형태
batch = torch.randn(32, 10) # 32개 샘플, 10개 특성
# 행 방향(샘플 방향)으로 평균을 낸다
mean = batch.mean(dim=0)    # 각 특성의 평균 (10, )
std = batch.std(dim=0)      # 각 특성의 표준편차 (10, )

# 정규화 (브로드캐스팅 활용)
normalized = (batch - mean) / std
# (batch - mean)에서 브로드캐스팅 동작

print(f'배치 크기: {batch.shape}')
print(f'평균 크기: {mean.shape}')
print(f'정규화 후: {normalized.shape}')
print(f'정규화 평균: {normalized.mean(dim=0)}')
print(f'정규화 표준편차: {normalized.std(dim=0)}')

# 집계 함수
x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
], dtype=torch.float32)

print(f'합계 : {x.sum()}')
print(f'평균 : {x.mean()}')
print(f'최소값 : {x.min()}')
print(f'최대값 : {x.max()}')
print(f'표준편차 : {x.std()}')
print(f'분산 : {x.var()}')

# 차원별 집계
x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
], dtype=torch.float32)

# dim=0 : 행 방향 (세로로 집계)
print(f'dim=0 합계 : {x.sum(dim=0)}')
print(f'dim=0 평균 : {x.mean(dim=0)}')

# dim=1 : 열 방향 (가로로 집계)
print(f'dim=1 합계 : {x.sum(dim=1)}')
print(f'dim=1 평균 : {x.mean(dim=1)}')

# keepdim=True
print(f'keepdim=True')
print(f'dim=0 합계 : {x.sum(dim=0, keepdim=True)}')
print(f'dim=1 합계 : {x.sum(dim=1, keepdim=True)}')

# argmax와 argmin
x = torch.tensor([
    [3, 2, 3],
    [9, 7, 0],
], dtype=torch.float32)
print(f'전채 armmax : {x.argmax()}')
print(f'전체 argmin : {x.argmin()}')

# 차원별 최대 위치
print(f'dim=0 argmax : {x.argmax(dim=0)}')
print(f'dim=1 argmax : {x.argmax(dim=1)}')

# 텐서 결합
a = torch. tensor ([[1, 2], [3, 4]])
b = torch. tensor ([[5, 6], [7, 8]])

# dim=0 : 행 방향 결합(아래로)
cat0 = torch.cat([a, b], dim=0)
print(f'cat dim=0 : \n {cat0}')

# dim=1 : 열 방향 결합(가로로)
cat1 = torch.cat([a, b], dim=1)
print(f'cat dim=1 : \n {cat1}')

# stack
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = torch.tensor([7, 8, 9])

# stack : 새차원 추가하며 결합

stacked0 = torch.stack([a, b, c], dim=0)
print(f'stacked dim=0 : \n {stacked0}')

stacked1 = torch.stack([a, b, c], dim=1)
print(f'stacked dim=1 : \n {stacked1}')

# split : 지정 크기로 분할
splits = torch.split(x, 2, dim=1) # 옆 방향 2개씩
print(f'split (size=2, dim=1)')
for i, s in enumerate(splits):
    print(f' {i} : {s.shape}')
    print(f' {i} : {s}')

# chunk : 지정 개수로 분할
chunks = torch.chunk(x, 3, dim=0) # 옆 방향 3등분
print(f'chunk (chunks=3, dim=0)')
for i, c in enumerate(chunks):
    print(f' {i} : {c.shape}')
    print(f' {i} : {c}')

# 비교 연산
a = torch.tensor([1, 2, 3, 4])
b = torch.tensor([2, 2, 2, 2])

# 비교
print(f' a > b : {a > b}')
print(f' a >= b : {a >= b}')
print(f' a == b : {a == b}')
print(f' a != b : {a != b}')

# torch 함수
print(f'torch.gt(a, b) : { torch.gt(a, b)}')
print(f'torch.eq(a, b) : { torch.eq(a, b)}')

# 조건 검사
print(f'전부 같음 : {torch.all (a == b)}')
print(f'하나라도 같음 : {torch.any (a == b)}')

