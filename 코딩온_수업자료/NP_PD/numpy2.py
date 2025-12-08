# 배열 모양 변경, 조작
import numpy as np
arr_1d = np.array([1, 2, 3, 4, 5, 6])
print('==========1차원=============')
print('shape', arr_1d.shape)  # 모양
print('ndim', arr_1d.ndim)  # 차원
print('size', arr_1d.size)  # 사이즈
print()
# print(arr_1d.reshape(2,)) 사이즈 오류
print(arr_1d.reshape(2, 3))
print()
print(arr_1d.reshape(3, -1))
# print(arr_1d.reshape(4, -1)) 차원 오류


arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print('==========2차원=============')
print('shape', arr_2d.shape)  # 모양
print('ndim', arr_2d.ndim)  # 차원
print('size', arr_2d.size)  # 사이즈

# 기본 산술 연산
a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 4, 3, 2, 1])

print('a 배열:', a)
print('b 배열:', b)

print('덧셈 (a + b)\n', a + b)
print('뺄셈 (a - b)\n', a - b)
print('곱셈 (a * b)\n', a * b)
print('제곱 (a ** b)\n', a ** b)
print('나눗셈 (a / b)\n', a / b)
print('몫 (a // b)\n', a // b)
print('나머지 (a % b)\n', a % b)
print()

# 스칼라와의 연산
a = np.array([1, 2, 3, 4, 5])
scalar = 10

print('+ 스칼라', a + scalar)
print('- 스칼라', a - scalar)
print('* 스칼라', a * scalar)
print('/ 스칼라', a / scalar)

print('스칼라 / 배열', scalar / a)

A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
B = np.array([
    [7, 8, 9],
    [10, 11, 12]
])

print('행렬 A\n', A)
print('행렬 B\n', B)
print()

print('행렬 A + B\n', A + B)
print()
print('행렬 A * B\n', A * B)
print()
print('행렬 A / B\n', A / B)

A = np.array([
    [1, 2],
    [3, 4]
])
B = np.array([
    [7, 8],
    [9, 10]
])
print('행렬의 곱셈(A @ B)\n', A @ B)

# 브로드 캐스팅(Broadcasting)
# 서로 다른 모양의 배열 간 연산을 사능하게 하는 기능
arr = np.array([1, 2, 3, 4, 5])
scalar = 10

# 스칼라가 자동으로 배열 크기로 '브로드캐스트' 됨
# [1, 2, 3, 4, 5] + [10, 10, 10, 10, 10]

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
vector = np.array([10, 20, 30])
print(matrix + vector)
'''
[
    [10, 20, 30],
    [10, 20, 30],
    [10, 20, 30]
]
'''
'''
브로드캐스팅 규칙 / 1차원 : (n, ) / 2차원 : (n, m) / 3차원 : (n, m, k)
a = np.array([1, 2, 3]) => 1차원 (3, )
b = np.array([[4], [5], [6]]) => 2차원 (3, 1)

 1. 차원 수가 다르면 작은 쪽의 앞에 1을 추가 (3,3) + (3, ) -> (1,3) => (3, 3)
 2. 각 차원에서 크기가 1이거나 같아야 함
    호환가능 : (3, 1) & (1, 4) = (3, 4)
    호환불가 : (3, 2) & (4, 2) = 에러...! 
    두 차원의 행열 중 하나가 같고 다른 하나가 다를 때, 다른 행열 중 요소가 1이면 성립 (그 외 오류)
    ex) (4,2) + (3, ) => 오류!!        왜냐 (3, )->(1, 3)이므로 요소가 다름
        그러나, (4,2) + (2, ) => 성립   왜냐 (2, )->(1, 2)이므로 요소가 같음
'''

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
np.sum(arr)  # 의미 github
np.mean(arr)        # 평균
np.median(arr)      # 중앙값
np.std(arr)         # 표준편차
np.max(arr)         # 최대값
np.min(arr)         # 최소값
np.var(arr)         # 분산
np.ptp(arr)         # max - min
np.cumsum(arr)      # 누적 합
np.cumprod(arr)     # 누적 곱
print()

martix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print('행별 합 (axis=1)', np.sum(matrix, axis=1))
print('열별 합 (axis=0)', np.sum(matrix, axis=0))

print('행별 평균 (axis=1)', np.mean(matrix, axis=1))
print('열별 평균 (axis=0)', np.mean(matrix, axis=0))

print('행별 누적 합 (axis=1)', np.cumsum(matrix, axis=1))
print('열별 누적 합 (axis=0)', np.cumsum(matrix, axis=0))
