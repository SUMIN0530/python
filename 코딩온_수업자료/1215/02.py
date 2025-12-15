# 행렬
# 행렬(Matrix) = 숫자들을 행과 열로 배열한 것
# 따오기

# 텐서 : 3차원 이상의 배열

# 일상에서의 행렬
# 성적표
# 따오기

# 이미지
# 28 X 28 픽셀 이미지 = 28행 X 28열 행렬
# 각 셀의 값 = 픽셀의 밝기 (0 ~ 255)

import numpy as np

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print(matrix)

zeros = np.zeros((3, 4))
print(zeros)

# 행렬 형태 (행, 열)
print(matrix.shape)
# 전체 원소 개수
print(matrix.size)
# 차원
print(matrix.ndim)

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(A + B)
print(A - B)
print(A * 2)

# 행렬 곱셈(중요!!!!!)
# 행렬 곱셈 != 원소별 곱셈

# A X B의 규칙
# A의 열 개수 = B의 행 개수 => 계산 가능
# (2 x 3) X (3 x 2) = (2 x 2)
# (2 x 3) X (2 x 2) = 계산 불가능

# A = [1 2]     B = [5 6]
#     [3 4]         [7 8]

# A X B = [1x5 + 2x7    1x6 + 2x8]   =======> [19 21]
#         [3x5 + 4x7    3x6 + 4x8]            [43 50]

# 방법 1 : @ 연산자(권장)
result = A @ B
print(result)

# 방법 2 : np.dot()함수
result = np.dot(A, B)
print(result)

# 방법 3 : np.matmul()함수
result = np.matmul(A, B)
print(result)

# 행렬 곱셈 주의사항 -> 순서 중요!!!
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[1, 2], [3, 4], [5, 6]])

# A @ B
print((A @ B).shape)

# B @ A
print((B @ A).shape)

# 행렬의 전치와 형태 변환
# 전치(transpose)
# 행과 열을 바꿈
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# 전치 행렬
A_T = A.T # A.transpose()
print(A_T)
print()

a = np.array([1, 2, 3, 4, 5, 6])
b = a.reshape(3, 2)
print(b)
print()

d = a.reshape(2, -1) # 2행, 열은 자동
print(d)

# AI에서 행렬의 역할
# 신경망에서의 행렬
# 신경망의 핵심 = 행렬 곱셈!

# 입력 데이터(벡터) X 가중치(행렬) = 출력
# 입력 : [x1, x2, x3]
# 가중치 : [
#   [w11, w12],
#   [w21, w22],
#   [w31, w33]
# ]

# 출력 = 입력 @ 가중치 = [y1, y2]

# 이미지 데이터
# 이미지 = 행렬(또는 3차원 텐서)

# 흑백 이미지 : 2차원 행렬

# 컬러 이미지 : 3차원 텐서(높이, 너비, 채널)

# ================ 실습====================
print('============== 실습 ===============')
A = np.array([[1, 2], [3, 4], [5, 6]]) # 3x2
B = np.array([[1, 2, 3], [4, 5, 6]]) # 2x3
c = np.arange(12)

print('A의 shape : ', A.shape)
print('A의 전치 행렬 : ', A.T)
print('A @ B  = ', A @ B)
print('B @ A  = ', B @ A)
print(c.reshape(3, 4))