import numpy as np
# 실습 01. 배열 형태 변형, 차원 확장 축소
# 문제 1. ravel, flatten을 사용하여 1차원으로 바꾸고 첫 번째 원소를 999로 바꾼 뒤 결과 확인
arr = np.array([[10, 20], [30, 40], [50, 60]])

rev = arr.ravel()
flattened = arr.flatten()
print('raval 결과: ', rev)
print('flattened 결과: ', flattened)
arr[0, 0] = 999
print('raval 결과: ', rev)  # 변경
print('flattened 결과: ', flattened)  # 변경X

# 문제 2. 크기가 32x32인 이미지 데이터를 가정하고, 이 배열에 대해 expand_dims를 사용하여
# shape (1, 32, 32)로 바꾸는 코드를 작성하세요.
img = np.random.rand(32, 32)
img1 = np.expand_dims(img, axis=0)
print('shape', img1.shape)

# 문제 3. 아래 배열에서 불필요한 1차원을 모두 제거하여 shape이 (28, 28)이 되도록 만드세요.
img = np.random.randint(0, 255, (1, 28, 28, 1))
img_s = np.squeeze(img)
print('img_s.shape:', img_s.shape)

# 문제 4. 아래 2차원 배열을
# 1) 1차원 배열로 만든 후
# 2) 중복값을 제거한 뒤 shape (1, n)으로 재구성 하세요.
arr = np.array([[3, 1, 2, 2],
                [1, 2, 3, 1],
                [2, 2, 1, 4]])

# 평탄화 -> 1차원 변환
flat = arr.flatten()
# 중복 제거
uniq = np.unique(flat)
# (1, n)으로 변환
reshaped = uniq.reshape(1, -1)
print('원본\n', arr)
print('1차원', flat)
print('중복 제거', uniq)
print('(1, n)형태', reshaped)
print('(1, n)형태 구성', reshaped.shape)
print()
'''============================'''

# 문제 5. 다음 배열을 shape (10,)로 만든 뒤 고유값 배열을 구하세요.
# shape (1, 10, 1)
arr = np.array([[[1], [3], [2], [1], [3], [2], [3], [1], [2], [3]]])
ar = np.squeeze(arr, axis=0)
ar1 = np.squeeze(ar, axis=1)
print(ar1.shape)
cnt = np.unique(ar1, return_counts=True)
print('고유값 배열', cnt)

# 문제 6. 다음 배열을 1차원 배열로 만든 후 고유값만 추출해서 shape (고유값 개수, 1)인 2차원 배열로 변환하세요.
arr = np.array([
    [[0, 1, 2, 3],
     [1, 2, 3, 4],
     [2, 3, 4, 5]],

    [[3, 4, 5, 6],
     [4, 5, 6, 7],
     [5, 6, 7, 8]]
])  # shape (2, 3, 4)

uniq = np.unique(arr)  # (9, )
cnt = np.unique(arr, return_counts=True)
print('고유값 배열', cnt)
'''============================'''

# ===================================================
# 실습 02. 배열의 결합과 분리
# 문제 1. 다음 두 배열을 행 방향으로 이어붙이세요.
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
c = np.vstack((a, b))
print('배열 결합\n', c)

# 문제 2. 아래 배열을 3개로 같은 크기로 분할하세요.
a = np.arange(12)
sep = np.split(a, 3)
print('3개로 분리', sep)

# 문제 3. 다음 배열들을 새로운 축에 쌓아 shape이 (3, 2)인 배열을 만드세요.
a = np.array([1, 2])
b = np.array([3, 4])
c = np.array([5, 6])
d = np.vstack((a, b, c))
print('vstack\n', d)
print('새로운 축 모양', d.shape)
result = np.stack((a, b, c), axis=0)
print('stack\n', result)

# 문제 4. shape가 (2, 3)인 아래 두 배열을 shape (2, 2, 3)인 3차원 배열을 만드세요.
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[7, 8, 9], [10, 11, 12]])
c = np.stack((a,  b), axis=0)
print('3차원 배열 생성', c.shape)

# 문제 5. 아래의 1차원 배열을 2:3:3 비율(총 3개)로 분할하세요.
arr = np.arange(8)
sep = np.split(arr, [2, 5])
print('2:3:3 비율로 분할', sep)

# 문제 6. 아래 두 배열을 axis=0, axis=1로 각각 stack하여 두 경우의 결과 shape을 모두 구하세요
a = np.ones((2, 3))
b = np.zeros((2, 3))
c0 = np.stack((a, b), axis=0)
c1 = np.stack((a, b), axis=1)
print(c0)
print('===================')
print(c1)
print()

# ===============================================
# 실습 03. 배열의 정렬
# 문제 1. 아래의 1차원 배열을 오름차순과 내림차순으로 각각 정렬하는 코드를 작성하세요.
arr = np.array([7, 2, 9, 4, 5])
arr.sort()
print(arr)
print(arr[::-1])

# 문제 2. 아래의 2차원 배열에서 각 행(row) 별로 오름차순 정렬된 배열을 구하세요.
arr = np.array([[9, 2, 5],
                [3, 8, 1]])
arr.sort()
print(arr)

# 문제 3. 아래의 1차원 배열에서 정렬 결과(오름차순)가 되는 인덱스 배열을 구하고,
# 그 인덱스를 이용해 원본 배열을 직접 재정렬하는 코드를 작성하세요.
arr = np.array([10, 3, 7, 1, 9])
idx = np.argsort(arr)
print(idx)
print(arr[idx])

# 문제 4. 아래 2차원 배열을 열(column) 기준(axis=0)으로 오름차순 정렬된 배열을 구하세요.
arr = np.array([[4, 7, 2],
                [9, 1, 5],
                [6, 8, 3]])
arr.sort()
print(arr)
