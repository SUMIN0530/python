'''
 1. python -m venv 폴더 이름
 1. python -m venv venv(파일명?)
 2. source venv/Scripts/activate (venv)생성
 3. pip install numpy (numpy 다운)
 4. pip list : numpy, pip

 Numpy(Numberiacl Python)는 파이썬에서 과학계산을 위한 핵심 라이브러리
 데이터 과하그 머신러니으 과학 연구 분야에서 가장 중요한 도구 중 하나

속도문제 해결
 파이썬은 인터프리터 언어로 실행 속도가 느림
 NumPy는 C언어로 구현되어 있어 대용량 데이터 연산 매우 빠르게 처리

메모리 효울성
 파이썬 리스트 : 각 요소가 객체로 저장되어 메모리 오버헤드가 크다
 Numpy 배열 : 연속된 메모리 공간에 같은 타임의 데이터 저장

벡터화 연산
 반복문 없이 전체 배열에 대한 연산 한번에 수행
 '''


import numpy as np

print('Numpy 버젼:', np.__version__)
print('Numpy 설치 경로:', np.__file__)

# ndarray(N-dimensional array) Numpy의 핵심 자료 구조
# 같은 타입의 요소들을 담는 다차원 컨테이너

arr = np.array([1, 2, 3, 4, 5])

print('1. 객체 타입', type(arr))
print('2. 데이터 타입', arr.dtype)
print('3. 배열 모양:', arr.shape)
print('4. 차원 수:', arr.ndim)
print('5. 전체 요소 수', arr.size)

python_list = [1, 2.5, '3.', True]
numpy_array = np.array([1, 2.5, '3.', True])
# 문자 존재 : 문자열로 통일 / 문자 X : 리스트?로 통일 / 소수 X : 정수로 통일
print('파이썬 리스트:', python_list)
print('Numpy 배열:', numpy_array)

# 중요한 차이점 2: 연산 방식
list1 = [1, 2, 3]
list2 = [4, 5, 6]
print('리스트 더하기:', list1 + list2)

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print('Numpy 배열 더하기:', arr1 + arr2)
print()

# 정수 배열
int_array = np.array([1, 2, 3, 4, 5])
print('정수 배열:', int_array)
print('데이터 타임:', int_array.dtype)
print()

# 실수 배열
float_array = np.array([1.1, 2.2, 3.3, 4.4, 5])
print('실수 배열:', float_array)
print('데이터 타임:', float_array.dtype)
print()

# 타입을 명시적으로 지정 배열
specified_array = np.array([1, 2, 3, 4, 5], dtype=np.float32)
print('명시적 배열:', specified_array)
print('데이터 타임:', specified_array.dtype)
print()
specified_array = np.array(['1', '3.2', 3, 4, 5], dtype=np.float32)
print('명시적 배열:', specified_array)
print('데이터 타임:', specified_array.dtype)
print()

# 문자열 배열
string_array = np.array(['apple', 'banana', 'cherry'])
print('문자열 배열:', string_array)
print('데이터 타임:', string_array.dtype)  # <U10 (유니코드 문자열, 최대 10자)

# 2차원 배열(3X3 행렬)
matrix = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [4, 5, 6]
])

print('2차원 배열:', matrix)
print('모양:', matrix.shape)  # 모양 (3, 3)
print('차원:', matrix.ndim)  # 차원 2
print('크기:', matrix.size)  # 크기 9

for i in range(3):
    for j in range(3):
        print()

rows = []
for i in range(3):
    row = [i * 3 + j for j in range(4)]  # [0, 1, 2, 3] -> # [3, 4, 5, 6]
    rows.append(row)

martix2 = np.array(rows)
print()
print('동적 생성 행렬:', martix2)

# 3차원 배열 (2 X 3 X 4)
# 2개의 3 X 4 행렬로 구성
tensor = np.array([
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
    ],
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
    ]
])

print('3차원 배열 모양:', tensor.shape)
print('차원:', tensor.ndim)

# numpy 내장 함수로 배열 생성
# 연속된 숫자 배열 arange
arr1 = np.arange(10)
print('0부터 9까지', arr1)

arr2 = np.arange(11)
print('0부터 10까지', arr2)

arr3 = np.arange(1, 21, 2)
print('1부터 20까지 홀수만', arr3)

arr4 = np.arange(1, 11, 0.5)  # range는 간격이 int만 가능
print('1부터 10까지 0.5간격', arr4)

# 균등 간격 배열 linspace
# 시작, 끝 사이를 균등하게 나눈 숫자들


arr1 = np.linspace(0, 10, 5)
print()
print('0부터 10까지 5개 요소로 균등하게 나눈다.\n', arr1)
'''
    step = (stop - start) / (num -1)
'''

arr2 = np.linspace(0, 10, 5, endpoint=False)
print('끝값을 제외하고 균등하게 나눈다.\n', arr2)
'''
    step = (stop - start) / num
'''

# 로그 간격 배열 logspace
# logspace(start, end, num)  # 지나가용

# zeros: 0으로 채운 배열
print()
zeros_1d = np.zeros(5)
print('1차원 zeros:', zeros_1d)

zeros_2d = np.zeros((3, 4))  # 실수형으로 도출
print('2차원 zeros:', zeros_2d)
print()
zeros_2d = np.zeros((3, 4), dtype=int)  # 정수형으로 도출
print('2차원 zeros:', zeros_2d)

# 2차원 배열(3X3 행렬)
matrix = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [4, 5, 6]
])
# 기존 배열과 같은 모양의 0 배열 생성
zeros_copy = np.zeros_like(matrix)
print('zeros_like:', zeros_copy)

# 1차원 1배열
ones_1d = np.ones(5)
print('1차원 ones:', ones_1d)

# 2차원 1배열(3, 4)
ones_2d = np.ones((3, 4))
print('2차원 ones:', ones_2d)

# 2차원 1배열(3, 4) bool타입
ones_2d_bool = np.ones((3, 4), dtype=bool)
print('2차원 ones:', ones_2d_bool)

# full
full_array = np.full((3, 4), 7)
print('2차원 배열:', full_array)

full_like = np.full_like(matrix, 999)
print('2차원 배열:', full_like)

# 메모리만 할당. 값은 쓰레기 값
empty_array = np.empty((2, 3))
print('2차원 배열(주의: 쓰레기 값):', empty_array)

# 3X3 항등 행렬
identity = np.eye(3)
print('3X3 항등 행렬:\n', identity)

# 4X5 행렬에서 대각선이 1
matrix = np.eye(4, 5)
print('4X5 대각 1:\n', matrix)

# 대각선 위치 조정(k 매개변수)
matrix = np.eye(4, k=1)
print('위쪽 대각선 1:\n', matrix)
print()
matrix = np.eye(4, k=-1)
print('아래쪽 대각선 1:\n', matrix)

# 정방 항등 행렬 - eye와 비슷
identity = np.identity(4)
print('4X4 항등 행렬:\n', identity)
print()

# 0과 1 사이에 균일 분포 (랜덤 도출)
random_uniform = np.random.rand(3, 3)
print('0과 1 사이에 균일 분포:\n', random_uniform)

rounded = np.round(random_uniform, 2)
print('0~1 균일 분포(소수점 2자리):\n', rounded)

# 특정 범위의 균일 분포 (예: 10~20)
low, high = 10, 20
random_range = low + (high - low) * np.random.rand(3, 3)
print(f'{low}부터 {high}까지 균일 분포\n', random_range)

uniform = np.random.uniform(low=0, high=100, size=(2, 3))
print(f'0부터 100까지 균일 분포\n', uniform)

# 정규 분포 난수
# 표준 정규 분포 (평균 0 표준편차 1)
random_normal1 = np.random.randn(3, 3)
print('표준 정규 분포\n', random_normal1)
print()

# 평균 100, 표준편차 15인 정규분포
mean, std = 100, 15
scores = mean + std * np.random.randn(3)
print('표준 정규 분포\n', scores)

mean, std = 100, 15
scores = mean + std * np.random.randn(1000)
print('표준 정규 분포\n', scores[:10])
print('실제 평균\n', scores.mean())
print('실제 표준편차\n', scores.std())

# 정수 난수
# 0~9사이 정수 난수
random_int = np.random.randint(0, 10, size=(3, 4))
print('0~9 정수 난수\n', random_int)

# 주사위 시뮬레이션(1~6)
dice = np.random.randint(1, 7, size=10)
print('주사위 10번 굴리기\n', dice)

# 시드 설정(재현 가능한 난수) --- 같은 난수 생성
np.random.seed(42)
random1 = np.random.rand(5)
print('첫 번째 난수:', random1)

np.random.seed(42)
random2 = np.random.rand(5)
print('두 번째 난수:', random2)
print('같은가?', np.array_equal(random1, random2))
print('같은가?', np.array_equal(random1, matrix))

# 새로운 방식(권장)
rng = np.random.default_rng(seed=42)
random3 = rng.random((2, 3))
print('새로운 방식 난수:\n', random3)

# =========================================
# 배열 인덱싱, 슬라이스
arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
print('원본 배열:\n', arr)

# 팬시 인덱스(Fancy indexing)
indices = [1, 3, 1, 7, 4, 7]
# print('dlseprtm [1, 4, 7] 선택: \n', arr[indices])
print('dlseprtm [1, 3, 1, 7,  4, 7] 선택: \n', arr[indices])

# 양수 인덱스 (0부터 시작)
print('첫 번째 요소 (인덱스 0)', arr[0])
print('두 번째 요소 (인덱스 2)', arr[2])
print('여덟 번째 요소 (인덱스 8)', arr[8])
# print('열 번째 요소 (인덱스 10)', arr[10]) : 인덱스 에러

arr[0] = 100
print('수정 후 배열 arr:\n', arr)


# 음수 인덱스 (뒤에서부터)
print('마지막 요소 (인덱스 -1)', arr[0])
print('-2번째 요소 (인덱스 -2)', arr[-2])
print('-8번재 요소 (인덱스 -8)', arr[-8])

arr[0] = 100  # 수정 400인데 몇 번째더라
print('수정 후 배열 arr:\n', arr)

# 배열 슬라이싱
arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
print('원본 배열:\n', arr)

print('인덱스 2부터 5까지 (5제외): \n', arr[2:5])
print('인덱스 처음부터 4까지 (4제외): \n', arr[:4])
print('인덱스 3부터 마지막까지: \n', arr[3:])
print('짝수 인덱스: \n', arr[::2])
print('홀수 인덱스: \n', arr[1::2])

# 슬라이싱으로 값 수정
arr[2:5] = 100
print('수정 후 배열 arr:\n', arr)

arr[2:5] = [10, 20, 30]  # 개수 맞춰야 됨
print('수정 후 배열 arr:\n', arr)

'''
일반 리스트에서 불가(에러발생)
new_list = [1, 2, 3, 4, 5]
new_list[2:4] = 40
print(new_list)
'''

# view (기존에 영향)
original = np.array([1, 2, 3, 4, 5])
view = original[1:4]
print('original \n', original)
print('view \n', view)

view[0] = 10
print('original \n', original)
print('view \n', view)
print()

view[1:] = 20
print('original \n', original)
print('view \n', view)
print()

# 독립적인 복사본 필요한 경우
original = np.array([1, 2, 3, 4, 5])
copy = original[1:4].copy()

copy[0] = 100
print('original \n', original)
print('copy \n', copy)

# 2차원 배열
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print('2차원 배열\n', matrix)
print('0, 0 요소 :', matrix[0, 0])
print('2, 2 요소 :', matrix[2, 2])
print('1, 2 요소 :', matrix[1, 2])
print('1, 2 요소 :', matrix[1][2])
# print('3, 0 요소 :', matrix[3, 0]) 에러 발생
print('-1, -2 요소 :', matrix[-1, -2])
print('-1, -2 요소 :', matrix[-1][-2])

print('첫 번째 행: ', matrix[0])
print('두 번째 행: ', matrix[1])

print('여러 행: \n', matrix[:2])
print()

# 부분 행렬 추출
matrix = np.array([
    [1,  2,  3,  4,  5],
    [6,  7,  8,  9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20]
])

print('matrix[:2, 1:]: \n', matrix[:2, 1:])
print('matrix[1:3, 1:4]: \n', matrix[1:3, 1:4])
print('matrix[::2, ::2]: \n', matrix[::2, ::2])

# 특정 행들 선택
row_indices = [0, 2, 3]
print('[0, 2, 3]행 선택:\n', matrix[row_indices])

# 특정 요소들 선택 (행, 열 인덱스)
row_indices = [0, 2, 2]
col_indices = [3, 2, 3]
print('특정 요소들 선택:\n', matrix[row_indices, col_indices])
print()

# 불리언형
arr = np.array([1, 5, 4, 7, 2, 3])
print('4이상', arr[arr > 4])
print('2미만, 4이상', arr[(arr > 4) | (arr < 2)])
# (2 <= arr <= 4) 파이썬은 가능하나 np는 불가 + & 사용 (and X)
print('2이상, 4이하', arr[(2 <= arr) & (arr <= 4)])
print()

matrix = np.array([
    [1,  2,  3,  4,  5],
    [6,  7,  8,  9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20]
])
print('원본 행렬\n', matrix)
print('9보다 큰 요소들 \n', matrix[matrix > 9])
print('첫 번째 열이 4 이상인 행들 \n', matrix[matrix[:, 0] >= 4])

matrix[matrix > 9] = 10
print('수정된 행렬\n', matrix)
