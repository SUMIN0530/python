# 실습 01. 배열 연산
# 문제 1. 다음 배열을 생성하고, 모든 요소에 3을 더하세요.
import numpy as np
arr = np.array([1, 2, 3, 4])
scalar = 3
''' result = arr + 3 '''
print('문제 1:', arr + scalar)

# 문제 2. 아래 2차원 배열에서 각 요소를 -1로 곱한 새로운 배열을 만드세요.
arr = np.array([
    [5, 10],
    [15, 20]
])
scalar = -1
''' result = arr * -1 '''
arr_copy = arr.copy()
arr1 = arr_copy*scalar
print(arr)
''' print(result) '''
print('새로운 배열 생성:\n', arr1)

# 문제 3. 아래 두 배열의 요소별 곱셈과 나눗셈 결과를 각각 출력하세요.
arr1 = np.array([2, 4, 6])
arr2 = np.array([1, 2, 3])
print('곱셈 :\n', arr1 * arr2)
print('나눗셈 :\n', arr1 / arr2)
print()

# 문제 4. 아래 배열에서 모든 요소를 최대값 100으로 만들기 위해 필요한 값을 더한 결과 배열을 브로드 캐스팅으로 만드세요.
arr = np.array([[95, 97],
                [80, 85]])
arr1 = np.array([[5, 3],
                [20, 15]])
'''
add_val = 100 - arr
print(add_val) => arr1 값이 출력 
=> result = arr + add_val
   print(result)
'''
print(arr)
print(arr1)
print('최대값 100:\n', arr + arr1)

# 문제 5. 아래 2차원 배열에서 각 행에 다른 값을 곱하여 새로운 배열을 만드세요.(브로드캐스팅 이용)
arr = np.array([[1, 2, 3], [4, 5, 6]])
# 2차원 배열 생성
mul_val = np.array([[10], [100]])
result = arr * mul_val
print('1행 X10, 2행 X100\n', result)

# 문제 6. 아래 배열에서 각 행마다 다른 스칼라 값을 더하기 위해 1차원 배열을 만들어 브로드캐스팅 연산을 수행하세요.
# 첫 번째 행에 100, 두 번째 행에 200, 세 번째 행에 300을 더하세요.
arr = np.array([[10, 20],
                [30, 40],
                [50, 60]])

add_val = np.array([100, 200, 300])
reshaped = add_val.reshape(-1, 1)     # 다른 크루원 문제 풀이도 보기.
result = arr + reshaped
print(result)
# =================================================

# 실습 03. 논리 연산과 조건 연산
# 문제 1. 1차원 배열 [5, 12, 18, 7, 30, 25]에서 10보다 크고 20보다 작은 값만 필터링하세요.
arr = np.array([5, 12, 18, 7, 30, 25])
print('10보다 크고 20보다 작은 값:\n', arr[(10 < arr) & (arr < 20)])

# 문제 2. 배열 [10, 15, 20, 25, 30, 35]에서 15 이하이거나 30 이상인 값만 선택하세요.
arr = np.array([10, 15, 20, 25, 30, 35])
print('15 이하, 30 이상인 값:\n', arr[(30 <= arr) | (arr <= 15)])

# 문제 3. 배열 [3, 8, 15, 6, 2, 20]에서 10 이상인 값을 모두 0으로 변경하세요.
arr = np.array([3, 8, 15, 6, 2, 20])
arr[10 <= arr] = 0
print('10 이상인 값은 모두 0으로 표기: \n', arr)

# 문제 4. 배열 [7, 14, 21, 28, 35]에서 20 이상인 값은 "High", 나머지는 "Low"로 표시하는 새로운 배열을 생성하세요.
arr = np.array([7, 14, 21, 28, 35])
answer = np.where(arr >= 20, 'High', 'Low')
print(answer)

'''
# 삼항 연산자
# 참일때 값 if 조건 else 거짓일 때 값
num = 7
result = "짝수" if num % 2 == 0 else "홀수"
a, b = 10, 20
max_value = a if a > b else b
print(result)
print(max_value)
'''

# 문제 5. 0~9 범위의 배열에서 짝수는 그대로 두고, 홀수는 홀수 값 × 10으로 변환한 배열을 만드세요.
arr = np.arange(10)
result = np.where(arr % 2 == 0, arr, arr * 10)
print('0~9범위에서 홀수만 (X10):\n', result)

# 문제 6.아래 2차원 배열 에서 20 이상 40 이하인 값만 선택하세요.
arr = np.array([
    [10, 25, 30],
    [40, 5, 15],
    [20, 35, 50]
])
result = arr[(arr >= 20) & (arr <= 40)]
print('20 이상, 40 이하의 수\n', result)

# 문제 7. 배열 [1, 2, 3, 4, 5, 6]에서 3의 배수가 아닌 값만 선택하세요.
arr = np.array([1, 2, 3, 4, 5, 6])
print(arr % 3 != 0)  # mask T F 판별
print('3의 배수가 아닌 값\n', arr[arr % 3 != 0])

# 문제 8. 랜덤 정수(0~100) 10개 배열에서 아래와 같이 새로운 배열을 만드세요.
# 50 이상인 값은 그대로
# 50 미만인 값은 50으로 변경
arr = np.random.randint(0, 101, 10)
result = np.where(arr >= 50, arr, 50)
print(arr)
print('50 이상 유지, 미만->50으로 변경\n', result)

# 문제 9. 2차원 배열에서 아래와 같이 분류된 문자열 배열을 생성하세요.
arr = np.array([
    [[5, 50, 95],
     [20, 75, 10],
     [60, 30, 85]]
])

result = np.where(arr >= 70, 'A', np.where(arr >= 30, 'B', 'C'))
print(result)
