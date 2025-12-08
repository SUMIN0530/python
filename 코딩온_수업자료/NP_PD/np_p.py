
import numpy as np
"""
# 실습 01. 배열 초기화 및 생성
# 문제 1. 0으로 채워진 크기 (3, 4) 배열을 생성한 후, 모든 값을 5로 채우는 새로운 배열을 만드세요.

''' 0으로 채운 배열 '''
zeros_1d = np.zeros((3, 4), dtype=int)
print('zeros:\n', zeros_1d)
''' 모든 값 5로 변경'''
zeros_1d = np.full((3, 4), 5)
print('2차원 full:\n', zeros_1d)
print()
'''
리더님
arr1 = np.zeros((3, 4)) + 5
print(arr1)
'''

# 문제 2. 0부터 20까지 2씩 증가하는 1차원 배열을 생성하세요.
arr2 = np.arange(0, 21, 2)
print(arr2)

# 문제 3. 0~1 사이의 실수 난수를 가지는 (2, 3) 크기의 배열을 생성하세요.
arr3 = np.random.rand(2, 3)
print('0~9 실수 난수\n', arr3)

# 문제 4. 평균이 100, 표준편차가 20인 정규분포 난수 6개를 생성하세요.
mean, std = 100, 20
scores = mean + std * np.random.randn(6)
print('표준 정규 분포\n', scores)
'''
리더님
# normal(평균, 표준편차, 사이즈)
arr4 = np.random.normal(100, 20, 6)
print(arr4)
'''

# 문제 5. 1부터 20까지의 정수를 포함하는 1차원 배열을 만들고,
#          이 배열을 (4, 5) 크기의 2차원 배열로 변환하세요
arr5 = np.arange(1, 21)
print('1~20까지 1차원 배열', arr5)
'''
리더님
arr5 = np.arange(1, 21).reshape(4, 5) # (4, 5)의 개수에 맞는 요소 수(20개) 필요
print(arr5)

arr6 = np.linspace(0, 1, 12).reshape(3, 4)
print(arr6)

arr7 = np.random.randint(0, 100, (10, 10))
arr7 = arr7 + np.eye(10) # 대각선에 1씩 더해짐
print(arr7)

arr8 = np.random.randint(0, 10, (2, 3, 4))
print(arr8)

''' / 
# =====================================================
# 실습 02. 인덱싱과 슬라이싱
# 문제 1. 다음 배열에서 2, 4, 6번째 요소를 Fancy Indexing으로 선택하세요.
arr = np.arange(10, 30, 2)
idx = [1, 3, 5]
print('2, 4, 6번쨰 요소:', arr[idx])
'''리더님 print(arr[[1, 3, 5]])'''

# 문제 2. 3x3 배열에서 왼쪽 위 → 오른쪽 아래 대각선의 요소만 인덱싱으로 추출하세요.
arr = np.arange(1, 10).reshape(3, 3)
print('대각선 요소(1, 5, 9): ', arr[0, 0], arr[1, 1], arr[2, 2])
'''print(arr[[0, 1, 2], [0, 1, 2]])'''
print()

# 문제 3. 3x4 배열에서 마지막 열만 선택해 모두 -1로 변경하세요.
arr = np.arange(1, 13).reshape(3, 4)
arr[-1] = -1
print('마지막 열 -1로 변환: \n', arr)  # 내가한건 행 변환이었다 ㅋㅋㅋㅋ
'''arr[:,-1] = -1 / print(arr)'''

# 문제 4. 4x4 배열에서 행을 역순, 열을 역순으로 각각 슬라이싱해 출력하세요.
arr = np.arange(1, 17).reshape(4, 4)
print('행 뒤집기:\n', arr[::-1])
print('열 뒤집기:\n', arr[:, ::-1])

# 문제 5. 4x5 배열에서 가운데 2x3 부분을 슬라이싱한 뒤 copy()를 이용해 독립 배열을 만드세요.
arr = np.arange(1, 21).reshape(4, 5)
arr_cp = arr[1:3, 1:4].copy()
print('arr:\n', arr)
print('arr_cp:\n', arr_cp)
print()

# 문제 6. 3x4 배열에서 짝수이면서 10 이상인 값만 선택하세요.(&을 활용)
arr = np.array([[4, 9, 12, 7], [10, 15, 18, 3], [2, 14, 6, 20]])
print('10 이상인 짝수:\n', arr[(arr % 2 == 0) & (arr >= 10)])
print()

# 문제 7. 5x5 배열에서 2, 4번째 행을 선택하고, 선택된 행에서 열 순서를 [4, 0, 2]로 재배치하세요.
arr = np.arange(1, 26).reshape(5, 5)
arr_cp = arr[1], arr[3].copy()
arr_t = arr[[1, 3]][:, [4, 0, 2]]
print('2, 4번째 배열:\n', arr_cp)
print('2, 4번째, 재배치:\n', arr_t)
'''--------------------------------------'''

# 문제 8. 5x3 배열에서 각 행의 첫 번째 값이 50 이상인 행만 Boolean Indexing으로 선택하세요.
arr = np.array([[10, 20, 30], [55, 65, 75], [
               40, 45, 50], [70, 80, 90], [15, 25, 35]])
print('첫 번째 열이 50 이상인 행들 \n', arr[arr[:, 0] >= 50])
print()

# 문제 9. 4x4 배열에서 (0,1), (1,3), (2,0), (3,2) 위치의 요소를 한 번에 선택하세요.
arr = np.arange(1, 17).reshape(4, 4)
print('(0,1), (1,3), (2,0), (3,2) 해당 위치 요소\n',
      arr[0, 1], arr[1, 3], arr[2, 0], arr[3, 2])
'''print(arr[(0, 1, 2, 3), (1, 3, 0, 2)])'''
print()

# 문제 10. 3차원 배열 (2, 3, 4)에서 모든 블록에서 두 번째 열만 추출해 새로운 2차원 배열 (2, 3)을 만드세요.
arr3d = np.arange(24).reshape(2, 3, 4)
arr = arr3d[:, :, 1]
'''
    [
        [
            [0, 1,  2,  3]
            [4, 5,  6,  7]
            [8, 9, 10, 11]
        ],
        [
            [12, 13, 14, 15]
            [16, 17, 18, 19]
            [20, 21, 22, 23]
        ]
    ]
'''
print(arr)
"""

# 실습 03. NumPy 종합 연습
# 문제 1. 0부터 24까지 정수를 가진 배열을 만들고, (5, 5) 배열로 변환한 뒤 가운데 행(3번째 행)과 가운데 열(3번째 열)을 각각 1차원 배열로 출력하세요.
arr = np.arange(25).reshape(5, 5)
print(arr)
print()
print('가운데 행:\n', arr[2])
print('가운데 열:\n', arr[:, 2])
print()

# 문제 2. 0~99 난수로 이루어진 (10, 10) 배열을 생성하고, 짝수 인덱스의 행만 선택하여 출력하세요
arr = np.random.randint(0, 100, size=(10, 10))
print('0~99 정수 난수\n', arr)
print('짝수 행 출력:\n', arr[1::2])
print()

# 문제 3. 0부터 49까지 정수를 가진 배열을 (5, 10) 배열로 변환한 후, 2행 3열부터 4행 7열까지의 부분 배열을 추출하세요.
arr = np.arange(50).reshape(5, 10)
print(arr)
print('2행 3열부터 4행 7열까지:\n', arr[1:4, 2:7])
print()

# 문제 5. 0~9 난수로 이루어진 (3, 4, 5) 3차원 배열을 생성하고, 두 번째 층에서 첫 번째 행과 마지막 열의 값을 출력하세요.
arr = np.random.randint(0, 10, size=(3, 4, 5))
print(arr)
print('두 번째 층의 첫 행과 마지막 열')

# 문제 9. 1부터 50까지의 난수로 된 5x6 배열을 만들고, 배열에서 짝수만 선택하여 출력하는 코드를 작성하세요.
arr = np.random.randint(1, 51, size=(5, 6))
print('1부터 50까지 중 짝수만:\n', arr[arr % 2 == 0])
print()

# 문제 11. 0~9 난수로 이루어진 1차원 배열(길이 15)을 생성하고, 짝수 인덱스 위치에 있는 값들 중에서 5 이상인 값만 선택해 출력하세요.
arr = np.random.randint(0, 10, size=(15))
print(arr)
print('5 이상의 짝수만: ', arr[(arr % 2 == 0) & (arr >= 5)])
