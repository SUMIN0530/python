# 실습 01. for문 기본 문제
# 문제 1. 리스트 값 두 배로 변환하기
import random
numbers = [3, 6, 1, 8, 4]
num = []
for i in numbers:
    num.append(i * 2)  # [i]를 사용할 경우 그 항의 값X 항 자체를 2개로 함.
print(num)

# 문제 2. 문자열의 길이 구해서 새 리스트 만들기
words = ["apple", "banana", "kiwi", "grape"]
new = []
for i in words:                 # len_words = [len(word) for word in words]
    new.append(len(i))          # print(len_words)
print(new)

# 문제 3. 좌표 튜플에서 x, y 좌표 나누기
coordinates = [(1, 2), (3, 4), (5, 6), (7, 8)]
x_values = []               # x_values = [x for x, y in coordinates]
y_values = []               # y_values = [y for x, y in coordinates]
for x, y in coordinates:
    x_values.append(x)
    y_values.append(y)
print(f'x: {x_values}, y: {y_values}')

# 실습 02. for문과 range
# 문제 1. 입력받은 수의 합 구하기
sum_num = 0
n = int(input('정수를 입력하세요: '))  # int 없으면 range에서 계산 X
for i in range(1, n+1):
    sum_num += i
print('입력 받은 수 이하의 수의 합: ', sum_num)

# 문제 2. 정수를 입력받아 구구단 출력
n = input('정수를 입력하세요: ')
n = int(n)                      # n을 정수로 형 변형을 안하면 트리만들기 처럼 출력됨.
for i in range(1, 10):          # n = int(input('정수를 입력하세요: '))
    print(f'{n}X{i} = {n*i}')

# 문제 3. 3의 배수의 합 구하기
sum = 0
for i in range(3, 101, 3):
    sum += i
print('1부터 100 사이의 3의 배수의 합: ', sum)

# 문제 4. 짝수이면서 5의 배수 출력
n = int(input('정수를 입력하세요: '))
for i in range(1, n+1):
    if (not i % 2) and (not i % 5):         # 짝수 : i % 2 == 0 / 0 = False / 5의 배수도 동일
        print(i)                            # not 0 => True

# 실습 03. 중첩 for문 연습
# 문제 1. 구구단 만들기(2)
for i in range(2, 10):
    print(f'[{i}단]')
    for j in range(1, 10):
        print(f'{i}x{j} = {i * j}')
    print()

# 문제 2. 중첩 for문 * 찍기
n = int(input('몇 줄 작성? > '))

print('[왼쪽 정렬]')
for i in range(1, n+1):
    for j in range(i):
        print('*', end='')
    print()
# ------------------------------
print('[가운데 정렬]')
for i in range(1, n+1):
    for j in range(n - i):
        print(' ', end='')
    # (i-1) * 2 + 1 = 2i - 1
    for k in range(2*i - 1):
        print('*', end='')
    print()
# ------------------------------
print('[오른쪽 정렬]')
for i in range(1, n+1):
    for j in range(n - i):
        print(' ', end='')
    # (i-1) * 2 + 1 = 2i - 1
    for k in range(i):
        print('*', end='')
    print()

# 실습 04. 리스트 컴프리헨션 연습문제
# 문제 1. 제곱값 리스트 만들기
sq = [x**2 for x in range(1, 11)]
print(sq)

# 문제 2. 3의 배수만 리스트로 만들기
time = [y for y in range(1, 51) if y % 3 == 0]
print(time)

# 문제 3. 문자열 리스트에서 길이가 5 이상인 단어만 뽑기
fruits = ["apple", "fig", "banana", "plum", "cherry", "pear", "orange"]
f = [word for word in fruits if len(word) >= 5]
print(f)

# =========================09.2 실습문제=====================================
# 실습 01. while문 연습 문제
# 문제 1. 비밀 코드 맞추기
secret_code = 'codingonre3'
# ' secret_code = input('비밀 코드를 입력하세요: ')
while (secret_code != 'codingonre3'):  # while secret_code != input(#')
    print('비밀 코드가 틀렸습니다.')
    secret_code = input('비밀 코드 다시 입력하세요: ')
print('입장이 허용되었습니다.')

# 문제 2. 업다운 게임
# import random 필요 (저장하면 자동으로 젤 위로 올라감)
random_value = 0
n = 0
random_value = random.randrange(1, 101)
m = int(input('숫자를 맞춰보세요.'))
while m > random_value:
    print('입력한 숫자보다 작아요')
    m = int(input('숫자를 다시 입력해보세요.'))
    n += 1
while m < random_value:
    print('입력한 숫자보다 커요')
    m = int(input('숫자를 다시 입력해보세요.'))
    n += 1
입력횟수 = n + 1
print(f'{입력횟수}만에 맞췄어요.')
'''
리더님
import random
random_value = random.randrange(1, 101)
n = 0   # 초기값
count = 1

while random_value != n:
    n = int(input('숫자를 입력하세요.'))
    count += 1

    if n > random_value:
        print('입력한 숫자보다 커요')
    else:
        print('입력한 숫자보다 작아요')

print(f'{count}번 만에 정답을 맞췄습니다.)
'''
# 실습 02. while문 연습 문제(2)
# 문제 1. 비밀 코드 맞추기(2)
secret_code = 'codingonre3'
code = input('비밀 코드를 입력하세요: ')
while code != secret_code:
    if code == secret_code:
        print('입장 완료! 환영합니다.')
        break
    print('비밀코드가 틀렸습니다.')
    code = input('비밀 코드를 다시 입력하세요: ')

# 유효한 나이만 평균 내기
times = 0
sum_age = 0
while times != 5:
    age = int(input('나이를 입력하세요:'))
    if 0 < age < 120:
        times += 1
        sum_age += age
    else:
        continue
a = sum_age / 5
print(f'총 합: {sum_age}, 평균: {a}')
