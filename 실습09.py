# 실습 01. for문 기본 문제
# 문제 1. 리스트 값 두 배로 변환하기
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
