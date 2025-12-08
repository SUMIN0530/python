# 루프 제어문
# break
for i in range(3):
    if i == 2:
        break       # for문을 종료한다.
    print(i)  # 0, 1
print()

for i in range(10):
    break
    print(i)  # 실행되지 않음
print('for문 종료')

for i in range(10):
    print(i)  # 0에서 종료됨.
    break
    print(i)  # 실행되지 않음
print('for문 종료')

# continuee
for i in range(10):   # 잘 안남옴
    if i % 2:  # 0을 만나면 실행이 안되므로(False) 짝수?
        continue
    print(i)
print()

# pass
for i in range(10):
    print(i)
    pass        # 종료가 아닌 넘어가기
print()

for i in range(10):
    pass
    print(i)     # 정상 작동
print()

# 이거 수업 내용이랑 다른 데 확인점
''' 이거 바바 '''
for i in range(10):
    print(i)
    if i == 4:
        break
else:
    print('루프 정상 종료')

for i in range(10):
    print(i)
    if i == 4:
        continue
else:
    print('루프 정상 종료')


colors = ['red', 'blue']
fruits = ['apple', 'banana']

for color in colors:
    for fruit in fruits:
        print(f'{color},{fruit}')

# comprehension
new_list = [(c, f) for c in colors for f in fruits]  # 위 예제

# ======================================================================
# 09.2
# while문
'''
while True:
    print('무한루프 생성')
'''
i = 1
while i <= 5:
    print('무한루프 생성')
    i += 1

print('반복문 종료')

# for문 vs while문
'''
for문 : 몇 번 반복할지 정해져 있을 때
while문 : 조건이 만족하는 동안 계속 반복

while 조건식 :
    반복할 코드
    (조건을 변경하는 코드) !!!!!!
'''
# 카운트 다운
count = 5

while count > 0:
    print(f'count {count}')
    count -= 1  # 중요!!! 조건 변경
print('while문 종료')

# 누적 합계 구하기
total = 0
num = 1

while num <= 100:
    total += num
    num += 1
print(f'1부터 100까지의 합{total}')

# while문으로 입력 검증하기
# 올바른 입력을 받을 때까지 반복
age = -1  # 초기값(무조건 반복 진입)

while age < 0 or age > 150:
    age = int(input('나이를 입력하세요(0-150): '))

    if age < 0 or age > 150:
        print('올바른 나이를 입력해주세요.')
print(f'입력된 나이: {age}세')

# 비밀번호 확인
correct_password = 'python123'
attempt = 0
max_attempts = 3

while attempt < max_attempts:
    password = input('비밀번호를 입력하세요:')
    attempt += 1

    if password == correct_password:
        print('로그인 성공')
        break   # 반복문 탈출
    else:
        remaining = max_attempts - attempt
        if remaining > 0:
            print(f'틀렸습니다. {remaining}번 남았습니다.')
        else:
            print('로그인 실패, 계정이 잠겼습니다.')

# 무한 루프와 break
while True:
    user_input = input('명령을 입력하세요(종료 q)')

    if user_input == 'q':
        print('프로그램을 종료합니다.')
        break
    print(f'입력한 명령: {user_input}')
    # 명령처리
    pass

# 계산기
while True:
    num1 = float(input('첫 번쨰 숫자:'))

    if num1 == 0:
        break

    num2 = float(input('두 번쨰 숫자:'))
    operator = input('연산자(+, -, *, /)')

    if operator == '+':
        result = num1 + num2
    elif operator == '-':
        result = num1 - num2
    elif operator == '*':
        result = num1 * num2
    elif operator == '/':
        if num2 != 0:
            result = num1 / num2

    # 놓쳤어용

# while - else
i = 10
while i < 15:
    print(i)
    i += 1
else:
    print('정상 종료 되었습니다.')

i = 10
while i < 15:
    print(i)
    if i <= 1:  # 여기 수정 필수
        print()
    else:
        print('정상 종료 되었습니다.')

# 이 위치가 맞는지 모르겠네
# 이중 for문 - 구구단
for i in range(2, 10):
    print(f'==={i}단===')
    for j in range(1, 10):
        print(f'{i} X {j} = {i * j}')
    print()
print()

# 이중 while문
i = 2  # 초기값
while i < 10:
    j = 1  # 초기값
    print(f'==={i}단===')
    while j < 10:
        print(f'{i} X {j} = {i * j}')
        j += 1  # 수 증가
    print()
    i += 1  # 단 증가
