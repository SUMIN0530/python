# 실습 01. 사칙연산 계산기 함수 만들기
# 문제 1. 사칙연산 계산기 함수 만들기
def calculate(a, b, operator):
    result = 0
    if operator == "+":
        result = a + b
    elif operator == "-":
        result = a - b
    elif operator == "*":
        result = a * b
    elif operator == "/":
        result = a / b
    else:
        result = '지원하지 않는 연산입니다.'
    return result


print(calculate(2, 3, '+'))
print(calculate(2, 3, '-'))
print(calculate(2, 3, '*'))
print(calculate(2, 3, '/'))
print(calculate(2, 3, '//'))

# ==============================================
# 실습 02. 가변인자 연습하기
# *args 연습
# 문제 1. 숫자를 받아 평균 구하기


def average(*args):
    return sum(args) / len(args)


print(average(85, 65, 60, 75, 40))

# 문제 2. 가장 긴 문자열 찾기 (max 함수 찾아보고 풀기)


def max_len_word(*args):
    # max함수 : max(매개변수, key = ____)
    return max(args, key=len)


print(max_len_word('rkawk', 'rhrnak', 'djacjdrsldhdl', 'didvk', 'ghqkr'))

# **kwargs 연습
# 문제 3. 사용자 정보 출력 함수


def info(**kwargs):
    for key, value in kwargs.items():
        print(f'{key} {value}')


info(name='홍길동', age=25, email='girdong@gmail.com')
print()
info(name='둘리', age=7, email='dully@gmail.com')
print()


# 문제 4. 할인 계산기
def price(**kwargs):
    for key, value in kwargs.items():
        print(f'{key} {value * 0.9}')


price(apple=20, peach=30, orenge=15)

# ==============================================
# 실습 03. 전역 변수 연습하기

current_user = ''


def login(ID):
    global current_user
    if current_user == '':
        current_user = ID
        print(f'{ID}님이 로그인 하셨습니다.')
    else:
        print('이미 로그인되어 있습니다.')


def logout():
    global current_user
    current_user = ''
    print('로그아웃 하셨습니다.')


login('rkawk')
login('qhrtbd')
logout()
login('qhrtbd')
print(current_user)

# ============================================
# 실습 04. 거듭제곱
# a**b = a * (a**(b-1))


def sq(a, b):
    if b == 0:
        return 1
    return a * sq(a, b-1)


print(sq(2, 7))

# ==========================================
# 실습 05. 팩토리얼(Factorial)
