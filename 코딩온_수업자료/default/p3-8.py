"""
x = 10
y = 20

print(f'x == y : {x == y}')
print(f'x != y : {x != y}')
print(f'x > y : {x > y}')
print(f'x < y : {x < y}')
print(f'x >= y : {x >= y}')
print(f'x <= y : {x <= y}')

x = 15
y = 15
print(f'x >= y : {x >= y}')  # True
print(f'x <= y : {x <= y}')  # True

# 논리 연산자
print(True and True)  # True
print(True and False)  # False
print(False and True)  # False
print(False and False)  # False

print(True or True)  # True
print(True or False)  # True
print(False or True)  # True
print(False or False)  # False

print(f'not True : {not True}')----뭔소리야
print(f'not False : {not False}')

print(True and False or True)  # or로 인해 True  사칙연산과 비슷하게 계산
print(True and (False or True) and False)  # False

# and or에 따라 아에 출력이 안된 것도 있음.
if True and print('단축평가'):    # 실행 안 됨
    print('실행')

if False and print('단축평가'):   # 실행 안 됨
    print('실행')

if True or print("단축평가"):
    print("실행")

if False or print("단축평가") or True:
    print("실행")


a = 10
if a == 10:
    print(f'a: {a}')
    print('if문 블럭 안')
print('if문 블럭 밖')

if a != 10:
    print(f'a: {a}')
    print('if문 블럭 안')
print('if문 블럭 밖')



age = 20
if age >= 18:
    print("성인입니다.")

name = "이수민"
name = ""
if name:
    print("이름이 존재합니다.")  # name을 빈 문자열로 재할당 => 문자 출력 X

if True:
    print('무조건 실행')

if False:
    print('실행되지 않습니다.')

if True:
    pass  # 다음에 작성하겠습니다.
print('조건문과 상관없습니다.')  # pass가 없으면 들여쓰기가 필요하다는 도움말 뜸

name = ''
if name:
    print(f'이름은: {name}')
else:
    print('이름을 작성해주세요.')

if True:
    print('if 실행')
else:
    print('else 실행')

if False:
    print('if 실행')
else:
    print('else 실행')
"""

# 아래에서 오류 2개 발생 (찾아야됨.)
'''
# 뭐를 많이 놓침 아니 왤케 빠르게 넘어감 github에 올려주심
name = '김철수'

if name == '김철수':
    print(f'김철수 입니다.')
elif name == '철수':
    print(f'철수 입니다.')
else:
    print('이름을 입력해주세요.')  # ?

name = '김철수'
age = 20

if age > 20:
    print('성인입니다.')
else:
    print('미성년자 입니다.')

if grade > 3:
    print('고학년 입니다.')
elif grade == 2:
    print('2학년 입니다.')


# 중첩 조건문

# for문 - 정해진 횟수만큼 반복
for i in range(5):  # for (변수) in range(n)
    print('hi')
# range(끝) - 0부터 끝-1까지
# range(5) - 0, 1, 2, 3, 4
# range(시작, 끝) - 시작부터 끝-1까지
# range(2, 6) - 2, 3, 4, 5
# range(시작, 끝, 간격) - 시작부터 끝-1까지 간격만큼
# range(2, 6, 2) - 2, 4

for i in range(5):
    print(f'i의 값 {i}')
print()

for i in range(2, 6):
    print(f'i의 값 {i}')
print()

for i in range(2, 6, 2):
    print(f'i의 값 {i}')
print()

# 리스트와 for문
# 과일 리스트 순회
fruits = ['사과', '바나나', '오렌지', '포도']
for fruit in fruits:
    print(f'과일 {fruit}')

# 점수
scores = [65, 27, 87, 86]
for score in scores:
    print(f'점수 : {score}')

# 총점
total = 0
count = 0
for score in scores:
    total += score
    count += 0  # ??
    print(f'점수 : {score}')

# 평균이랑 github

print('===========')

word = 'python'
for char in word:
    print(char, end=' ')

'''

# 별 패턴 1: 직각삼각형 ----- 중첩 for문 반복 속에 반복
# *
# **
# ***
# ****
# *****
for i in range(1, 6):
    for j in range(i):
        print('*', end=' ')
    print()
print()

# 정사각형 만들기  --- 중첩으로 안해도 됨
# *****
# *****
# *****
# *****
# *****
for i in range(1, 6):
    for j in range(1, 6):
        print('*', end=' ')
    print()
