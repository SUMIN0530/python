# 실습 01. 계산기 모듈 만들어보기
import time
import os
import sys
import datetime
import random
import math
import calc

print(calc.add(10, 5))
print(calc.subtract(10, 5))
print(calc.mulriply(10, 5))
print(calc.divide(10, 5))
print(calc.divide(10, 0))

# 실습 02. math 모듈 사용해보기
# 문제 1. 실제 거리 계산 : 좌표 두 점 사이 거리 구하기
# import math
x1, y1 = int(input('x1:')), int(input('y1:'))
x2, y2 = int(input('x2:')), int(input('y2:'))

a = math.sqrt(math.pow((x2-x1), 2) + math.pow((y2-y1), 2))
print(f'두 좌표의 거리: {a}')

# 문제 2. 상품 나누기 : 최소공배수와 최대공약수
# import math
s = 18
t = 24
M = math.gcd(s, t)
m = (s * t) / math.gcd(s, t)
print(f'{s}와 {t}의 최대공약수는 {M}')
print(f'{s}와 {t}의 최소공배수는 {m}')

# 실습 03. 로또 번호 뽑기
# import random
lotto = random.sample(range(1, 46), 6)
lotto.sort()
print(f'로또 추첨 번호: {lotto}')

'''
두 개의 차이
random.choice()
random.choices(seq, k= n) k= 필수 / samole은 k없이 바로 숫자 대입
'''

'''
# 실습 04. 가위바위보 게임 만들기
# import random
RPS = random.choices(('가위', '바위', '보'), k=1)  # sanple과 다르게 k = 필수
rps = input('가위, 바위, 보 중 하나를 택하세요: ')
if RPS == rps:
    print('무승부')
if rps == '가위':
    if RPS == '바위':
        print('입력자가 이겼습니다.')
    if RPS == '보':
        print('컴퓨터가 이겼습니다.')
if rps == '바위':
    if RPS == '보':
        print('입력자가 이겼습니다.')
    if RPS == '가위':
        print('컴퓨터가 이겼습니다.')
if rps == '보':
    if RPS == '가위':
        print('입력자가 이겼습니다.')
    if RPS == '바위':
        print('컴퓨터가 이겼습니다.')

# 실습 05. 다음 생일까지 남은 날짜 계산하기
b_day = input('생일을 입력하세요(월-일): ').split('-')
t_day = datetime.date.today()
bn_day = datetime.date(year=2026, month=6, day=29)
l_day = bn_day - t_day
print(b_day)
'''

# 실습 06. 타자 연습 게임 만들기
# 1. 영단어 리스트 중 무작위 단어 제시
# import time
en_list = ['moon', 'potato', 'sky', 'garlic']
test = random.choices(en_list, k=1)
ready = input('[타자 게임] 준비되면 엔터!')
start = time.time()
for i in range(10):
    while i == 10:
        print(f'문제: {i}')
        print(f'{test}')
        answer = input()
        if answer == test:
            print('통과 !!')
        else:
            print('다시')

for i in range(10):
    print(i)
    time.sleep(1)
end = time.time()
print('수행시간 : ' + str(end-start) + '초')


# ==================따라 쓰기======================
# sys 모듈
x = input("수 입력 : ")
n = int(x)

if n == 0:
    print('0으로 나눌 수 없습니다.')
    sys.exit(0)

result = 10 / n

print(result)

# os 모듈
# 1. 현재 작업 디렉터리 확인
print('현재 작업 디렉터리:', os.getcwd())

# 2. 새 폴더 생성 (이미 있으면 예외 발생 가능)
folder_name = 'sample_folder'
if not os.path.exists(folder_name):
    os.mkdir(folder_name)
    print(f'{folder_name} 폴더를 생성했습니다.')
else:
    print(f'{folder_name} 폴더가 이미 존재합니다.')

# 3. 현재 디렉터리 내 파일/촐더 목록 출력
print('현재 디렉터리 내 파일 및 폴더 목록')
print(os.listdir())
