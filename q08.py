# 실습 08.1 날씨에 따른 준비물 안내
날씨 = input('날씨를 입력하세요.')  # 단어 입력시 if 문장 ''필수 / input : 문자열

if 날씨 == '비':
    print("우산을 챙기세요.")
if 날씨 == '맑음':
    print("선크림을 바르세요.")

age = int(input('나이'))    # 숫자 입력 -> ''불필요 / 정수로 형태 변환(int) 필요
if age > 16:
    print('고등학생')
if age <= 16:
    print('중학생')

# 실습 8.2 짝수 홀수 판별하기
num = int(input('숫자를 입력하세요.'))
if num % 2 == 0:              # if num % 2: 홀수입니다.
    print('짝수 입니다.')
else:
    print('홀수 입니다.')

# 8.3 나이에 따른 영화 관람 가능 여부
age = int(input('나이를 입력하세요.'))
if 0 <= age <= 12:
    print('전체관람가')
elif 13 <= age <= 15:
    print('12세 이상 관람가')
elif 16 <= age <= 18:
    print('15세 이상 관람가')
else:
    print('청소년 관람불가 가능')
# ========================================
'''
# 실습 8.4 시, 분, 초 구하기
num = int(input('정수을 입력해주세요: '))
if num < 60:
    print(f'{num}초')
elif 60 <= num < 3600:
    min = num // 60
    sec = num % 60
    print(f'{min}분 {sec}초')
else:
    hour = num // 3600
    min = num % 3600 // 60
    sec = num % 60
    print(f'{hour}시 {min}분 {sec}초')

# 강사님
sec = int(input('초 입력 : '))

if sec >= 3600:
    print(f'{sec // 3600} 시', end=' ')  # 시간 구하기
sec %= 3600  # 시간을 뺀 초 / 이게 없으면 문제가 생길까?

if sec >= 60:
    print(f'{sec // 60} 분', end=' ')
sec %= 60

if sec < 60:
    print(f'{sec} 초')
'''

# 오류있다하네
'''
# 실습 8.5
money = int(input('소지하신 금액을 입력하세요: '))
food = input('구매할 식품을 입력하세요. ')

if money


김밥 = 2500
삼각김밥 = 1500
도시락 = 4000
if money >= food:
    print('구매 완료 했습니다.')
else:
    print('금액이 부족합니다.')

# 강사님
김밥 = 2500
삼각김밥 = 1500
도시락 = 4000
money = int(input('소지하신 금액을 입력하세요: '))
food = input('구매할 식품을 입력하세요: (김밥, 삼각김밥, 도시락)')

if food == '김밥':
    if money >= 김밥:
        print('구매 성공')
    else:
        print('구매 실패')
else:
    print('잘못된 메뉴입니다.')
'''
"""
# 구구단 4단 만들기
for i in range(1, 10):
    print(f'4 X {i} = {4*i}')
print()

# 구구단 전체 나타내기
for num in range(1, 10):
    print(f'==={num} 단===')
    for i in range(1, 10):
        print(f'{num} X {i} = {num*i}')
    print()
"""
