"""
# 실습 02.1 영화정보 출력하기.

Title = 'Inception'
Director = 'Christopher_Nolan'
Year = 2010
Genre = 'Sci-Fi'

print(f'Title: {Title} Director: {Director} Year: {Year} Genre: {Genre}')

# 실습 02.2 자기소개 하기

name = '이수민'
age = 23
MBTI = 'ESFJ'

print('안녕하세요.', f'제 이름은 {name}이고,', f'{age}살 입니다.',
      f'제 MBTI는 {MBTI}에요.', sep='\n')

# print('안녕하세요.', f'제 이름은 {name}이고,', f'{age}살 입니다.', f'제 MBTI는 {MBTI}에요.', end='\n')
# end 사용시 줄 바꿈 X

print(f'안녕하세요.\n제 이름은 {name}이고,\n{age}살 입니다.\n제 MBTI는 {MBTI}에요.')

print(f'''안녕하세요
제 이름은{name}이고,
{age}살 입니다.
제 MBTI는 {MBTI}에요.
''')
"""
"""
# 실습 03.1 대학생의 용돈 관리

a = 30  # 시작용돈 30만원
a -= 8  # 교재비 8만원
a -= 0.9*5  # 평일 밥값
a += 12  # 알바비 totla = 29.5만원
a += a*0.2  # 부모님 용돈 or a *= 1.2
a -= a/3  # 공과금 => 23.6 or a *= 2/3 => 23.59999998
print(a)

# 실습 03.2 EDM 리듬 트랙 만들기
intro = '둠칫'
drop = '두둠칫'
print(intro+drop*3)

# 실습 03.3 input 연습하기
name = input('이름을 입력하세요 : ')
age = input('나이를 입력하세요 : ')
print(f'안녕하세요. 저는{name}이고, {age}살입니다.')

# 실습 03.4 입력과 연산 연습하기
# 4-1. 가로 세로를 입력하여 넓이와 둘레 알기
가로 = int(input('가로 길이 = '))
세로 = int(input('세로 길이 = '))
print(f'넓이 = {가로*세로}\n둘래 = {(가로+세로)*2}')

# 4-2. 네 자리수 정수를 받고 각 자리마다 분리 출력
천, 백, 십, 일 = input('네 자리 정수를 입력하세요 : ').split()  # 숫자 입력시 공백 필수 원인 .split() 없으면 공백 X
print(f'천의 자리: {천}\n백의 자리: {백}\n십의 자리: {십}\n일의 자리: {일}')

# 추가 자료
num = int(input('네 자리수 입력 : '))
print(f'천의 자리: {num // 1000}')
num %= 1000                         # num 재할당
print(f'백의 자리: {num // 100}')
num %= 100
print(f'십의 자리: {num // 10}')
num %= 10
print(f'일의 자리: {num // 1}')

# 실습 03.5 발표 순서와 발표 주제 정하기
name1, name2, name3 = input('각 조의 발표자 이름을 쓰시오 : ').split() # split() : 입력창 구분용
sub1, sub2, sub3 = input('발표 주제를 적으시오 : ').split()
print(f'''재생에너지 발표 순서 안내입니다.'\n
1조 발표자: {name1} - 주제: {sub1}
2조 발표자: {name2} - 주제: {sub2}
3조 발표자: {name3} - 주제: {sub3}''')

# 실습 03.6 날짜와 시간
print('날짜를 입력해 주세요.')
Y, M, D = input().split()     # Y, M, D = input().split('.') 으로 작성하면 sep=(.) 필요없음
print('시간을 입력해 주세요.')                      # split() : 구분자
h, m, s = input().split()     # 마찬가지
print(Y, M, D, sep='.')
print(h, m, s, sep=':')
print(f'RE_4rh의 개강일은 {Y}년 {M}월 {D}일\n시작 시간은 {h}시 {m}분 {s}초입니다.')

# 실습 8.1 날씨에 따른 준비물 안내
날씨 = input('날씨를 입력하세요.')  # 단어 입력시 if 문장 ''필수

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
"""
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
