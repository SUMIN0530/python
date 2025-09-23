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

# ==================================================================================
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

# 실습 03.4. 입력과 연산 연습하기
# 4-1. 가로 세로를 입력하여 넓이와 둘레 알기
가로 = int(input('가로 길이 = '))
세로 = int(input('세로 길이 = '))
print(f'넓이 = {가로*세로}\n둘래 = {(가로+세로)*2}')

# 4-2. 네 자리수 정수를 받고 각 자리마다 분리 출력
# 숫자 입력시 공백 필수 원인 .split() 없으면 공백 X
천, 백, 십, 일 = input('네 자리 정수를 입력하세요 : ').split()
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
name1, name2, name3 = input('각 조의 발표자 이름을 쓰시오 : ').split()  # split() : 입력창 구분용
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
