# 실습 02.2 자기소개 하기
"""
name = '이수민'
age = 23
MBTI = 'ESFJ'

print('안녕하세요.', f'제 이름은 {name}이고,', f'{age}살 입니다.',
      f'제 MBTI는 {MBTI}에요.', end='\n')   # end 사용시 줄 바꿈 X

# 여쭤보기
X, Y = 30, 'a'
print('X', X, 'Y', Y, end='\n')


X, Y = 30, 'a'
print('X', X, sep='/', end='\n')  # end='-' : - 로 줄 이어짐 / 근데 end 안되는거 같은디?
print('Y', Y)
"""

x = 10
y = 20
print(f'x != y : {x != y}')
print(f'x > y : {x > y}')
print(f'x < y : {x < y}')
print(f'x >= y : {x >= y}')
print(f'x <= y : {x <= y}')
print(f'x == y : {x == y}')

print(f'not True : {not True}')
print(f'not False : {not False}')
