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

'''
numbers = [3, 6, 1, 8, 4]
for i in numbers:
    num = [i] * 2
print(num)  # 왜 [4, 4]가 나오지??


# 문제 2. 문자열의 길이 구해서 새 리스트 만들기
words = ["apple", "banana", "kiwi", "grape"]
for word in words:
    for i in range(len(word)):
        list[i]
print(list)  # 이건 왜 type이 나오지??


# 실습 02. for문과 range
# 문제 1. 입력 받은 수의 합 구하기
list1 = []
for i in range(1, 5):
    list.append(i)
print(list) # 아니 이거는 왜 또 안됨?

# 정수를 입력받아 구구단 출력
n = input('정수를 입력하세요: ')
for i in range(1, 10):
    print(n * i) # 이거는 트리 만들기랑 똑같더라??

# 실습 03. 중첩 for문 연습
# 문제 1. 구구단 만들기(2)
for i in range(2, 10):
    print(f'[{i}단]')
    for j in range(1, 10):
        print(f'{i}x{j} = {i * j}')
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
