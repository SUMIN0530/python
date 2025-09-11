# 주석 (계산에 해당 X)

'''(""")
주석 
#과 다르게 여러줄 가능

1. 코드 설명
2. 나중에 봐도 이해할 수 있도록
3. 협업할 때 소통
(""")'''

# 변수 : 데이터 저장 공간
age = 20                    # = -> 할당 연산자 (넣는다는 의미, 등호와는 다른 의미)
student_name = '김철수'     # 변수는 재할당 가능
height = 175.5

x, y = 10, 20               # 한 줄에 여러 변수 대입 가능
x, y = y, x                 # 값의 교환
# x, y = 20, 10 -> 재할당

# sep= 여러값 구분자 / end= 줄바꿈 구분자(\n : 기본설정)
X, Y = 30, 'a'
print('X', X, sep='/', end='\n')  # end='-' : - 로 줄 이어짐 / 근데 end 안되는거 같은디?
print('Y', Y)

# 나쁜 변수
a = 24  # 정보가 무엇을 의미하는지 한번에 파악할 수 없음.
na = "김철수"  # 정확한 변수를 이용 => na(X), name(O)
# my-name = '김철수'  # 불가능한 변수 / 연산이랑 혼동

# 변수명 규칙
'''
1. 영문자, 숫자, 언스스코어(_)만 사용
2. 숫자로 시작 불가
3. 공백 사용 불가
4. 의미있는 이름 사용
5. 대소문자 구분
6. 예약어 사용불가
'''

# 스테이크 케이스(snake_case)
user_name = '김철수'
user_age = 25

# 파스칼 케이스(Pascal Case)
UserName = '김철수'
UserAge = 25

# 카멜 케이스(camel Case)
userName = '김철수'
userAge = 25

# 자료형
'''
1. 정수(int): 소수점이 없는 숫자 → 2, 3, 12, 25, -10
2. 실수(float): 소수점이 있는 숫자 → 1.1, 41.123, 3.1415
3. 문자(char): 한 글자 → 'a', 'B', '가' (Python에서는 길이 1인 문자열)
4. 문자열(str): 여러 글자의 조합 → "hello", "안녕하세요"                  # ' ' or " "사용
5. 불린(bool): 참/거짓 값 → True, False                                 # 첫 글자는 대문자
6. 시퀸스형 : list, tuple, range
7. type : 자료 확인형 함수
'''
# 자료형 확인
blue_roses_do_exist = True  # 파란 장미는 실존한다.
blue_roses_do_not_exist = False  # 파란 장미는 실존하지 않는다.

print(f'파란_장미는_실존한다. {blue_roses_do_exist}', type(blue_roses_do_exist))
print(f'파란_장미는_실존하지_않는다. {blue_roses_do_not_exist}',
      type(blue_roses_do_not_exist))

# 형 변환
'''
int() : 숫자, 문자열 => 정수형
float() : 숫자, 문자열 => 실수형
str() : 모든  값 => 문자열
'''
a = '1'
b = "1"
a1 = int(a)  # 형 변환
b1 = float(b)
print(a)  # 숫자형으로 오해가능
print(b1)  # float 경우 소수점이 붙는다.
print('a type: ', type(a))
print('a1 type: ', type(a1))
print('b1 type: ', type(b1))

# 문자열 포매팅 f-string  --- 문자열 앞에 f, {변수}
c = 2
d = 2.1
print(f'c의 숫자는 {c}입니다.')
print(f'c의 숫자는 {c}, d의 숫자는 {d}입니다.')
