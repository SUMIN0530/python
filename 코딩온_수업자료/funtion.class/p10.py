# 함수 (Function)
'''
함수는 특정 작업을 수행하는 코드의 묶음
한 번 정의하면 필요할 때마다 호출하여 재사용 가능

함수 사용 이유
 - 코드 재사용성
 - 모듈화 : 프로그램을 작은 단위로 나누어 관리
 - 가독성 : 코드 의도 명확함
 - 유지보수 용이 : 수정이 필요할 때 함수만 변경
 - 추상화 : 복잡한 로직을 단순한 인터페이스로 제공
'''
# 함수 사용 X
print('=' * 20)
print('첫 번째 섹션')
print('=' * 20)

print('=' * 20)
print('두 번째 섹션')
print('=' * 20)

# 함수 사용


def print_section(title):
    print('=' * 20)
    print(f'{title}섹션')
    print('=' * 20)


print_section('첫 번째')
print_section('두 번째')

# 함수 정의와 호출
'''
# 함수 정의(Defintion)
def 함수이름(매개변수):
    # 실행코드
    return 반환값

# 함수 호출(Call)
결과 = 함수이름(인자)
'''

# 사용자 정의 함수


def greet(name):
    print(f'Hello, {name}!')  # 함수 정의


greet('sumin')  # 함수 호출!! => 사용자 정의 함수 출력 가능
greet('jungi')
print("greet('jungi')", greet('jungi'))  # None 출력


def say_hello():
    print('안녕하세요!')


say_hello()


def add(a, d):
    result = a + d
    return result


sum_result = add(3, 5)
print('sum_result', sum_result)
print('add(10, 5)', add(10, 5))

# 사각형 넓이


def calculate_area(width, height):
    # 문서화 문자열(Docsting)
    '''
    직사각형 넓이를 계산 합니다.
    Parameters:
        width (float) : 직사각형의 너비
        height: (float) : 직사각형의 높이
    Return:
        float : 직사각형의 넓이
    '''
    return width * height


print(calculate_area(10, 20))
# Docsting 확인
print(calculate_area.__doc__)  # 문서화 출력
help(calculate_area)  # 문서화 설명

# =================== 다시 읽어봐
# 매개변수와 인자
'''
매개변수(Parameter) : 함수 정의 시 사용하는 변수
인자(Argument) : 함수 호출시 전달하는 실제 값
'''


def multiply(x,  y):  # x, y는 매개변수
    return x * y


result = multiply(3, 5)  # 3, 5는 인자

# 위치 인자 (Positional Arguments)


def introduce(name, age, city):
    print(f'{name} {age} {city}')


# name = 김철수 , age = 25, city = 서울
introduce('김철수', 25, '서울')

# 키워드 인자 (Keyword Arguments)


def introduce(name, age, city):
    print(f'{name} {age} {city}')


# 순서와 상관없이 이름을 전달
introduce(city='서울', age=25, name='김철수')

# 위치 인자, 키워드 혼용
introduce('김철수', city='서울', age=25)
# 주의 : 위치 인자는 키워드 인자보다 앞에

# introduce(20, city = '부산', name = '이영희') 오류 발생

# 반환값(return)
# 단일 값 반환


def square(n):
    return n ** 2


result = square(5)
print(result)  # 25

# 여러 값 반환


def calculate_stats(numbers):
    total = sum(numbers)
    avg = total / len(numbers)
    maxnum = max(numbers)
    minnum = min(numbers)

    return total, avg, maxnum, minnum


numbers = [100, 140, 230, 200]
a, b, c, d = calculate_stats(numbers)  # 본래 이름 a => total ...

print('total:', a)
print('avg:', b)
print('maxnum:', c)
print('minnum:', d)
stats = calculate_stats(numbers)
print(stats)  # (a, b, c, d) 형태로 출력


# return의 특징
def check_positive(number):
    if number > 0:
        return "양수"
    elif number < 0:
        return '음수'
    else:
        return '0'
    print('실행 안됨')     # return이 함수를 종료시키기 때문

# 조기 반환(Early return)


def divide(a, b):
    # 예외 상황 먼저 처리
    if b == 0:
        return "0으로 나눌 수 없다."
    return a / b


print(divide(10, 2))
print(divide(10, 0))

# 기본값 매개변수


def greet(name, message='안녕하세요'):  # 기본값은 뒤에서부터 차례대로 기입
    print(f'{name} {message}')
    print(f'{message}, {name}님')


greet('sumin')  # 두 번째 예 : 안녕하세요, sumin님
greet('jungi', '안녕')  # 기본값 위치에 다른 값 입력 시 기본값 무시
# 안녕, jungi님
# 여러 기본값


def create_profile(name, age=25, city='서울', job='개발자'):
    return {
        'name': name,
        'age': age,
        'city': city,
        'job': job
    }


print(create_profile('박민수'))
print(create_profile('김철수', 30))
print(create_profile('이영희', job='모델'))


'''
def add (a, b, c = 13, d = 30):
    print('a', a)
    print('b', b)
    print('c', c)
    print('d', d)

add(1, 2, d = 15) --------------- c는 기본값을 쓰고 d는 다른 값을 입력하고 싶으 때
'''

# 위치 가변인자(*args)


def add_all(*new_tuple):  # * 중요 / tuple 형태 (본 이름 *args)
    return sum(new_tuple)


result = add_all(1, 2, 3, 4, 5)
print('result', result)


def sum_all(*numbers):
    total = 0
    for num in numbers:
        total += num
    return total


print(sum_all(1, 2, 3))
print(sum_all(1, 2, 3, 4, 5, 6, 7, 8))
print(sum_all())  # 0


# 키워드 가변인자(**kwargs)


def print_info(**new_dic):  # ** 중요 / dict 형태 (본 이름 **kwargs)
    for key, value in new_dic.items():
        print(f'{key} {value}')


print_info(name='홍길동', age=25, city='서울')

# 예제


def create_student(**info):
    # 학생 정보 생성
    student = {
        'name': info.get('name', '이름 없음'),
        # get('name', '이름 없음') => [name] 전환가능 / 아래 선언
        'age': info.get('age', 20),
        # get('age', 20) => [age]로 전환 시 에러 (아래에서 age 선언 X)
        'grade': info.get('grade', 1),
        'subjects': info.get('subjects', [])
    }
    return student


student1 = create_student(name='김철수')
student2 = create_student(name='이영희', subjects=['python'])
