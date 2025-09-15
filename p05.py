"""
tuple : 순서 O, 중복 허용, 불변 (시퀀스 자료 구조)
한 번 생성 시 수정 불가 / 리스트와 구조 비슷
여러 개의 값을 하나의 단위로 묶을 때 사용

tuple의 필요성
 - 정보 보호 ex) 특정 좌표 / 변경 시도시 TypeError!!!

특징
 - 해시기능 : 딕셔너리 키로 사용가능
 - 메모리 효율적 : 리스트보다 적은 메모리 사용
"""
# 소괄호 사용
empty_tuple = ()
numbers = (1, 2, 3, 4, 5)
mixed = (1, "hello", 3.14, True)
print('mixed: ', mixed)

# 소괄호 없이 사용
numbers2 = 1, 2, 3, 4, 5
print('type(numbers2): ', type(numbers2))

# tuple() 생성자 사용
from_list = tuple([1, 2, 3, 4])
print('type(from_list): ', type(from_list))

form_str = tuple("hello")
print('type(form_str): ', type(form_str))

# 단일 요소 튜블(, 필수!!)
single = (10,)
print('type(single): ', type(single))

not_tuple = (10)
print('type(not_tuple): ', type(not_tuple))

# range로 튜플 생성
range_tuple = tuple(range(1, 10))
print('type(range_tuple): ', type(range_tuple))

# tuple 접근과 슬라이싱
fruits = ('사과', '바나나', '수박', '오렌지', '포도')
print(fruits[1])  # 바나나
print(fruits[-1])  # 포도

print(fruits[1:3])  # (바나나, 수박)
print(fruits[:2])  # (사과, 바나나)
print(fruits[::-1])  # (포도, 오렌지, 수박, 바나나, 사과)

# 슬라이싱으로 새 튜블 생성
first_two = fruits[0:2]  # 사과, 바나나, 수박
last_two = fruits[-2:]    # 오렌지, 포도

# 처음 두 개, 마지막 두 개
combined = first_two + last_two
print('combined: ', combined)

# 불변성 확인
numbers = (1, 2, 3, 4, 5, 6)
# 수정 시도 - 모두 에러 발생
# numbers[0] = 10 # TypeError
# numbers.append(6)
# del numbers[1]

# 새로운 튜플 생성 가능
new_numbers = numbers + (1, 2)
tuple_with_list = ([1, 2], [3, 4])
tuple_with_list[0].append(3)  # 리스트의 수정이므로 가능 => ([1, 2, 3], [3, 4])
tuple_with_list[0] = [2, 3]  # 튜플 요소인 리스트를 수정했으므로 에러

# 언패킹(Unpacking)
coordinates = (3, 5)
x, y = coordinates
print(f'x : {x}, y : {y}')

# 직접 언패킹
x, y = (10, 20)
print(f'x : {x}, y : {y}')
x = 20  # 변경
print(f'x : {x}, y : {y}')  # 변경 가능 / x는 tuple이 아닌 정수

x, y = (10, 20, 30)
print(f'x : {x}, y : {y}')  # 에러 발생 : 할당할 변수가 모자람

numbers = (1, 2, 3, 4, 5)
first, middle, last = numbers
print(first)
print(middle)
print(last)  # 위와 동일하게 에러 발생

numbers = (1, 2, 3, 4, 5, 6, 7, 8, 9)
first, *middle, last = numbers  # 가능하게 함.
print('first: ', first)  # 1
print('middle: ', middle)  # 1 ~ 8
print('last: ', last)  # 9 / 주석처리 하더라도 기존 값 잘 나옴 (확인 필요)

# 빈 리스트 생성
first, *rest = (1,)
print('first:', first)  # 1
print('rest:', rest)  # 2

# tuple 메서드
numbers = (1, 1, 3, 3, 2, 2, 5, 4, 3)

# count() - 특정 값의 개수 ----- 리스트와 동일
count_2 = numbers.count(2)
print('count_2', count_2)

# index() - 특정 값의 인덱스 --- 리스트와 동일
# 없는 값 검색 시 에러 발생
index_4 = numbers.index(4)

# 안전한 검색
value = 10
if value in numbers:
    print(f'{value}의 인덱스: {numbers.index(value)}')
else:
    print(f'{value}는 tuple에 없습니다.')

# 연산
tuple1 = (1, 2, 3)
tuple2 = (4, 5)

print(tuple1 + tuple2)
print(tuple2 * 3)

# 비교 연산 (사전식 비교) -- 앞이서부터 비교
tuple1 = (1, 3, 3)
tuple2 = (1, 2, 4)
print(tuple1 < tuple2)
print(tuple1=tuple2)

# 길이, 최대, 최소, 합
numbers = (1, 2, 3, 4)
print(len(numbers))
print(max(numbers))
print(min(numbers))
print(sum(numbers))

# del tuple ------ 튜플 전체 삭제
