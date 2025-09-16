'''
Key - Value 쌍으로 데이터를 저장하는 자료구조
해시 테이블 기반으로 배우 빠른 검색 속도
Python의 가장 강력하고 유용한 자료 구조 중 하나
'''
# 딕셔너리 사용 X
import copy
student_ids = ["20230001", "20230002", "20230003"]
student_names = ["김철수", "이영희", "박민수"]

target_id = "20230002"
if target_id in student_ids:
    index = student_ids.index(target_id)  # O(n)
    name = student_names[index]
    print(name)

# 딕셔너리 사용
students = {
    "20230001": "김철수",
    "20230002": "이영희",
    "20230003": "박민수"
}
print(students["20230002"])  # O(1) - 즉시 접근!

'''
특징
 - Key - Value 쌍: 각 값에 고유한 키로 접근
 - 빠른 검색: 0(1) 시간 복잡도
 - 변경 가능 (Mutable): 요소 추가, 수정, 삭제 가능
 - Key는 유일: 중복 키 불가 / 중복 값 가능
 - 순서 보장: Python3.7+ 부터 삽입 순서 유지
'''

# 빈 Dictionary 생성
empty_dict = {}
print(type(empty_dict))

# 값이 있는 Dictionary 생성
person = {
    'name': "김철수",
    'age': 25,
    'city': 'seoul'
}
print(person)

# dict() 생성자 사용
person2 = dict(name='이영희', age=25, city='부산')
print('person2', person2)

# 리스트/튜플로부터 생성
# [[]], (()), [()], ([]) 내부 요소가 리스트 일 때, 외부가 튜블일 때 모두 가능
pairs = [('a', 1), ('b', 2)]
dict_from_pairs = dict(pairs)
print('dict_from_pairs', dict_from_pairs)

# zip()를 이용한 생성
keys = ['name', 'age', 'city']
values = ['박민수', 21, '대전']

person3 = dict(zip(keys, values))
print(person3)

# dictionart comprehension
squares = {x: x**2 for x in range(1, 6)}
print('squares', squares)

# fromkeys() 메서드
keys = ['a', 'b', 'c']
default_dict = dict.fromkeys(keys, 0)  # 모든 값이 0이 된다.
print('default_dict', default_dict)

# key로 사용 가능한 타입
# Hashable 타입만 key로 사용 가능
valid_dict = {
    1: '정수',
    3.14: '실수',
    '문자열': 'string',
    (1, 2): '튜플',
    True: '불리언',
    frozenset([1, 2]): 'frozenset'
}

# UnHashable 타입은 key로 사용 불가 ---- 오류
'''
invalid_dict = {
    [1, 2]: '리스트',
    {1, 2}: 'set',
    {'a': 1}: '딕셔너리'
}
'''

# 접근과 수정
student = {
    'name': '김철수',
    'age': 20,
    'major': '컴퓨터공학',
    'gpa': 3.7
}

# 대괄호 표기법(KeyError 위험)
name = student['name']
print('name', name)
# grade = student['grade']
# print('grade', grade)

# get() 메서드(안전한 접근)
major = student.get('major')
print('major', major)

grade1 = student.get('grade')
print('grade1', grade1)

# 기본값 지정
grade2 = student.get('grade', 'N/A')
print('grade2', grade2)

# 뭐가 있음


student = {
    'name': '김철수',
    'age': 20,
    'major': '컴퓨터공학',
    'gpa': 3.7
}

# 값 수정, 추가
student['age'] = 23
print('student', student)

student['grade'] = 3
print('student', student)

# update() 메서드로 여러 값 한번에 수정
student.update({
    'gpa': 4.0,
    'city': 'seoul',
    'email': 'sumin@gmail.com'
})
print('student', student)

info_dict = {
    'age': 22,
    'grade': 1,
    'phone': '010-1234-1234'
}
student.update(info_dict)
print('student', student)

# 값 삭제
del student['grade']  # del

popped = student.pop('phone')  # pop : 값을 반환하면서 삭제 ()빈 괄호는 에러

last_item = student.popitem()  # popitem : 마지막 항목 제거

student.clear()  # clear : 안에 있는 모든 요소 삭제

# 중요 메서드
scores = {
    '김철수': 85,
    '이영희': 92,
    '박민수': 78
}

# setdefault() - 키가 없으면 추가, 있으면 기존 값 반환
scores.setdefault('정수진', 88)  # 새로 추가
scores.setdefault('김철수', 100)  # 기존 값 유지
print(scores)  # -------134에 에러

# import : 라이브러리를 쓰겠다. 저장하면 제일 위로 올라감

# copy() --- 얕은 복사
scores_copy = scores.copy()
scores_copy['최동훈'] = 95
print('scores', scores)
print('scores_copy', scores_copy)
print()


# deepcopy() --- 깊은 복사
nested_dict = {
    "team1": {'leader': '김철수', 'members': ['이영희', '박민수']},
    "team2": {'leader': '정수진', 'members': ['최동훈', '강미나']}
}
shallow = nested_dict.copy()  # 얕은 복사
deep = copy.deepcopy(nested_dict)  # 깊은 복사

# 원본 값에 추가
nested_dict['team1']['members'].append('신입')
print('shallow: ', shallow['team1']['members'])  # 얕은 복사에 영향 있음
print('deep: ', deep['team1']['members'])       # 깊은 복사에 영향 없음
print()

# 순회
'''
scores = {
    '김철수': 85,
    '이영희': 92,
    '박민수': 78
}
'''
# 키만 순회(기본)
for key in scores:
    print(f'{key}')
print()
for key in scores.keys():            # 명시적
    print(f'{key}: {scores[key]}')
print()
for value in scores.values():
    print(f'{value}')

# 평균값 계산
average = sum(scores.values()) / len(scores)
print('average', average)

# 키 - 값 쌍 순회
for key, value in scores.items():
    print(f'{key}: {value}')

for idx, (key, value) in enumerate(scores.items(), 1):  # 1 : 처음을 1로 표기  e어쩌고가 뭔데
    print(f'{idx}번째 {key} : {value}점')
