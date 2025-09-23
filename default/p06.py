'''
Set : 순서 X, 중복 X, 가변
수학의 집합 개념을 구현
해시 테이블 기반으로 빠른 멤버십 테스트 가능

필요 상황
- 중복 제거 (방문자 수 파악)  /
visitor = ['철수', '영희', '철수', '민수', '영희', '철수']
# 리스트로 제거 시 비효율적 O(n)
nuique_visitors_list = []:
for visitor in visitor:
    if visitor not in nuique_visitors_list:
        nuique_visitors_list.append(visitor)
print(nuique_visitors_list)
# Set으로 중복 제거 효율적 O(1) 검색
nuique_visitors_set = set(visitors)
print(nuique_visitors_set)

특징
 - 순서 없음 : 순서 보장 X
 - 중복 불가 : 같은 값 하나만 저장
 - 변경 가능 
 - 인덱싱 불가 : 순서가 없으므로 접근 불가
 - 빠른 검색 O(1) 시간 복잡도로 요소 확인 (스터디로 알아내기)
'''

# 빈 set 생성 방법
# empty_set = {} ------ 딕셔너리!
empty_set = set()

# 값이 있는 set 생성
numbers = {1, 2, 3, 5, 4, 3, 2, 4}
fruits = {'사과', '바나나', '오렌지'}

# 리스트 / 튜블에서 set 생성
list_numbers = [11, 2, 13, 5, 4, 3, 2, 4]
set_numbers = set(list_numbers)
print(set_numbers)

# 문자열에서 set 생성
chars = set('hello')
print(chars)

# Comprehension
for i in range(10):
    print(i)

com_set = {i for i in range(10)}  # 위 아래 동일 / {1, 2, 3, 4, 5, 6, 7, 8, 9}
com_set1 = {i * 2 for i in range(10)}  # {0, 2, 4, 6, 8, 10, ...}
com_set2 = {(i ** 2 + 1) for i in range(10)}  # {1, 5, 10, 17, ...}
com_set3 = {(i * 3 + 2 - 1) for i in range(10)}  # {1, 4, 7, ...}
com_set4 = set()
for i in range(10):
    com_set4.add((i * 3 + 2 - 1))

com_list = [i for i in range(2, 10, 2)]  # [2, 4, 6, 8]

new_list = [1, 2, 5, 1, 5]
com_set5 = {i for i in new_list}
print(com_set5)

# set에 저장 가능한 데이터 타입 (스터디)
# Hashable 타입만 가능 (불변 타입)
valid_set = {1, '문자열', (1, 2), 3.14, True, }

# Unhashable 타입 불가능 (가변 타입)
invalid_set = {[1, 2], {'key': "value"}, {1, 2}}  # 딕셔너리 형태??

# 중첩 set을 만들려면 frozenset() 사용
nested_set = {frozenset([1, 2]), frozenset([3, 4])}
print()

# set 메서드
colors = {'빨강', '노랑', '파랑'}

colors.add('초록')
print(colors)

colors.update(['보라', '주황'])
print(colors)

colors.update(['검정'], {'하양', '회색'})
print(colors)

colors.remove('검정')
print(colors)

# colors.remove('검정') --- 에러
# print(colors)

colors.discard('검정')
print(colors)

colors.discard('주황')
print(colors)

popped = colors.pop()  # -------copy
print(colors)
print(popped)

colors.clear()
print(colors)

# 집합 연산
A = {1, 2, 3, 4, 5}
B = {1, 2, 6, 7, 8}

intersection1 = A & B                  # 교집합
intersection2 = A.intersection(B)

union1 = A | B                         # 합집합
union2 = A.union(B)

difference1 = A - B                    # 차집합
difference2 = A.difference(B)

sym_difference1 = A ^ B                # 대칭 차집합
sym_difference2 = A.symmetric_difference(B)

A = {1, 2, 3}
B = {3, 4, 5}

A.intersection_update(B)  # 교집합으로 업데이트 하겠다
A &= B
print("A", A)  # A 3

A = {1, 2, 3}
A.difference_updata(B)  # 차집합으로 업데이트 하겠다
A -= B
print("A", A)  # A {1, 2}

A = {1, 2, 3}
A.symmetric_difference_update(B)  # 대칭 차집합으로 업데이트 하겠다
A ^= B
print("A", A)

A = {1, 2, 3}
A.update(B)  # 교집합으로 업데이트 하겠다
A |= B
print("A", A)

# 집합 관계 확인
A = {1, 2, 3}
B = {1, 2, 3, 4, 5}
C = {6, 7, 8}

# 부분집합인지 확인
print(A.issubset(B))  # True
print(A <= B)  # True

print(B.issubset(A))  # False

# 상위집합인지 확인
print(A.issuperset(B))  # False

print(B.issuperset(A))  # True
print(B >= A)  # True

# 진부분집합 확인
print(A < B)  # A == B 일때 False
print(A > B)

# 서로수집합
# 교집합이 없는지 확인
print(A.isdisjoint(C))

# 불변집합
fs1 = frozenset([1, 2, 3, 3, 4])
# fs1.add(5) --- 에러 발생 / 불변
# fs1.remove()
# fs1.discard()
