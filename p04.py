'''
컬렉션 자료형
list : 순서 O, 중복 허용, 가변 / []
tuple : 순서 O, 중복 허용, 불변 / ()
set : 순서 X, 중복 불허, 가변 / {}
dict : 키 - 값 쌍 저장, 중복 키 불허, 순서 O (3.7+) / {'a':1, 'b':2}

in / not in : 포함 여부 확인
for문 순회
len() : 길이 확인

시퀀스 (컬렉션 하위개념): list, tuple
시퀀스 자료형
str / list / tuple / range (list만 가변)
'''

# 리스트 변형 (형 변환 같은건가) --- 이터러블을 리스트로 변환  리스트는 []사용
list1 = list()
list2 = list('Hello')

print(list1)   # []
print(list2)   # ['H', 'e', 'l', 'l', 'o']

# 인덱싱 --- 시퀀스 자료형에서 특정 위치값 도출
#  H  e  l  l  o
#  0  1  2  3  4
# -5 -4 -3 -2 -1  / -5 = 0

print('인덱스 0: ', list2[0])  # 0번째 항복 도출
print('인덱스 3: ', list2[3])
print('인덱스 -3: ', list2[-3])

# 리스트 항목 변경(인덱싱) --- 문자열 불변
list2[4] = 'a'
print('list2: ', list2)  # 리스트 타입이라 변경가능

text = 'python'
text[1] = 'a'   # 에러 발생
print('text', text)  # 타입이 문자열이기 때문에 변경 불가!!!

# 슬라이싱 문자열
list3 = list('python')
print('list3[:]', list3[:])
text3 = 'Hello'
print('text3[:]', text3[:])
print('text3[:3]', text3[:3])  # Hel
print('text3[2:4]', text3[2:4])  # ll
print('text3[-3:-1]', text3[-3:-1])  # ll

print('text3[::-1]', text3[::-1])  # olleH / 문자열 뒤집기
print('text3[::-2]', text3[::-2])  # olH / 마자막 칸 = 간격
print('text3[:-4:-2]', text3[:-4:-2])  # ol / 시작점 o

# 슬라이싱 리스트
numbers = [10, 20, 30, 40]
# 10, 20, 30만 뽑을 때
print('numbers[1:3]', numbers[1:3])  # 10, 20, 30
print('numbers[:3:2]', numbers[:3:2])  # 10, 30
print('numbers 뒤집기', numbers[::-1])

# [10, 40, 20, 40]으로 변경
numbers = [10, 20, 30, 40]
print('1. numbers', numbers)
numbers[1:3] = [40, 20]
print('2. numbers', numbers)

# 인덱스 요소 삭제
list1 = [10, 20, 30, 40, 50]
del list1[3]
print(list1)  # [10, 20, 30, 50]

del list[1:3]
print(list1)  # [10, 50]

del list1
print(list1)  # 없는 파일 => 에러 발생

fruits = ['사과', '바나나', '오렌지', '바나나', '포도']
fruits.remove('바나나')  # 처음 만나는 값 삭제
print(fruits)  # ['사과', '오렌지', '바나나', '포도']

removed = fruits.pop()  # 마지막 항 삭제
print(removed)
print(fruits)

removed = fruits.pop(1)  # 오렌지 삭제
print(removed)
print(fruits)

fruits.clear()  # 모든 요소 삭제 (리스트 유지) != del
print(fruits)

# 리스트 연결 (+)
list2 = [1, 2, 3, 4, 5]
list3 = [2, 3, 4, 5]

result = list2 + list3
print(result)

# 리스트 반복 (*)
result = list2 * 3
print(result)

# 리스트 포함 여부 (in / not in)
print('1' in list2)  # False : 해당 1은 문자형
print(1 in list2)  # True : 해당 1은 숫자형

# 요소 추가 메서드
numbers = [10, 21, 15, 22, 54]
numbers.append(20)  # 요소 하나만 추가
print(numbers)

numbers.append([1, 5, 7])  # 리스트 자체가 요소로 추가
print(numbers)

numbers.extend([19, 27])  # 요소 여러게 추가(하나도 가능) + 리스트 형식[]!!
print(numbers)

numbers.insert(2, 30)  # .insert(n, m) n번째 항에 m삽임
print(numbers)

list2 = [6, 7, 8]
numbers.extend(list2)  # 리스트 괄호가 사라지고 요소들이 추가
print(numbers)

# 요소 검색, 정렬 메서드
numbers = [1, 2, 6, 9, 5, 3, 2, 4, 7]

idx = numbers.index(6)  # 6 찾기
print('idx: ', idx)

idx = numbers.index(8)  # 8 없음 => 에러 발생
print('idx: ', idx)

count = numbers.count(2)  # 2가 몇 개 있냐
print('count: ', count)

numbers.sort()  # 오름차순 (한글 영어는 자음, 알파벳 순)
print('numbers: ', numbers)

numbers.sort(reverse=True)  # 내림차순
print('numbers: ', numbers)

# sorted
original = [3, 2, 5, 7, 1]
sorted_list = sorted(original)
sorted_list_r = sorted(original, reverse=True)

print('original: ', original)
print('sorted_list: ', sorted_list)
print('sorted_list_r: ', sorted_list_r)

# 연산 ? 메서드
numbers = [5, 2, 6, 4, 12, 26, 4, 11]
max_mun = max(numbers)
min_mun = min(numbers)

print('max_mun: ', max_mun)
print('min_mun: ', min_mun)

sum_mun = sum(numbers)
print('sum_mun: ', sum_mun)
