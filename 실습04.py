# 04.1.1 첫번째 요소와 마지막 요소 출력
nums = [10, 20, 30, 40, 50]
print('f:', nums[:1], 'l:', nums[4:])
# 리더님 : print(nums[0]) / print(nums[-1])

# 04.1.2 가운데 세 개의 요소 추출하기
nums = [100, 200, 300, 400, 500, 600, 700, 800, 900]  # 왜 전 리스트에 영향을 안주지?
print('가운데 세 수:', nums[3:6])
# 리더님 : mid = 7//2  /  print(nums[mid-1: mid-2])  (100~700기준)


# 04.1.3 리스트 원소 두 배하기
nums = [1, 2, 3, 4, 5]
nums = [num*2 for num in nums]  # 리스트여서 함축이 가능한건가??
print('원소 두 배:', nums)

'''
리더님
nums = [1, 2, 3, 4, 5]
for i in range(5):    -------- for i in range(len(nums))
    nums[i] *= 2

print('nums', nums)
'''
# 04.1.4 리스트 뒤집어 출력하기
items = ["a", "b", "c", "d", "e"]
print('리스트 뒤집기:', items[::-1])

# 04.1.5 짝수 인덱스 요소만 출력
data = ["zero", "one", "two", "three", "four", "five"]
print('짝수만 출력:', data[::2])

# 04.1.6 슬라이싱으로 리스트 수정
movies = ["인셉션", "인터스텔라", "어벤져스", "라라랜드", "기생충"]
# 어벤져스, 라라랜드 -> 매트릭스, 타이타닉
movies[2:4] = ["매트릭스", "타이타닉"]
print('수정된 리스트:', movies)

# 04.1.7 특정 규칙에 따라 요소 추출
subjects = ["국어", "수학", "영어", "물리", "화학", "생물", "역사", "지구과학", "윤리"]
# 물리, 생물, 지구과학
subs = subjects[3::2]
print('새 리스트:', subs)
# 리더님 : result = [subjects[3], subjects[5], subjects[7]] -> print(result)

# 04.1.8 리스트를 3 구간으로 나누어 역순 병합
data = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
# 리더님 : data1 = data[:3][::-1] -> print(data1, data2, data3, sep=" ")
data1 = data[:3]
data2 = data[3:6]
data3 = data[6:]
print('등분 후 역순 병렬:', data1[::-1], data2[::-1], data3[::-1])

# 04.2.1 부분 삭제 후 연결
fruits = ["apple", "banana", "cherry", "grape", "watermelon", "strawberry"]
del fruits[1:4]
print(fruits)

# 04.2.2 반복 리스트 내부 요소 삭제
letters = ["A", "B"]
result = letters*3
del result[2]
print(result)

# 04.3.1 기차 탑승 시뮬레이션
train = ['철수', '영희']            # 1. 철수와 영희 탑승
train.extend(['민수', '지훈'])      # 2. 다음 역에서 민수와 지훈이 탑승
del train[1]                        # 3. 다음 역에서 영희 하차
train.insert(1, '수진')             # 4. 수진이 1번 자리에 탑승
train.remove('민수')                # 5. 마지막 역에서 민수 하차
train.reverse()                     # 기차 안의 순서를 뒤집었다
print(train)

# 04.3.2 숫자 처리 게임
list1 = [5, 3, 7]                # 다음과 같은 카드 리스트
list2 = [4, 9]                   # 카드 두 장 추가
list = list1+list2               # 리스트 모음
max_mun = max(list)
min_mun = min(list)
sum = sum(list)
print('max: ', max_mun)
print('min: ', min_mun)          # 최대, 최소값 구하기
print('sum: ', sum)              # 총 합 구하기
print()
list.sort()                      # 리스트 정렬
del list[4]
print(list)
