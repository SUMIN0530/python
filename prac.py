# 실습 02.2 자기소개 하기

name = '이수민'
age = 23
MBTI = 'ESFJ'
"""
print('안녕하세요.', f'제 이름은 {name}이고,', f'{age}살 입니다.',
      f'제 MBTI는 {MBTI}에요.', end='\n')   # end 사용시 줄 바꿈 X

# 여쭤보기
X, Y = 30, 'a'
print('X', X, 'Y', Y, end='\n')


X, Y = 30, 'a'
print('X', X, sep='/', end='\n')  # end='-' : - 로 줄 이어짐 / 근데 end 안되는거 같은디?
print('Y', Y)
"""
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
