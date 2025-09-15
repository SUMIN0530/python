"""
# 실습 05.1 튜플 종합 연습
# Step 1. 손상된 고객 정보 복원하기
user = ("minji", 25, "Seoul")
restored_user = ('eunji',) + user[1:]  # , 필수
print('restored_user: ', restored_user)

# Step 2. 고객 정보 언패킹하여 변수에 저장
name, age, city = restored_user
print(f'name: {name}, age: {age}, city: {city}')

# Step 3. 지역별 보안 정책 분기 처리
if city == 'Seoul':
    print('서울 지역 보안 정책 적용 대상입니다.')
else:
    print('일반 지역 보안 정책 적용 대상입니다.')

# Step 4. 고객 데이터 통계 분석
users = ("minji", "eunji", "soojin", "minji", "minji")
count = users.count("minji")
print(f'minji라는 이름이 {count}번 등장한다.')
index = users.index('soojin')
print(f'soojin이라는 이름은 {index}번째 위치에 등장한다.')

# Step 5. 고객 이름 정렬
sorted_users = list(users)
sorted_users.sort()
print(users)
print(sorted_users)
"""
# ==========================================================================
# 실습 06.1 set 종합 연습
# 문제 1.중복 제거 및 개수 세기
submissions = ['kim', 'lee', 'kim', 'park', 'choi', 'lee', 'lee']
fn = set(submissions)
count = len(fn)
print(f'제출한 학생 수: {count}\n제출지 명단: {fn}')

# 문제 2. 공통 관심사 찾기
user1 = {'SF', 'Action', 'Drama'}
user2 = {'Drama', 'Romance', 'Action'}
t = user1 & user2
e = user1 ^ user2
a = user1 | user2
print(f'공통 관심 장르: {t}\n서로 다른 장르: {e}\n전체 장르: {a}')

# 문제 3. 부분집합 관계 판단
my_certificates = {'SQL', 'Python', 'Linux'}
job_required = {'SQL', 'Python'}
print('지원 자격 충족 여부: ', job_required.issubset(my_certificates))
