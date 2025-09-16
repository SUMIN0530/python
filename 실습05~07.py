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

# ==========================================================================
# 실습 06.1 set 종합 연습
# 문제 1.중복 제거 및 개수 세기
submissions = ['kim', 'lee', 'kim', 'park', 'choi', 'lee', 'lee']
fn = set(submissions)
count = len(fn)          # 없이 print에 len 바로 적용 가능
print(f'제출한 학생 수: {count}\n제출지 명단: {fn}')  # f 사용 없이 , fn 가능

# 문제 2. 공통 관심사 찾기
user1 = {'SF', 'Action', 'Drama'}
user2 = {'Drama', 'Romance', 'Action'}
t = user1 & user2  # 교집합
e = user1 ^ user2  # 대칭 차집합
a = user1 | user2  # 합집합
print(f'공통 관심 장르: {t}\n서로 다른 장르: {e}\n전체 장르: {a}')

# 문제 3. 부분집합 관계 판단
my_certificates = {'SQL', 'Python', 'Linux'}
job_required = {'SQL', 'Python'}
print('지원 자격 충족 여부: ', job_required.issubset(my_certificates))  # my >= job

# ==========================================================================
# 실습 07.1 딕셔너리 종합 연습 문제
# # 문제 1. 딕셔너리 핵심 개념 통합 실습
user = {}                           # 1. user 빈 딕셔너리 생성

user['username'] = 'skywalker'      # 2. 사용자 기본 정보 추가 / updata 이용가능
user['email'] = 'sky@example.com'
user['level'] = 5
print(user)

email_value = user['email']         # 3. email 값을 변수 email_value에 저장
print('email_value:', email_value)

user['level'] = 6                   # 4. level 값 변경
print(user)

print(user.get('phone', '미입력'))  # phone 값이 없다면 '미입력' 출력

user.update({'nickname': 'sky'})    # 항목 추가 및 삭제
del user['email']
user.setdefault('singup_date', "2025-07-10")
print(user)

# 문제 2. 학생 점수 관리
students = {}               # 빈 딕셔너리 생성
students.update({           # 세 학생의 점수 추가
    'Alice': 85,
    'Bob': 90,
    'Charlie': 95
})
print(students)

students['David'] = 80      # 데이비드 점수 추가
print(students)

students['Alice'] = 88      # 앨리스 점수 수정
print(students)

students.pop('Bob')         # Bob 삭제
print(students)
