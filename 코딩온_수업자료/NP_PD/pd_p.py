# 실습 01. Series 연습
# 문제 1. 파이썬 리스트 [5, 10, 15, 20]을 이용해 Series를 생성하세요.
import numpy as np
import pandas as pd
num_list = [5, 10, 15, 20]
num = pd.Series(num_list)
print(num)

# 문제 2. 값[90, 80, 85, 70]에 대해 인덱스를 각각 '국어', '영어', '수학', '과학'으로 지정한 Series를 만드세요.
sub_list = [90, 80, 85, 70]
product_series = pd.Series(sub_list, index=['국어', '영어', '수학', '과학'])
print(product_series)

# 문제 3. {'서울': 950, '부산': 340, '인천': 520} 딕셔너리를 이용해 Series를 만들고, 인천의 값을 출력하세요.
d = {
    '서울': 950,
    '부산': 340,
    '인천': 520
}
sd = pd.Series(d)
print('인천 값: ', sd['인천'])

# 문제 4. Series [1, 2, 3, 4]를 만들고, 데이터 타입(dtype)을 출력하세요.
dt = pd.Series([1, 2, 3, 4])
print('데이터 타입: ', dt.dtype)

# 문제 5. 아래 두 Series의 합을 구하세요.
s1 = pd.Series([3, 5, 7], index=['a', 'b', 'c'])
s2 = pd.Series([10, 20, 30], index=['b', 'c', 'd'])
result = s1 + s2  # 두 값의 차로 상품 가격 차이 유추 가능
print(result)

# 문제 6. Series[1, 2, 3, 4, 5]의 각 값에 10을 더한 Series를 만드세요.
s = [1, 2, 3, 4, 5]
s1 = pd.Series(s) + 10
print(s1)

# 실습 02. DataFrame 연습
# 문제 1. 다음 데이터로 DataFrame을 생성하고, 컬럼명을 '이름', '나이', '도시'로 지정하세요.
data = pd.DataFrame(
    [['홍길동', 28, '서울'],
     ['김철수', 33, '부산'],
     ['이영희', 25, '대구']],
    columns=['이름', '나이', '도시']
)
print(data)

# 문제 2. 아래와 같은 딕셔너리로 DataFrame을 생성하세요.
data = {
    'A': [1, 2, 3],
    'B': [4, 5, 6]
}
dt = pd.DataFrame(data)
print(dt)

# 문제 3.  아래 데이터를 사용해 DataFrame을 만드세요.
data = [
    {'과목': '수학', '점수': 90},
    {'과목': '영어', '점수': 85},
    {'과목': '과학', '점수': 95}
]
dt = pd.DataFrame(data)
print(dt)

# 문제 4. 아래 데이터를 사용해 DataFrame을 생성하되, 인덱스를 ['학생1', '학생2', '학생3']으로 지정하세요.
data = {
    '이름': ['민수', '영희', '철수'],
    '점수': [80, 92, 77]
}
dt = pd.DataFrame(data, index=['학생1', '학생2', '학생3'])
print(dt)

# 문제 5. 아래 Series 객체 2개를 이용해 DataFrame을 만드세요.
kor = pd.Series([90, 85, 80], index=['a', 'b', 'c'])
eng = pd.Series([95, 88, 82], index=['a', 'b', 'c'])
data = pd.DataFrame({'kor': kor, 'eng': eng})
print(data)

# 문제 6. 아래 딕셔너리로 DataFrame을 만들고, 컬럼 순서를 ['B', 'A']로 지정해 출력하세요.
data = {
    'A': [1, 2],
    'B': [3, 4]
}
dt = pd.DataFrame(data)
print(dt[['B', 'A']])

# 문제 7. 데이터를 DataFrame으로 만들고, 컬럼명을 ['product', 'price', 'stock']으로 변경하세요.
data = pd.DataFrame(
    [['펜', 1000, 50],
     ['노트', 2000, 30]],
    columns=['product', 'price', 'stock']
)
print(data)

# 문제 8. 아래 DataFrame을 생성한 뒤, '국가' 컬럼만 추출하세요.
data = {
    '국가': ['한국', '일본', '미국'],
    '수도': ['서울', '도쿄', '워싱턴']
}
dt = pd.DataFrame(data)
print(dt[['국가']])

# 실습 CSV
# 문제 1. CSV파일 읽고 저장하기
practice_data = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['Seoul', 'Busan', 'Daegu']
})

practice_data.to_excel('practice_data.xlsx', index=False)
print('엑셀 파일 생성')
prac_excel = pd.read_excel('practice_data.xlsx')
print('===Excel 파일 읽기===')
print(prac_excel)


# 문제 2. 한글 데이터를 UTF-8로 저장하고 읽기
korean_data = pd.DataFrame({
    '이름': ['김철수', '이영희', '박민수'],
    '직급': ['사원', '대리', '과장']
})

korean_data.to_csv('korean_data.csv', index=False, encoding='UTF-8')
print('csv 파일 생성')
k_csv = pd.read_csv('korean_data.csv', encoding='UTF-8')
print('===csv 파일 읽기===')
print(k_csv)

# 실습 4. 통계함수 결측값 처리 연습
data = {
    "도시": ["서울", "부산", "광주", "대구", np.nan, "춘천"],
    "미세먼지": [45, 51, np.nan, 38, 49, np.nan],
    "초미세먼지": [20, np.nan, 17, 18, 22, 19],
    "강수량": [0.0, 2.5, 1.0, np.nan, 3.1, 0.0]
}

df = pd.DataFrame(data)
# 미세먼지 평균값
fill_mean = df.copy()
print('미세먼지 평균값:', fill_mean['미세먼지'].mean())

# 미세먼지 중앙값
fill_median = df.copy()
print('미세먼지 중앙값:', fill_median['미세먼지'].median())
print()

# 초미세먼지 최댓값
fill_max = df.copy()
print('초미세먼지 최댓값:', fill_max['초미세먼지'].max())

# 초미세먼지 최솟값
fill_min = df.copy()
print('초미세먼지 최솟값:', fill_min['초미세먼지'].min())
print()

# 전체 결측값 개수
print('결측값 전체 개수:', df.isna().sum().sum())
print()

# 결측값이 있는 행 삭제 후 초미세먼지 평균 출력
drop_rows = df.dropna()  # 결측값이 있는 행 삭제
print(drop_rows)
print('초미세먼지 평균:', drop_rows['초미세먼지'].mean())
print()

# 결측값을 모두 0으로 채운 뒤 미세먼지와 초미세먼지 각각의 합
fill_zero = df.copy()
fill_zero['미세먼지'] = fill_zero['미세먼지'].fillna(0)
fill_zero['초미세먼지'] = fill_zero['초미세먼지'].fillna(0)
print(fill_zero)
print('미세먼지 합계:', fill_zero['미세먼지'].sum())
print('초미세먼지 합계:', fill_zero['초미세먼지'].sum())

# 미세먼지 컬럼 결측값을 평균값으로 채운 뒤 표준편차 도출
fill_mean['미세먼지'] = fill_mean['미세먼지'].fillna(fill_mean['미세먼지'].mean())
print(fill_mean)
print('미세먼지 표준편차:', fill_mean['미세먼지'].std())
