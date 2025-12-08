
# 데이터 정제
# 현실의 더러운 데이터 예신
'''
문제점
 중독 데이터
 결측값 : None Nan
 형식 불일치: age, join_data
 이상 값: age 250
'''

'''
결측값(Missing Value)
 비어있는, 알 수 없는, 기록되지 않는 데이터

종류
 None: Python의 빈 객체
 np.nan: Numpy의 Not a Number
 pd.NA: Pandas의 결측값(최신)
 빈 문자열: ''또는 ""(공백)
 특수 값: -99999, 00000 
 '''
import pandas as pd
import numpy as np

missing_types = pd.DataFrame({
    'none_type': [1, 2, None, 4],           # Python None
    'nan_type': [1, 2, np.nan, 4],         # NumPy NaN
    'empty_string': ['A', 'B', '', 'D'],   # 빈 문자열
    'whitespace': ['A', 'B', ' ', 'D'],    # 공백
    'special_value': [1, 2, -999, 4]       # -999를 결측값으로 사용하는 경우
})
print(missing_types)

# 결측값 탐지
# isnull() / isna()
# 결측값이면 True

# notna() / notnull)
# 값이 있으면 True

print('===isna()===')
print(missing_types.isna())
print(missing_types.isnull())
print()
print('===notna()===')
print(missing_types.notna())
print(missing_types.notnull())

# 결측값 통계 확인
print('===열 별 결측값 개수===')
print(missing_types.isna().sum())

# 전체 결측값 개수
print('결측값 전체 개수:', missing_types.isna().sum().sum())
print()

# 결측값 처리 전략
'''
1. 삭제 - 결측값이 있는 행/열 제거
2. 대체 - 다른 값으로 채우기
3. 예측 - 앞뒤 값이나 패턴으로 추정
'''
# 결측값이 있는 샘플 데이터
sales_data = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=7),
    'sales': [100, 120, np.nan, 150, np.nan, 180, 200],
    'customers': [20, 25, 22, np.nan, 30, 35, 40],
    'region': ['Seoul', 'Busan', np.nan, 'Daegu', 'Seoul', np.nan, 'Busan']
})

print('===원본===')
print(sales_data)

# 삭제
# 1-1 결측겂이 있는 행 전체 삭제
drop_rows = sales_data.dropna()
print('결측값이 있는 행 삭제:')
print(drop_rows)

# 1-2 결측값이 있는 열 전체 삭제
drop_cols = sales_data.dropna(axis=1)
print('결측값이 있는 열 삭제:')
print(drop_cols)

# 1-3 특정 열 기준 삭제
drop_sales = sales_data.dropna(subset=['sales'])
print('sales 열 기준으로만 삭제:')
print(drop_sales)
print()

# 대체
# 2-1 평균값으로 대체
fill_mean = sales_data.copy()
fill_mean['sales'] = fill_mean['sales'].fillna(fill_mean['sales'].mean())
print(fill_mean)
print()

# 2-2 중앙값으로 대체 (이상값이 있을 때 유용)
fill_median = sales_data.copy()
fill_median['sales'] = fill_median['sales'].fillna(
    fill_median['sales'].median())
print(fill_median)

# 시계열 대체
# 시간 순서가 있는 데이터에서 앞뒤 값으로 결측값을 채운다.
# 3-1 Forward Fill (앞의 값으로 채우기)

fill_forward = sales_data.copy()
fill_forward['sales'] = fill_forward['sales'].fillna(method='ffill')
print(fill_forward)

fill_forward['customers'] = fill_forward['customers'].fillna(method='ffill')
print(fill_forward)

# 3-2 Backward Fill (뒤의 값으로 채우기) -----------------뭐가 다른거야 확인좀
fill_forward['customers'] = fill_forward['customers'].fillna(method='bfill')
print(fill_forward)

# 그룹화의 집계
'''
GroupBy
그룹화는 데이터를 특정 기준에 따라 묶어서 분석하는 것

전체 평균만으로는 부족한 경우 많음
카테고리별, 기간별로 나누넝 보면 숨겨진 패턴 발견
세그먼트별 비교분석 가능
'''
employee_data = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Henry'],
    'department': ['Dev', 'Dev', 'Sales', 'Sales', 'Dev', 'HR', 'HR', 'Sales'],
    'years': [3, 5, 2, 7, 10, 4, 6, 3],
    'salary': [4500, 5500, 4000, 6500, 8000, 4800, 5800, 4200]
})

print('전체 직원 데이터')
print(employee_data)

# 전체 분석 vs 그룹별 분석
print('전체 분석')
overall_avg = employee_data['salary'].mean()
print(f'전체 푱균 연봉: {overall_avg}')
print()

print('그룹별 분석')
dept_avg = employee_data.groupby('depqrtment')['salary'].mean()
print(dept_avg)
print()

# GroupBy 핵심
'''
# Split - Apply - Combine
# 3단계 프로세스
 1. Split(분할) - 데이터를 그룹으로 나누기
 2. Apply(적용) - 각 그룹에 함수 적용
 3. Combine(결합) - 결과를 하나로 합치기
'''
simple_data = pd.DataFrame({
    'category': ['A', 'B', 'A', 'B', 'A', 'B'],
    'value': [10, 20, 15, 25, 12, 22]
})

print('원본 데이터')
print(simple_data)
print()

# 1단계 Split
for category, group in simple_data.groupby('category'):
    print(f'{category} 그룹')
    print(group)

# 2단계 Apply
for category, group in simple_data.groupby('category'):
    avg = group['value'].mean()
    print(f'{category} 그룹 평균 {avg}')
print()

# 3단계 Combine
result = simple_data.groupby('category')['value'].mean()
print(result)
print()

# groupby(by= None, axis=0, level=None, as_index=True, sort=True...)
'''
 by : 그룹화 기준 (컬럼명 / 컬럼명 리스트)
 as_index : 그룹 키를 인덱스로 사용여부 (기본 = True)
 sort : 그룹 키를 정렬할지 여부 (기본 = True)
'''
employee_data = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Henry'],
    'department': ['Dev', 'Dev', 'Sales', 'Sales', 'Dev', 'HR', 'HR', 'Sales'],
    'years': [3, 5, 2, 7, 10, 4, 6, 3],
    'salary': [4500, 5500, 4000, 6500, 8000, 4800, 5800, 4200]
})

# 방법 1 - 컬럼명 문자열
result1 = employee_data.groupby('department')['salary'].mean()

# 방법 2 - 컬럼 접근 후 집계 (차이 X)
result2 = employee_data.groupby('department').salary.mean()

# 여러 컬럼 선택
result3 = employee_data.groupby('department')[['salary', 'years']].mean()

# as_index 매개변수
result_indexed = employee_data.groupby(
    'department', as_index=True)['salary'].mean()
print('result_indexed')
print(f'타입 : {type(result_indexed)}')

result_indexed_n = employee_data.groupby(
    'department', as_index=False)['salary'].mean()
print('result_indexed_n')
print(f'타입 : {type(result_indexed_n)}')

# sort 매개변수 ---- 사용시 약간의 성능 저하.
result_sorted = employee_data.groupby(
    'department', sort=True)['salary'].mean()
print('result_sorted')

result_sorted_n = employee_data.groupby(
    'department', sort=False)['salary'].mean()
print('result_sorted_n')

'''
    count() - 개수
    var() - 분산
    std() - 표준편차
    ...
'''

# describe() 매서드
# 여러 통례를 한번에 계산하는 매서드
result = employee_data.groupby('department')['salary'].describe()
print(result)

employee_detail = pd.DataFrame({
    'department': ['Dev', 'Dev', 'Dev', 'Sales', 'Sales', 'Sales', 'HR', 'HR'],
    'position': ['Junior', 'Mid', 'Senior', 'Junior', 'Mid', 'Senior', 'Mid', 'Senior'],
    'gender': ['M', 'F', 'M', 'F', 'M', 'M', 'F', 'F'],
    'salary': [4000, 4500, 5500, 3800, 4300, 5200, 4500, 5300]
})

multy_group = employee_detail.groupby(
    ['department', 'position'])['salary'].mean()
print(multy_group)

# ===========================================
monthly_sales = pd.DataFrame({
    'month': [1, 1, 2, 2, 3, 3, 1, 2, 3],
    'store': ['A', 'B', 'A', 'B', 'A', 'B', 'C', 'C', 'C'],
    'sales': [100, 80, 120, 90, 150, 100, 110, 95, 130],
    'customers': [50, 40, 60, 45, 75, 50, 55, 48, 65]
})

# agg() 다양한 사용법
# 1. 함수 이름 리스트
result1 = monthly_sales.groupby('store')['sales'].agg(['mean', 'sum', 'std'])

# 2. 함수 객체 --- 경고 뜸
# result1 = employee_detail.groupby('department')['salary'].agg('np.mean', 'np.sum')

# 3. 딕셔너리로 컬럼별 다른 함수
result2 = monthly_sales.groupby('store').agg({
    'sales': ['mean', 'sum'],
    'customers': ['mean', 'max']
})

print(result2)
