'''
데이터 분석
과정
    데이터 수집 - 데이터 정제 - 데이터 탐색 - 데이터 분석 - 시각화 - 인사이트 도출

1. 데이터 수집
    분석할 자료를 모으는 단게
2. 데이터 정제
    분석 가능한 형태로 만드는 단계
3. 데이터 탐색
    데이터의 특성 파악 단계
4. 데이터 분석
    가설을 검증하고 패턴을 찾는 단계
5. 시각화
    결과를 이해하기 쉽게 표현하는 단계
6. 인사이트 도출
    분석 결과를 의사결정에 활용하는 단계

예시)
    편의점 사장님의 고민
    문제 : 어떤 제품을 더 많이 주문해야 할까??
    데이터 : 지난 3개월간 판매 기롤
    분석 : 요일별, 시간대별 판매 패턴 파악
    인사이트 : 금요일 저녁에  맥주가 가장 많이 팔린다
    행동 : 금요일 전에 맥주 재고 확충
'''
import pandas as pd
print('pandas 버젼:', pd.__version__)

# Excelm, pandas 비교
'''
 Excel로 100만개 데이터를 처리한다면?
 2019 버젼 1,000,000행까지만 제한
 파일만 열어도 5분이상 소요
 수식 계산 프로그램 멈춤
 반복 작업 매번 수동으로 진행 
'''

# Series
# Pandas의 가장 기본이 되는 1차원 데잍터 구조
'''
 1차원  배열 : 데이터가 일렬로 나열
 레이블(인덱스) 보유 : 각 데이터에 이름표를 붙일 수 있다.
 동일 타입 : 하나의 series 안의 모든 데이터는 같은 타입
'''
simple_series = pd.Series([10, 20, 30, 40, 50])
print(simple_series)
'''
Series = Value(값) + Index(인덱스) + Name(이름)
'''

data_series = pd.Series(
    data=[10, 20, 30, 40, 50],  # 값 : 실제 저장된 데이터
    index=['Alice', 'Bob', 'Charlie', 'David', 'Eve'],  # 인덱스 : 각 값의 레이블
    name='Test_Score'  # 이름 : Series 전체의 이름
)
'''
data_series = pd.Series(data = None, index = None, dtype = None, name = None)

매개변수 설명
data = None 실제 데이터(필수)
    - 리스트, 딕셔너리, 스칼라 값, Numpy 배열
Index = None 인덱스 레이블(선택)
    - 기본값 0, 1, 2...
    - 리스트, 배열 등으로 지정


'''

'''
각 구성요소의 역할:
Value
    실제 데이터가 저장되는 부분
    Numpy 배열로 저장
    빠른 수치 연산 가능
Index
    각 값을 식별하는 레이블
    기본값 : 0, 1, 2, ...(정수)
    사용자 정의 가능 (문자열, 날짜 등)
Name
    Series 전체를 설명하는 이름
    선택사항 (없어도 됨.)
    DataFrame 결합 시 컬럼명이 됨!
'''
int_series = pd.Series([1, 2, 3, 4, 5])
print(f'Integer Series dtype: {int_series}')  # int64

float_series = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5])
print(f'float Series dtype: {float_series}')  # float64

str_series = pd.Series(['Apple', 'Banana', 'Cherry'])
print(f'string Series dtype: {str_series}')  # object

bool_series = pd.Series([True, False, True])
print(f'Boolean Series dtype: {bool_series}')  # bool

mixed_series = pd.Series([1, 2.5, 3])
print(f'Mixed Series dtype: {mixed_series}')  # 자동 변환

# 리스트 생성
temp_list = [15.5, 17.2, 18.9, 19.1, 20.1]
temp = pd.Series(temp_list, name='4월 기온')
print(temp)

date = pd.date_range('2025-04-01', periods=5)
print(date)
temp_date = pd.Series(temp_list, index=date, name='4월 기온')
print(temp_date)

product = {
    '노트북': 15,
    '마우스': 40,
    '키보드': 20
}

product_series = pd.Series(product, name='현재 재고')
print(product_series)

scalar_series = pd.Series(0, index=['월', '화', '수', '목'], name='판매량')
print(scalar_series)
print()

test_scores = pd.Series(
    data=[85, 86, 59, 97, 65],
    index=['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    name='Exam'
)

print('=== Series 속성 전체 ===')
# 1. value - 실제 데이터(Numpy 배열)
# 놓쳤음!!!


# 인덱싱과 슬라이싱

'''
# 인덱싱
Series에서 특정 데이터를 선택하는 방법
위치 가반 0, 1, 2
레이블 기반 : 인덱스 이름으로 접근
'''
sales = pd.Series([100, 200, 150, 300],
                  index=['Mon', 'Tue', 'Wed', 'Thu'],
                  mane='Daily_Sales'
                  )
wed_sales = sales['Wed']
print('수요일 매출', wed_sales)

selected_days = sales[['Mon', 'Wed', 'Thu']]
print(selected_days)

# wed_sales2 = sales[2]
# print('수요일 매출', wed_sales2)  에러난데 숫자로 접근하는 방법이

# 해결 방안
print('sales.iloc 수요일 매출', sales.iloc[2])
print('sales.loc 수요일 매출', sales.loc['Wed'])

# 슬라이싱
print('sales.loc 처음부터 수요일포함 매출\n', sales.loc[:'Wed'])
print('sales.loc 수요일부터 끝까지 매출\n', sales.loc['Wed':])
print('sales.iloc 처음부터 끝까지 매출', sales.iloc[:])
print('sales.iloc 역순 매출', sales.iloc[::-1])  # step(간격) 가능

# Boolean 인덱싱
sales = pd.Series(
    [100, 200, 150, 300],
    index=['Mon', 'Tue', 'Wed', 'Thu'],
    mane='Daily_Sales'
)

condition = sales >= 200
print(condition)
print()

# result = sales[sales >= 200]
result = sales[sales >= 200]  # ??

# 비교 연산자 아니 그냥 다 놓치는거 그냥 나중에 복붙해
print('sales[sales == 200]')
print(sales[sales == 200])

print('sales[sales >= 150]')
print(sales[sales >= 150])

print('sales[sales == 200]')
print(sales[sales == 200])

# 복합조건
sales = pd.Series(
    [100, 200, 150, 300, 250],
    index=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
    mane='Daily_Sales'
)

weekday_high = sales[(sales >= 200) & (sales.index != 'Fri')]
print('weekday_high')
print(weekday_high)
print()

# 인덱스 안에 'Mon', 'Fri' 있는 것만 출력
weeked_or_low = sales[(sales < 200) | (sales.index.isin(['Mon', 'Fri']))]
print('weeked_or_low')
print(weeked_or_low)
print()

# 벡터와 연산
prices = pd.Series(
    [3000, 1500, 4000, 2000],
    index=['apple', 'banana', 'orange', 'grape'],
    name='Price'
)
print('500원 추가:')
print(prices + 500)

print('1000원 할인:')  # 기존 가격에서 변동
print(prices - 1000)

print('20% 할인:')
print(prices * 0.8)

print('반값 세일:')
print(prices / 2)

''' 과일 가격표 뭐시기 복붙 '''
# a에 바나나 2천원, b에 포도 2500원

# Nan 결측값 처리하며 연산하기
# 1. fill_value 사용 (a에 없는 값을 b에서 가져와 채운다)

# Grape 2500 - 0으로 계산됨

# 2. reindex로 먼저 맞추기 ()안에 있는 인덱스를 기준으로 값을 0으로 지정. (b)

# 3. dropna로 결측값 제거 후 연산

# 비교 연산
is_b_cheaper = store_a > store_b
print('B 상점이 더 저렴한 제품')
print(is_b_cheaper)

# 저렴한 상점의 가격만 선택
best_prices = store_a.where(is_b_cheaper, store_b)
print('best_prices\n', best_prices)

# 통계 함수
'''
데이터의 특징을 숫자로 요약하는 것
import pandas as pd
import numpy as np
'''
# 한 달간 일일 매출 데이터
daily_sales = pd.Series([
    302, 423, 123, 437, 890,
    124, 781, 920, 478, 901,
    241, 891, 123, 678, 912,
    367, 894, 355, 123, 674,
    891, 234, 678, 943, 524,
    782, 394, 327, 891, 237],
    index=pd.date_range('2025-09-01', periods=30),
    name='Sales'
)
# 평균(mean)
mean_value = daily_sales.mean()
print(f'평균: {mean_value:.2f}만원')

# 중앙값(median)
median_value = daily_sales.median()
print(f'중앙값: {median_value}만원')

# 최빈값(mode)
mode_value = daily_sales.mode()
print(f'최빈값: \n {mode_value}만원')

# 산포도 측정 : 데이터가 얼마나 퍼져있는지 알려주는 통계량
print('=== 산포도 측정 ===')
max_valrue = daily_sales.max()
min_value = daily_sales.min()

print(f'최댓값: {max_valrue}')
print(f'최솟값: {min_value}')

# 범위(Range) : 최댓값 - 최솟값
range_value = max_valrue - min_value
print(f'범위: {range_value}')

# 분산(Variance) 평균으로부터 떨어진 정도의 제곱의 평균
variance = daily_sales.var()
print(f'분산: {variance:.2f}')

# 표준편차 (Standard Deviation) : 분산의 제곱근
std_dev = daily_sales.std()
print(f'표준편차: {std_dev:.2f}')

# 표준편차 해석
print('표준편차 해석')

# 불확실
print(f'평균 표준편차: {mean_value - std_dev:.2f} ~ {mean_value + std_dev:.2f}')

# 한 번에 모든 통계
print(daily_sales.describe())  # 용량보고 2번 파일로 넘기기.
