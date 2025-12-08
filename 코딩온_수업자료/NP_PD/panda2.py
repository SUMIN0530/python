import pandas as pd
# Data Frame
'''
pandas의 핵심 자료 구조
2차원 표 형태의 데이터를 다루는 객제

핵심 개념
    2차원 구조 : 행, 열로 구성
    Series의 집합 : 여러 개의 Series가 열로 배치된 형태
    레이블 기반 : 각 행과 열에 이름(레이블)을 붙일 수 있음
    각 열마다 다른 데이터 타입 가능
'''
# Data Frame의 구성 요소
test_data = pd.DataFrame(
    # 데이터(value) - 2차원 배열
    data=[
        ['김철수', 27, 'Dev', 4500],
        ['이영희', 23, 'Hr', 4800]
    ],
    # 행 인덱스(index) - 각 행의 레이블
    index=['E001', 'E002'],
    # 열 이름(columns) - 각 열의 레이블
    columns=['name', 'age', 'department', 'salary']
)

print(' === Data Frame')
print(test_data)
print('===구성 요소 분석===')
print('행 인덱스', test_data.index.tolist())
print('열 이름', test_data.columns.tolist())
print('데이터 형태', test_data.shape)  # (2, 4)
print('행 개수', test_data.shape[0])  # 2
print('열 개수', test_data.shape[1])  # 4
print('전체 셀 개수', test_data.size)  # 8?

# DataFrane vs Series
'''
Series
 1차원

DataFrame
 2차원(series들의 묶음)
 데이터(Values) - 2차원 배열
 행 인덱스(Index) - 각 행의 레이블
 열 이름(Columns) - 각 열의 레이블
'''

df_default = pd.DataFrame({
    # 'name': ['Kim', 'Lee', 'Park'],
    'age': [25, 26, 27]
}, index=['Kim', 'Lee', 'Park'])  # 인덱스 이름 변환


print(f'{df_default}')
print(f'인덱스: {df_default.index}')
print(f'열 이름: {df_default.columns.tolist()}')

# CSV
'''
CSV(Comma-Separated Values) 가장 널리 사용되는 데이터 파일 형식

특징
 쉼표(,)로 값 구분
 텍스트 파일이므로 어디서나 열람 가능
 가볍고 빠름
 Excel, Google Sheets 등과 호환 가능

예시
 name, age, city, salary
 John,  25, Seoul, 50000
 Jane,  30, Busan, 60000
 Park,  35, Daegu, 55000
 '''
# 샘플 csv 파일 생성
samole_data = pd.DataFrame({
    'name': ['John', 'Jane', 'Park'],
    'age': [25, 30, 35],
    '도시': ['서울', '부산', '대구'],
    'Salary': [50000, 60000, 55000]
})

# UTF-8로 저장 (기본값, 권장)
# samole_data.to_csv('sample_data.csv', encoding='UTF-8')  # 또 놓침 그냥 아 모르겠다.

# CP949로 저장 (Window 한글)


samole_data.to_csv('sample_data.csv', index=False)
df = pd.read_csv('sample_data.csv')
# df = pd.read_csv('sample_data.csv', encoding='cp949')
print('===데이터 파일 읽기===')
print(df)
print(f'데이터 타입:\n {df.dtypes}')
print(f'데이터 크기:\n {df.shape}')

# sep - 구분자 설정 (기본은 ,)
samole_data.to_csv('tap_separated.txt', sep='\t', index=False)
df_tap = pd.read_csv('tap_separated.txt', sep='\t')
print('=== CSV sep=tap 파일 읽기===')
print(df_tap)
# head 처음 5개 행 가져옴(기본값) --- github에서 확인
print(df_tap.head)


# Excel
'''
엑셀은 마이크로소프트의 스프레드시트
특징
 여러 시트(Sheet) 지원
 서식, 수식 포함 가능
 비지니스에서 가장 많이 사용
 확장자 (.xlsx)최신 (.xls)구버전
 pip install openpyxl
'''

samole_data = pd.DataFrame({
    'name': ['John', 'Jane', 'Park'],
    'age': [25, 30, 35],
    '도시': ['서울', '부산', '대구'],
    'Salary': [50000, 60000, 55000]
})

samole_data.to_excel('sample_data.xlsx', index=False, sheet_name='Default')
print('샘플 엑셀 파일 생성 완료')

df_excel = pd.read_excel('sample_data.xlsx')
print('===Excel 파일 읽기===')
print(df_excel)

# 여러 시트 다루기
with pd.ExcelWriter('multi_sheet.xlsx') as writer:
    samole_data.to_excel(writer, sheet_name='Default1', index=False)
    samole_data['name'].to_excel(writer, sheet_name='name', index=False)
print('2개의 시트를 가진 엑셀 파일 생성 완료')

# JSON
'''
JSON(JavaScript Object Notation)
웹에서 많이 사용되는 데이터 형식
'''
samole_data = pd.DataFrame({
    'name': ['John', 'Jane', 'Park'],
    'age': [25, 30, 35],
    '도시': ['서울', '부산', '대구'],
    'Salary': [50000, 60000, 55000]
})

samole_data.to_json('sample_data.json', orient='records', indent=2)
print('JSON 파일 저장')

df_json = pd.read_json('sample_data.json')
print('===json 파일 읽기===')
print(df_json)

data = {
    '이름': ['홍길동', '이순신', '김유신', '강감찬', '장보고', '이방원'],
    '나이': [23, 35, 31, 40, 28, 34],
    '직업': ['학생', '군인', '장군', '장군', '상인', '왕자']
}
df = pd.DataFrame(data)

# 인덱싱
print('===인덱싱===')
print(df['이름'])
print(df[['이름', '나이', '직업']])
print()

# 슬라이싱
print(df[1:3])
print()
print(df[-2:])
print()

# DaraFrame의 슬라이싱은 행(Row) 기준으로 동작한다.
# 열 단위 슬라이싱은 명시적으로 지정
print(df[-2:]['이름'])

# iloc
print('===iloc===')
print(df)
print(df.iloc[0])       # 0번째 행 전체
print(df.iloc[:, 1])    # 1번째 열 전체
print(df.iloc[[0, 2, 4], [0, 2]])

# loc
print('===loc===')
print(df.loc[0])    # 0번째 행 전체
print(df.loc[:, '나이'])
print(df.loc[1:3, ['이름', '나이']])  # 주의!! 1~3행(포함)까지
