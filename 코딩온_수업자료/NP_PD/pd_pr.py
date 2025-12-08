import numpy as np
import pandas as pd

# 실습 1. 조건 필터링 연습
df = pd.DataFrame({
    '이름': ['민준', '서연', '지후', '서준', '지민'],
    '점수': [78, 92, 85, 60, 88],
    '반': [1, 2, 1, 2, 1]
})

# 문제 1. 점수가 80이상인 학생만 추출
print('점수가 80이상인 학생만 추출')
print(df[df['점수'] >= 80])
print()

# 문제 2. 1반 학생 중 점수 85점 이상인 학생만 추출
print('1반 학생 중 점수 85점 이상인 학생만 추출')
print(df[(df['반'] == 1) & (df['점수'] >= 80)])
print()

# 문제 3. 이름이 서연 또는 지민인 학생만 추출
print('이름이 서연 또는 지민인 학생만 추출')
df1 = df[(df['이름'] == '서연') | (df['이름'] == '지민')]
# df1 = df[(df['이름'].isin(['서연', '지민']))]
print(df1)
print()

# 문제 4. 인덱스 0부터 재배열
print(df1.reset_index(drop=True))

# 문제 5. 점수가 80점 미만이거나 2반인 학생 추출
print('점수가 80점 미만이거나 2반인 학생 추출')
df2 = df[(df['반'] == 2) | (df['점수'] < 80)]
print(df2)

# 문제 6. 5번의 컬럼에서 점수가 70점 이상인 학생만 추출하고 인덱스를 재정렬하여 추출
print('5번의 컬럼에서 점수가 70점 이상인 학생만 추출하고 인덱스를 재정렬하여 추출')
df3 = df2[df2['점수'] >= 70]
print(df3.reset_index(drop=True))

# ==============================================================================
# 실습 2. 행/열 추가 수정 삭제
df = pd.DataFrame({
    '이름': ['김철수', '이영희', '박민수'],
    '국어': [90, 80, 70]
})

# 문제 1. 수학점수 [95, 100, 88]을 새 열로 추가
print('수학 점수 추가')
df['수학'] = 95, 100, 88
print(df)
print()

# 문제 2. 이름 열 삭제
print('"이름"열 삭제')
df1 = df.drop('이름', axis=1)
print(df1)
print()

# ==================
df = pd.DataFrame({
    '제품': ['A', 'B'],
    '가격': [1000, 2000]
})

# 문제 3. 제품 C 가격 1500인 새 행 추가
new_row = pd.DataFrame([{'제품': 'C', '가격': 1500}])
df = pd.concat([df, new_row], ignore_index=True)
print('제품 C 추가')
print(df)
print()

df.loc[len(df)] = ['D', 2500]
print(df)

# 문제 4. 제품 A 삭제
df1 = df.drop(0)
print('제품 A 제거')
print(df1)
print()

# ==================================
df = pd.DataFrame({
    '과목': ['국어', '영어', '수학'],
    '점수': [85, 90, 78]
})

# 문제 5. 점수가 80미만인 행 삭제  -> 80이상인 애들 뽑아내기
print('80이상인 애들 뽑아내기')
df1 = df[df['점수'] >= 80]
print(df1)

print('점수가 80미만인 행 삭제')
df = df.drop(df[df['점수'] < 80].index)
print(df)
print()

# 문제 6. 학년 열(모두 1) 추가
print('학년 열 (1) 추가')
df['학년'] = 1
print(df)
print()

# ==================
df = pd.DataFrame({
    '이름': ['A', 'B'],
    '나이': [20, 22]
})

# 문제 7. 이름이 'C', 나이가 25, 키가 NaN(결측값)인 새 행을 추가
new_row = pd.DataFrame([{'이름': 'C', '나이': 25, '키': np.nan}])
df = pd.concat([df, new_row], ignore_index=True)
print(df)

# =====================================
df = pd.DataFrame({
    '부서': ['영업', '기획', '개발', '디자인'],
    '인원': [3, 2, 5, 1]
})

# 문제 8. 인원이 2명 이하인 행을 모두 삭제
print('인원이 2명 이하인 행을 모두 삭제')
df = df[df['인원'] > 2]
print(df)
print()

# 문제 9. '평가' 열을 새로 추가해 모든 값을 '미정'으로 채우세요.
df['평가'] = '미정'
print(df)
print()

# ===============================================================================
# 실습 3. 정렬
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'score': [88, 95, 70, 100]
})

# 문제 1. score 컬럼 기준으로 오름차순 정렬한 결과를 출력
print('점수 오름차순 정렬')
df_sorted_score = df.sort_values(by='score')
print(df_sorted_score)
print()

# 문제 2. score 컬럼 기준 내림차순으로 0부터 재정렬한 결과 출력
print('내림차순 재정렬')
df_sorted_score_r = df.sort_values(by='score', ascending=False).sort_index()
# df_sorted_score_r = df.sort_values(by='score', ascending=False).reset_index(drop=True)
print(df_sorted_score_r)
print()

# ===================================
df = pd.DataFrame({
    '이름': ['가', '나', '다', '라', '마'],
    '반': [2, 1, 1, 2, 1],
    '점수': [90, 85, 80, 95, 85]
})

# 문제 3. 반(class) 기준 오름차순, 같은 반 내에서는 점수(score) 기준 내림차순으로 정렬
print('오름차순으로 반 정렬, 내림차순으로 점수 정렬')
df_sorted_multi = df.sort_values(by=['반', '점수'], ascending=[True, False])
print(df_sorted_multi)
print()

# 문제 4. 열(컬럼) 이름을 알파벳순으로 정렬
print('알파벳 순으로 이름 정렬')
df_sorted_sp = df.sort_values(by='이름')
# df_sorted_sp = df.sort_index(axis=1)
print(df_sorted_sp)
print()

# ======================
df = pd.DataFrame({
    'value': [10, 20, 30, 40]
}, index=[3, 1, 4, 2])

# 문제 5. 인덱스 기준 오름차순 정렬
print('인덱스 기준 오름차순 정렬')
df_sorted_idx = df.sort_index()
print(df_sorted_idx)
print()

# 문제 6. 인덱스 기준 내림차순, value 컬럼 기준 오름차순 각각 출력
print('인덱스 기준 내림차순 정렬')
df_sorted_idx = df.sort_index(ascending=False)
print(df_sorted_idx)
print()

print('value 기준 오름차순')
df_sorted_vlu = df.sort_values(by='value')
print(df_sorted_vlu)
print()

# =============================================================
# 실습 4. groupby 연습문제

# 문제 1. 각 학년 별 평균 국어 점수
df = pd.DataFrame({
    'grade': [1, 2, 1, 2, 1, 3],
    'name': ['Kim', 'Lee', 'Park', 'Choi', 'Jung', 'Han'],
    'kor': [85, 78, 90, 92, 80, 75]
})

grade = df.groupby('grade')
print('각 학년별 평균 국어 점수')
print(grade['kor'].mean())
print()

# 문제 2. 반 별, 과목 별 학생수와 평균
df = pd.DataFrame({
    'class': [1, 1, 1, 2, 2, 2],
    'subject': ['Math', 'Math', 'Eng', 'Math', 'Eng', 'Eng'],
    'score': [80, 90, 85, 70, 95, 90]
})

print('반 별, 과목 별 학생수와 평균')
result = df.groupby(['class', 'subject'])['score'].agg(['count', 'mean'])
result.rename(columns={'mean': 'avg'}, inplace=True)
print(result)

# 문제 3. 지역 별, 판매자 별 판매액의 합계와 최대값
df = pd.DataFrame({
    'region': ['Seoul', 'Seoul', 'Busan', 'Busan', 'Daegu', 'Daegu'],
    'seller': ['A', 'B', 'A', 'B', 'A', 'A'],
    'sales': [100, 200, 150, 120, 130, 200]
})

print('지역 별, 판매자 별 판매액의 합계와 최대값')
result1 = df.groupby(['region', 'seller'])['sales'].agg(['sum', 'max'])
print(result1)
print()

# 문제 4. 팀 별, 포지션 별 결측치를 포함한 평균 점수
df = pd.DataFrame({
    'team': ['A', 'A', 'B', 'B', 'A', 'B'],
    'position': ['FW', 'DF', 'FW', 'DF', 'DF', 'FW'],
    'score': [3, 2, None, 1, 4, 2]
})

print('팀 별, 포지션 별 결측치를 포함한 평균 점수')
result = df.groupby(['team', 'position'], dropna=False)['score'].mean()
print(result)
print()

# 문제 5. 부서 별, 성 별 인원수와 총 연봉 합계
df = pd.DataFrame({
    'dept': ['HR', 'HR', 'IT', 'IT', 'Sales', 'Sales'],
    'gender': ['M', 'F', 'F', 'M', 'F', 'F'],
    'salary': [3500, 3200, 4000, 4200, 3000, 3100]
})

print('부서 별, 성 별 인원수와 총 연봉 합계')
result = df.groupby(['dept', 'gender'])['salary'].agg(['count', 'sum'])
result = df.groupby(['dept', 'gender']).agg(
    count=('salary', 'count'),
    total_salary=('salary', 'sum')
)
print(result)
