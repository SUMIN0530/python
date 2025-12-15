import pandas as pd
'''호기에 따른 이상치 여부'''
# 1️⃣ 데이터 읽기
df = pd.read_excel('발전량_일조량_병합.xlsx')

# 2️⃣ 발전효율 계산 (총량 / 일사량)
df['발전효율'] = df['총량(KW)'] / df['합계 일사량(MJ/m2)']

# 3️⃣ 호기별 이상치 탐지 함수


def detect_outliers(group):
    Q1 = group['발전효율'].quantile(0.25)
    Q3 = group['발전효율'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    group['이상치여부'] = (group['발전효율'] < lower) | (group['발전효율'] > upper)
    return group


# 4️⃣ 발전소+호기 그룹 키 만들기
df['발전소_호기'] = df['발전구분'] + ' ' + df['호기'].astype(str)

# 5️⃣ 호기별 이상치 여부 계산
df = df.groupby('발전소_호기', group_keys=False).apply(detect_outliers)

# 6️⃣ 정상 / 이상치 개수 출력
정상_count = (~df['이상치여부']).sum()
이상치_count = df['이상치여부'].sum()

print(f"✅ 정상 데이터 개수: {정상_count}행")
print(f"⚠️ 이상치 데이터 개수: {이상치_count}행")
