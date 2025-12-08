import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import glob
import os
'''<합계일사량 그래프 및 엑셀 파일 분류>'''
# 1️⃣ 한글 폰트 설정
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 2️⃣ 경로 설정
base_folder = r"C:/Users/alsl0/Documents/python/일조량1"  # 지역별 일조량 들어있는 폴더 경로
excel_files = glob.glob(os.path.join(base_folder, "*.xlsx"))

# 3️⃣ 결과 저장 폴더
output_folder = os.path.join(base_folder, "그래프")  # 일조량이 들어있는 폴더 내 폴더 생성
os.makedirs(output_folder, exist_ok=True)

for file_path in excel_files:
    # 지역명 추출
    region_name = os.path.splitext(os.path.basename(file_path))[0]

    # 지역별 폴더 생성
    region_folder = os.path.join(output_folder, region_name)
    os.makedirs(region_folder, exist_ok=True)

    # 엑셀 파일 읽기
    df = pd.read_excel(file_path)
    df['일시'] = pd.to_datetime(df['일시'])
    df['연도'] = df['일시'].dt.year
    df['월'] = df['일시'].dt.month
    df['일'] = df['일시'].dt.day

    # 그래프 초기화
    plt.figure(figsize=(12, 5))

    # 4️⃣ 연도별 처리
    for year in sorted(df['연도'].unique()):
        year_df = df[df['연도'] == year].copy()

        # 연도별 엑셀 파일
        excel_path = os.path.join(region_folder, f"{year}.xlsx")

        # openpyxl 엔진으로 월별 시트 작성
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='w') as writer:
            monthly_avg_list = []
            months = sorted(year_df['월'].unique())

            for month in months:
                month_df = year_df[year_df['월'] == month].copy()
                if month_df.empty:
                    continue

                # 일별 합계 일사량 계산
                daily_sum = month_df.groupby(month_df['일시'].dt.date)[
                    '합계 일사량(MJ/m2)'].sum().reset_index()
                daily_sum.columns = ['일자', '합계 일사량(MJ/m2)']

                # 평균, 표준편차 계산
                avg = daily_sum['합계 일사량(MJ/m2)'].mean()
                std = daily_sum['합계 일사량(MJ/m2)'].std()

                # 🌟 일별 편차 계산 (일사량 - 월 평균)
                daily_sum['일별 편차(MJ/m2)'] = daily_sum['합계 일사량(MJ/m2)'] - avg

                # 평균, 표준편차 행 추가
                summary = pd.DataFrame({
                    '일자': ['평균', '표준편차'],
                    '합계 일사량(MJ/m2)': [avg, std],
                    '일별 편차(MJ/m2)': [None, None]
                })

                # 결합
                month_sheet = pd.concat(
                    [daily_sum, summary.dropna(axis=1, how='all')],
                    ignore_index=True
                )

                # 시트에 기록
                sheet_name = f"{month}월"
                month_sheet.to_excel(
                    writer, sheet_name=sheet_name, index=False)

                # 그래프용 월별 평균 저장
                monthly_avg_list.append(avg)

        # 5️⃣ 그래프용 월별 평균 꺾은선 추가
        plt.plot(months, monthly_avg_list, marker='o', label=f"{year}년")

    # 6️⃣ 그래프 설정
    plt.title(f"{region_name} 월별 평균 일사량 (연도별)")
    plt.xlabel("월")
    plt.ylabel("평균 합계 일사량 (MJ/m²)")
    plt.xticks(range(1, 13))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 그래프 저장
    graph_path = os.path.join(region_folder, f"{region_name}_월별_평균_일사량.png")
    plt.savefig(graph_path)
    plt.close()

print("✅ 모든 지역의 엑셀 파일(월별 시트 포함)과 그래프 생성 완료.")

# ================================================================================================
'''<발전구분별 호기별 산점도 그래프>'''
# 한글 폰트 설정
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 엑셀 파일 읽기
file_path = '발전량_일조량_병합.xlsx'
df = pd.read_excel(file_path)

# 저장 폴더 생성
save_folder = '개별_그래프'
os.makedirs(save_folder, exist_ok=True)

# 발전구분과 호기별로 그룹화
groups = df.groupby(['발전구분', '호기'])

# 그룹별로 개별 그래프 생성 및 저장
for (plant, unit), group in groups:
    plt.figure(figsize=(10, 6))
    plt.scatter(group['합계 일사량(MJ/m2)'], group['총량(KW)'],
                alpha=0.7)

    plt.xlabel('합계 일사량(MJ/m2)')
    plt.ylabel('총량(KW)')
    plt.title(f'{plant} - {unit} 발전량 vs 일사량')
    plt.grid(True)
    plt.tight_layout()

    # 파일명 생성
    file_name = f'{plant}_{unit}.png'
    file_path_save = os.path.join(save_folder, file_name)

    # 저장
    plt.savefig(file_path_save, dpi=300)
    plt.close()  # 메모리 절약을 위해 닫기
