import pandas as pd
import glob
import os
'''<추가된 식의 호기별 이론발전량>'''
# -----------------------------
# 경로 설정
input_folder = r'output_files'      # 일조량 파일들이 있는 폴더
area_file = 'area.xlsx'             # 발전구분+호기별 면적
output_folder = r'theoretical_output'  # 결과 저장 폴더

# 표준 모듈 효율 및 보정 계수
n0 = 0.175
alpha = 0.0045
beta = 0.012
gamma = 0.0063

# 월별 계절 보정
f_month = {
    1: 0.9, 2: 0.95, 3: 1.0, 4: 1.05, 5: 1.1, 6: 1.1,
    7: 1.05, 8: 1.0, 9: 0.95, 10: 0.9, 11: 0.85, 12: 0.8
}
# -----------------------------

# 면적 파일 불러오기
df_area = pd.read_excel(area_file)

# 일조량 파일 목록
sun_files = glob.glob(os.path.join(input_folder, '*.xlsx'))

for file in sun_files:
    basename = os.path.basename(file)

    # 발전구분 추출 (파일명에 포함된 규칙)
    try:
        region_name = basename.split('(')[1].split(')')[0]
    except IndexError:
        print(f"[스킵] 파일명 규칙 오류: {basename}")
        continue

    # 해당 발전구분 면적 데이터만 추출
    df_region_area = df_area[df_area['발전구분'] == region_name]

    if df_region_area.empty:
        print(f"[스킵] 면적 정보 없음: {region_name}")
        continue

    # 1) 일조량 파일 읽기
    df_sun = pd.read_excel(file)
    df_sun['일시'] = pd.to_datetime(df_sun['일시'])
    df_sun['Month'] = df_sun['일시'].dt.month

    # 2) 각 호기별로 행 복사 → 호기별 면적 적용
    df_list = []
    for _, row_area in df_region_area.iterrows():
        df_copy = df_sun.copy()
        df_copy['호기'] = row_area['호기']
        df_copy['면적'] = row_area['면적(m2)']
        df_list.append(df_copy)

    df_all = pd.concat(df_list, ignore_index=True)

    # 3) n(T,V,H,Month) 계산
    df_all['n'] = (
        n0
        * (1 - alpha * (df_all['평균기온(°C)'] - 25))
        * (1 + beta * df_all['평균 풍속(m/s)'])
        * (1 - gamma * df_all['평균 상대습도(%)'])
        * df_all['Month'].map(f_month)
    )

    # 4) 이론발전량 계산
    df_all['이론발전량(KW)'] = df_all['합계 일사량(MJ/m2)'] * df_all['면적'] * df_all['n']

    # 5) 결과 저장
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(
        output_folder, f"{region_name}_theoretical.xlsx")
    df_all.to_excel(output_path, index=False)
    print(f"{basename} → {region_name} 호기별 이론발전량 계산 완료")

# =====================실제 발전량이랑 비교 => 불량률 산출=================================
'''다른 파일에서 실행'''
# -----------------------------
# 경로 설정
theoretical_folder = r'theoretical_output'   # 호기별 이론발전량 파일
actual_file = 'generation_clean.xlsx'       # 실제 발전량 파일
output_folder = r'final_output'              # 최종 결과 저장 폴더
# -----------------------------

# 실제 발전량 불러오기
df_actual = pd.read_excel(actual_file)
df_actual['일시'] = pd.to_datetime(df_actual['일시'])

# 호기별 이론발전량 파일 목록
theo_files = glob.glob(os.path.join(theoretical_folder, '*.xlsx'))

for file in theo_files:
    basename = os.path.basename(file)
    df_theo = pd.read_excel(file)
    df_theo['일시'] = pd.to_datetime(df_theo['일시'])

    # 1) 실제 발전량과 병합 (호기 + 일시 기준)
    df_merge = pd.merge(df_theo, df_actual, on=['호기', '일시'], how='left')

    # 2) 불량률 계산
    df_merge['불량률(%)'] = ((df_merge['이론발전량(KW)'] - df_merge['총량(KW)'])
                          / df_merge['이론발전량(KW)']) * 100

    # 3) NaN 처리: 실제 발전량 없으면 불량률 NaN으로 둠
    df_merge['불량률(%)'] = df_merge['불량률(%)'].fillna(0)

    # 4) 결과 저장
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, basename)
    df_merge.to_excel(output_path, index=False)
    print(f"{basename} → 불량률 계산 완료")
