import pandas as pd
import os
import re
'''발전량 일사량 파일 병합'''
power_df = pd.read_excel('generation_clean.xlsx')
power_df = power_df[['일시', '발전구분', '총량(KW)', '호기']]

irradiance_folder = r"C:/Users/alsl0/Documents/python/일조량1"
merged_df_list = []

# 폴더 내 파일명 목록 (확장자 제거)
irr_files_dict = {os.path.splitext(f)[0]: os.path.join(irradiance_folder, f)
                  for f in os.listdir(irradiance_folder)}

for gen_type in power_df['발전구분'].unique():
    # 발전구분에서 숫자와 # 제거하여 핵심 문자열 추출
    core_name = re.sub(r'[#\d\s]', '', gen_type)  # #, 숫자, 공백 제거

    # 핵심 문자열이 포함된 파일 찾기
    matched_files = [f for f in irr_files_dict.keys() if core_name in f]

    if not matched_files:
        print(f"⚠️ '{gen_type}'에 맞는 일조량 파일 없음")
        continue

    irr_df_list = []
    for file in matched_files:
        irr_df = pd.read_excel(irr_files_dict[file])
        irr_df = irr_df[['일시', '합계 일사량(MJ/m2)']]  # 실제 컬럼명 확인
        irr_df_list.append(irr_df)

    irr_df = pd.concat(irr_df_list, ignore_index=True)
    power_sub = power_df[power_df['발전구분'] == gen_type].copy()
    merged = pd.merge(power_sub, irr_df, on='일시', how='left')

    # 결측값 제거 (두 컬럼 모두)
    merged = merged.dropna(subset=['총량(KW)', '합계 일사량(MJ/m2)'])

    # 0 제거 (두 컬럼 모두 0이 아닌 행만 남기기)
    merged = merged[(merged['총량(KW)'] != 0) & (merged['합계 일사량(MJ/m2)'] != 0)]
    merged_df_list.append(merged)

final_df = pd.concat(merged_df_list, ignore_index=True)
final_df.to_excel("발전량_일조량_병합.xlsx", index=False)
print("✅ 병합 완료! 0과 결측값 제거 후 '발전량_일조량_병합.xlsx' 생성됨")
