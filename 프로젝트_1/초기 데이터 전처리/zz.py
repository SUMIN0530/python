import pandas as pd
import glob
import os

'''<데이터 삽입>'''

# 1. 일조량_기온 파일 불러오기
env_file = '일조량_기온.xlsx'
df_env = pd.read_excel(env_file)

# 2. 일조량 파일이 있는 폴더
sun_folder = '일조량1'  # 실제 폴더 경로로 변경
sun_files = glob.glob(os.path.join(sun_folder, '*.xlsx'))

# 3. 각 일조량 파일에 환경 데이터 추가
for file in sun_files:
    basename = os.path.basename(file)
    region_name = basename.split('_')[1].split('(')[0]  # 파일명 규칙에 맞게 수정

    # 해당 지역 환경 데이터 선택
    env_region = df_env[df_env['지역'] == region_name]
    if env_region.empty:
        print(f"{region_name} 환경 데이터 없음")
        continue

    # 일조량 파일 읽기
    df_sun = pd.read_excel(file)

    # merge: 문자열 그대로 YYYY-MM-DD 기준으로 매칭
    df_sun = pd.merge(
        df_sun,
        env_region[['일시', '평균기온(°C)', '평균 풍속(m/s)', '평균 상대습도(%)']],
        on='일시',
        how='left'
    )

    # 파일 덮어쓰기
    df_sun.to_excel(file, index=False)
    print(f"{file} 업데이트 완료")
