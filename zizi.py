import pandas as pd
import glob
import os
'''일조량 결측값 제거 후 새 파일 생성'''
# ================================
# 폴더 경로 설정
# ================================
folder_path = r"C:/Users/alsl0/Documents/python/일조량"
excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))

# ================================
# 전체 결측값 합계 변수 초기화
# ================================
total_missing_all_files = 0

# ================================
# 파일별 결측값 개수 확인
# ================================
for file_path in excel_files:
    df = pd.read_excel(file_path)

    # 파일 전체 결측값 개수
    missing_count = df.isnull().sum().sum()
    total_missing_all_files += missing_count  # 12개
    print(f"결측값 개수: {total_missing_all_files}")

    # 결측값 제거
    df_clean = df.dropna()

    # 1️⃣ 일시 열을 datetime으로 변환 (자동 인식)
    df_clean['일시'] = pd.to_datetime(df['일시'], errors='coerce')

    # 2️⃣ 날짜만 남기기 (YYYY-MM-DD)
    df_clean['일시'] = df['일시'].dt.strftime('%Y-%m-%d')

    # 새 파일로 저장 (원본 유지) 주의!!
    # 단순히 결측값을 제거할 경우 마지막 파일 내용만 저장
    # 각 파일마다 새로운 파일 생성
    clean_file_name = "clean_" + os.path.basename(file_path)
    clean_file_path = os.path.join(folder_path, clean_file_name)
    df_clean.to_excel(clean_file_path, index=False)

'''일사량 결측값 제거 전 후 행 개수 확인용'''  # 반드시 다른 파일에서 실행할 것.
# ================================
# 폴더 경로 설정
# ================================
folder_path = r"C:/Users/alsl0/Documents/python/일조량"
excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
'''
    일조량 폴더 / 결측값 제거 전 12036 행
    일조량_1 폴더 / 결측값 제거 후 12024 행
'''
# ================================
# 전체 결측값 합계 변수 초기화
# ================================
total_missing_all_files = 0
total_rows_all_files = 0

# ================================
# 파일별 결측값 개수 확인
# ================================
for file_path in excel_files:
    df = pd.read_excel(file_path)

    # 파일 전체 결측값 개수
    missing_count = df.isnull().sum().sum()
    total_missing_all_files += missing_count  # 12개
    print(f"결측값 개수: {total_missing_all_files}")

    # 행 수 확인
    row_count = len(df)
    total_rows_all_files += row_count
    print(f"{file_path} 행 수: {row_count}")

# ================================
# 전체 파일 행 수 합계
# ================================
print(f"\n결측값 제거 후 11개 파일 전체 행 수 합계: {total_rows_all_files}")


# ======================='''<발전량 결측치 제거>'''================================
# ======================================
# 엑셀 파일 읽기
# ======================================
df = pd.read_excel('generation.xlsx')

# 1️⃣ 일시 열을 datetime으로 변환 (자동 인식)
df['일시'] = pd.to_datetime(df['일시'], errors='coerce')

# 2️⃣ 날짜만 남기기 (YYYY-MM-DD)
df['일시'] = df['일시'].dt.strftime('%Y-%m-%d')
# ======================================
# 결측값 + 0 값 제거 (총량(KW) 기준)
# ======================================
df_clean = df.dropna(subset=['총량(KW)'])  # 결측값 제거
df_clean = df_clean[df_clean['총량(KW)'] != 0]  # 0 값 제거

# ======================================
# 새 파일로 저장
# ======================================
clean_file_path = r"C:/Users/alsl0/Documents/python/generation_clean.xlsx"
df_clean.to_excel(clean_file_path, index=False)

# ================================
# 제거 후 행 수 확인
# ================================
print(f"결측값 + 0 제거 후 행 수: {len(df_clean)}")
