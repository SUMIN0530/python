import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
'''누수 히트맵'''
# -----------------------------
# 한글 폰트 설정
# -----------------------------
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------
# 기본 경로 설정
# -----------------------------
base_path = r"C:\Users\alsl0\Documents\python\Update_set"
leak_path = os.path.join(base_path, "누수방지")

# -----------------------------
# SMAPE 계산 함수
# -----------------------------


def smape(a, f):
    a, f = np.array(a), np.array(f)
    denom = (np.abs(a) + np.abs(f)) / 2
    mask = denom != 0
    return np.abs(a[mask] - f[mask]) / denom[mask] * 100  # 개별 SMAPE_i


# -----------------------------
# 모든 지역 탐색
# -----------------------------
for region_folder in os.listdir(base_path):
    if "태양광" in region_folder and os.path.isdir(os.path.join(base_path, region_folder)):
        region_path = os.path.join(base_path, region_folder)
        leak_region_path = os.path.join(leak_path, region_folder)

        # 누수제거 파일 찾기
        leak_files = [f for f in os.listdir(
            leak_region_path) if "누수제거" in f and f.endswith(".xlsx")]
        if not leak_files:
            print(f"[스킵] {region_folder} 누수제거 파일 없음")
            continue
        leak_file_path = os.path.join(leak_region_path, leak_files[0])

        # 예측값/실제값/SMAPE 읽기
        df_pred = pd.read_excel(leak_file_path, sheet_name='예측결과')
        df_eval = pd.read_excel(leak_file_path, sheet_name='모델평가')
        if 'SMAPE_i' not in df_pred.columns:
            df_pred['SMAPE_i'] = smape(df_pred['실제값'], df_pred['예측값'])

        # 일사량 데이터 읽기
        weather_files = [f for f in os.listdir(
            region_path) if f.endswith('_data.xlsx')]
        if not weather_files:
            print(f"[스킵] {region_folder} 일사량 데이터 없음")
            continue
        df_weather_list = [pd.read_excel(os.path.join(
            region_path, f)) for f in weather_files]
        df_weather = pd.concat(df_weather_list, ignore_index=True)

        # 일사량 컬럼 통일
        if '합계 일사량(MJ/m2)' in df_weather.columns:
            df_weather.rename(columns={'합계 일사량(MJ/m2)': 'solar'}, inplace=True)
        elif '일사량(MJ/m2)' in df_weather.columns:
            df_weather.rename(columns={'일사량(MJ/m2)': 'solar'}, inplace=True)
        else:
            print(f"[스킵] {region_folder} 일사량 컬럼 없음")
            continue

        # 날짜 컬럼 형식 통일
        df_pred['일시'] = pd.to_datetime(df_pred['일시'], errors='coerce')
        df_weather['일시'] = pd.to_datetime(df_weather['일시'], errors='coerce')

        # 데이터 병합
        df_merged = pd.merge(
            df_pred, df_weather[['일시', 'solar']], on='일시', how='left')

        # 히트맵용 상관계수 계산
        heatmap_cols = ['solar', '실제값', '예측값', 'SMAPE_i']
        corr_df = df_merged[heatmap_cols].corr().round(2)

        # 히트맵 생성
        plt.figure(figsize=(6, 5))
        sns.heatmap(corr_df, annot=True, cmap='coolwarm',
                    vmin=-1, vmax=1, fmt=".2f")
        plt.title(f"{region_folder} Ensemble 기반 상관 히트맵", fontsize=13)
        plt.tight_layout()

        # -----------------------------
        # 이미지 저장: 누수제거 파일 있는 지역 폴더로
        # -----------------------------
        heatmap_path = os.path.join(
            leak_region_path, "Ensemble_heatmap_corr.png")
        plt.savefig(heatmap_path, dpi=300)
        plt.close()

        print(f"[완료] {region_folder} 히트맵 저장 → {heatmap_path}")
