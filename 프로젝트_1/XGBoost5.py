import os
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
'''MAE, RSME, R^2, MAPE, SMAPE (%)'''
# =============================
# 한글 폰트 설정
# =============================
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

root_path = r"C:/Users/alsl0/Documents/python/지역별_발전량_비교"
regions = [d for d in os.listdir(
    root_path) if os.path.isdir(os.path.join(root_path, d))]
print(f"📂 탐색된 지역: {regions}")

summary_list = []  # 전체 요약 저장용

# =============================
# 보조 함수
# =============================


def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-6, None))) * 100


def SMAPE(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-6))


def get_season(month):
    if month in [12, 1, 2]:
        return "겨울"
    elif month in [3, 4, 5]:
        return "봄"
    elif month in [6, 7, 8]:
        return "여름"
    else:
        return "가을"


# =============================
# 지역별 반복
# =============================
for region in regions:
    print(f"\n==================== {region} ====================")
    base_path = os.path.join(root_path, region)
    train_files = [
        os.path.join(base_path, "23_train_data.xlsx"),
        os.path.join(base_path, "24_test_data.xlsx")
    ]
    test_path = os.path.join(base_path, "25_test_data.xlsx")

    # 필요한 파일 체크
    if not all(os.path.exists(f) for f in train_files) or not os.path.exists(test_path):
        print(f"⚠️ {region}: 필요한 파일이 없습니다. 건너뜀.")
        continue

    # =============================
    # 데이터 로드
    # =============================
    train_df = pd.concat([pd.read_excel(f, engine="openpyxl")
                         for f in train_files], ignore_index=True)
    test_df = pd.read_excel(test_path, engine="openpyxl")

    X_train = train_df[['합계 일사량(MJ/m2)', '평균기온(°C)']]
    y_train = train_df['총량(KWh)']
    X_test = test_df[['합계 일사량(MJ/m2)', '평균기온(°C)']]
    y_test = test_df['총량(KWh)']
    dates_test = pd.to_datetime(test_df['일시'], errors='coerce')

    # =============================
    # 숫자형 변환 & 결측치 처리
    # =============================
    X_train[X_train.columns] = X_train.apply(pd.to_numeric, errors='coerce')
    X_test[X_test.columns] = X_test.apply(pd.to_numeric, errors='coerce')
    y_train = pd.to_numeric(y_train, errors='coerce')
    y_test = pd.to_numeric(y_test, errors='coerce')

    for df in [X_train, X_test]:
        df.replace(0, np.nan, inplace=True)
        df.interpolate(method='linear', inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
    y_train = y_train.replace(0, np.nan).interpolate().ffill().bfill()
    y_test_interp = y_test.replace(0, np.nan).interpolate().ffill().bfill()

    # =============================
    # 스케일링
    # =============================
    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)
    y_train_s = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    # =============================
    # XGBoost + RandomizedSearch
    # =============================
    param_grid = {
        'max_depth': [4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [200, 500, 800],
        'gamma': [0, 0.1, 0.3],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0]
    }
    model = xgb.XGBRegressor(random_state=42, eval_metric='rmse')
    search = RandomizedSearchCV(
        model, param_grid, n_iter=50, scoring='r2', cv=3, n_jobs=-1, verbose=0)
    search.fit(X_train_s, y_train_s)
    best_model = search.best_estimator_

    # =============================
    # 예측
    # =============================
    y_pred_s = best_model.predict(X_test_s)
    y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()

    # =============================
    # 전체 지표 계산 (KWh + %)
    # =============================
    mae = mean_absolute_error(y_test_interp, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_interp, y_pred))
    r2 = r2_score(y_test_interp, y_pred)
    mape = MAPE(y_test_interp, y_pred)
    smape = SMAPE(y_test_interp, y_pred)

    mean_actual = np.mean(y_test_interp)
    mae_pct = mae / mean_actual * 100
    rmse_pct = rmse / mean_actual * 100

    print(f"\n{region} 전체 지표:")
    print(f"MAE: {mae:.2f} ({mae_pct:.2f}%), RMSE: {rmse:.2f} ({rmse_pct:.2f}%), R²: {r2:.4f}, MAPE: {mape:.2f}%, SMAPE: {smape:.2f}%")

    # =============================
    # 월별·계절별 분석
    # =============================
    temp = pd.DataFrame({
        '일시': dates_test,
        '실제발전량': y_test_interp.ravel(),
        '예측발전량': y_pred
    })
    temp['월'] = temp['일시'].dt.month
    temp['계절'] = temp['월'].apply(get_season)

    monthly_stats = temp.groupby('월').apply(lambda g: pd.Series({
        'MAE': mean_absolute_error(g['실제발전량'], g['예측발전량']),
        'MAPE': MAPE(g['실제발전량'], g['예측발전량']),
        'SMAPE': SMAPE(g['실제발전량'], g['예측발전량'])
    })).reset_index()

    seasonal_stats = temp.groupby('계절').apply(lambda g: pd.Series({
        'MAE': mean_absolute_error(g['실제발전량'], g['예측발전량']),
        'MAPE': MAPE(g['실제발전량'], g['예측발전량']),
        'SMAPE': SMAPE(g['실제발전량'], g['예측발전량'])
    })).reset_index()

    # =============================
    # 결과 폴더 저장
    # =============================
    result_path = os.path.join(base_path, "결과")
    os.makedirs(result_path, exist_ok=True)

    with pd.ExcelWriter(os.path.join(result_path, f"{region}_상세분석.xlsx")) as writer:
        temp.to_excel(writer, sheet_name='일자별', index=False)
        monthly_stats.to_excel(writer, sheet_name='월별분석', index=False)
        seasonal_stats.to_excel(writer, sheet_name='계절별분석', index=False)

    plt.figure(figsize=(14, 5))
    plt.plot(temp['일시'], temp['실제발전량'], label='실제값', marker='o', linewidth=1)
    plt.plot(temp['일시'], temp['예측발전량'], label='예측값', marker='s', linewidth=1)
    plt.title(f"{region} 일자별 발전량 예측 비교")
    plt.xlabel('일시')
    plt.ylabel('발전량(KWh)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, f"{region}_예측그래프.png"), dpi=300)
    plt.close()

    # =============================
    # 전체 요약 저장
    # =============================
    summary_list.append({
        '지역': region,
        'MAE(KWh)': mae,
        'MAE(%)': mae_pct,
        'RMSE(KWh)': rmse,
        'RMSE(%)': rmse_pct,
        'R²': r2,
        'MAPE(%)': mape,
        'SMAPE(%)': smape,
        '최적파라미터': str(search.best_params_)
    })

# =============================
# 전체 요약 엑셀 저장
# =============================
summary_df = pd.DataFrame(summary_list).sort_values('MAE(KWh)')
summary_path = os.path.join(root_path, "전체_요약.xlsx")
summary_df.to_excel(summary_path, index=False)
print(f"\n✅ 전체 지역 요약 저장 완료: {summary_path}")
