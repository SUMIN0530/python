import os
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
import matplotlib.pyplot as plt
from matplotlib import rc
'''XGBoost, Light BGM 앙상블 1:1 어쩌고 예측값과 기타 등등 구함 + 그래프'''
# 한글 폰트
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# =========================
# 파일 경로
# =========================
base_path = r"C:\Users\alsl0\Documents\python\Update_set"

# =========================
# 데이터 로딩
# =========================

name = '예천태양광_1'


def load_data():
    files = [os.path.join(base_path, f"{name}",
                          f"{name}_{year}_data.xlsx") for year in range(22, 26)]
    dfs = [pd.read_excel(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df['일시'] = pd.to_datetime(df['일시'], errors='coerce')
    return df

# =========================
# 날짜, lag, rolling 생성
# =========================


def add_date_and_lag_features(df):
    df = df.copy().sort_values('일시').reset_index(drop=True)
    df['year'] = df['일시'].dt.year
    df['month'] = df['일시'].dt.month
    df['day'] = df['일시'].dt.day
    df['dayofweek'] = df['일시'].dt.dayofweek
    df['dayofyear'] = df['일시'].dt.dayofyear
    df['season'] = ((df['month'] % 12 + 3)//3)

    # 주기형 변환
    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)
    df['dayofweek_sin'] = np.sin(2*np.pi*df['dayofweek']/7)
    df['dayofweek_cos'] = np.cos(2*np.pi*df['dayofweek']/7)

    # lag, rolling
    for col in ['총량(KWh)', '평균기온(°C)', '합계 일사량(MJ/m2)', '평균 풍속(m/s)', '평균 상대습도(%)']:
        for lag in [1, 3, 7, 14]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
        for window in [3, 7, 14]:
            df[f'{col}_rolling{window}'] = df[col].rolling(
                window=window, min_periods=1).mean()

    df = df.dropna().reset_index(drop=True)
    return df

# =========================
# 지표 함수
# =========================


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100


def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true)+np.abs(y_pred))/2
    mask = denom != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(y_true[mask]-y_pred[mask])/denom[mask])*100

# =========================
# 학습 / 평가 / 최적 가중치
# =========================


def train_test_optimal_ensemble(df, n_iter_search=50):
    df = add_date_and_lag_features(df)

    # 결측/0 처리
    for col in ['합계 일사량(MJ/m2)', '평균기온(°C)', '평균 풍속(m/s)', '평균 상대습도(%)', '총량(KWh)']:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan).fillna(df[col].mean())

    df['전일_발전량'] = df['총량(KWh)'].shift(1).fillna(method='bfill')

    features = [c for c in df.columns if c not in [
        '일시', '총량(KWh)'] and df[c].dtype != 'object']

    # 데이터 분리
    train = df[df['year'].isin([2022, 2023])]
    val = df[df['year'] == 2024]
    test = df[df['year'] == 2025]

    X_train, y_train = train[features], train['총량(KWh)']
    X_val, y_val = val[features], val['총량(KWh)']
    X_test, y_test = test[features], test['총량(KWh)']

    # 스케일링
    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(
        y_train.values.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()

    # PredefinedSplit
    X_total = np.vstack([X_train_scaled, X_val_scaled])
    y_total = np.concatenate([y_train_scaled, y_val_scaled])
    ps = PredefinedSplit(
        test_fold=[-1]*len(X_train_scaled) + [0]*len(X_val_scaled))

    # =========================
    # XGBoost
    # =========================
    xgb_model = xgb.XGBRegressor(
        random_state=42, eval_metric='rmse', tree_method='hist')
    param_dist_xgb = {
        'n_estimators': [500, 700, 1000],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.3],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 0.5, 1],
        'reg_lambda': [1, 1.2]
    }

    rs_xgb = RandomizedSearchCV(xgb_model, param_distributions=param_dist_xgb, n_iter=n_iter_search,
                                cv=ps, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=1, random_state=None)
    rs_xgb.fit(X_total, y_total)
    best_xgb = rs_xgb.best_estimator_

    # =========================
    # LightGBM
    # =========================
    lgb_model = lgb.LGBMRegressor(random_state=42)
    param_dist_lgb = {
        'n_estimators': [500, 700, 1000],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.5, 1],
        'reg_lambda': [1, 1.2]
    }

    rs_lgb = RandomizedSearchCV(lgb_model, param_distributions=param_dist_lgb, n_iter=n_iter_search,
                                cv=ps, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=1, random_state=None)
    rs_lgb.fit(X_total, y_total)
    best_lgb = rs_lgb.best_estimator_

    print("✅ XGBoost 최적 파라미터:", rs_xgb.best_params_)
    print("✅ LightGBM 최적 파라미터:", rs_lgb.best_params_)

    # =========================
    # 검증셋 기반 최적 가중치 탐색
    # =========================
    y_pred_val_xgb = scaler_y.inverse_transform(
        best_xgb.predict(X_val_scaled).reshape(-1, 1)).ravel()
    y_pred_val_lgb = scaler_y.inverse_transform(
        best_lgb.predict(X_val_scaled).reshape(-1, 1)).ravel()

    best_rmse, best_w = float('inf'), 0
    for w in np.linspace(0, 1, 101):
        y_ens_val = w*y_pred_val_xgb + (1-w)*y_pred_val_lgb
        rmse = np.sqrt(mean_squared_error(y_val, y_ens_val))
        if rmse < best_rmse:
            best_rmse, best_w = rmse, w

    print(f"🌟 최적 앙상블 가중치: XGB={best_w:.2f}, LGB={1-best_w:.2f}")

    # =========================
    # 테스트 예측
    # =========================
    y_pred_xgb = scaler_y.inverse_transform(
        best_xgb.predict(X_test_scaled).reshape(-1, 1)).ravel()
    y_pred_lgb = scaler_y.inverse_transform(
        best_lgb.predict(X_test_scaled).reshape(-1, 1)).ravel()
    y_test_pred = best_w*y_pred_xgb + (1-best_w)*y_pred_lgb

    # =========================
    # 평가
    # =========================
    mask = y_test > 5
    mae = mean_absolute_error(y_test[mask], y_test_pred[mask])
    rmse = np.sqrt(mean_squared_error(y_val, y_ens_val))
    r2 = r2_score(y_test[mask], y_test_pred[mask])
    mape = mean_absolute_percentage_error(y_test[mask], y_test_pred[mask])
    smape_val = smape(y_test[mask], y_test_pred[mask])

    mean_y = np.mean(y_test[mask])
    mae_pct, rmse_pct = (mae/mean_y)*100, (rmse/mean_y)*100

    results = {
        'MAE(kWh)': round(mae, 4),
        'MAE(%)': round(mae_pct, 4),
        'RMSE(kWh)': round(rmse, 4),
        'RMSE(%)': round(rmse_pct, 4),
        'R2': round(r2, 6),
        'MAPE(%)': round(mape, 4),
        'SMAPE(%)': round(smape_val, 4),
        '최적가중치_XGB': round(best_w, 4),
        '최적가중치_LGB': round(1-best_w, 4),
        'XGB_파라미터': str(rs_xgb.best_params_),
        'LGB_파라미터': str(rs_lgb.best_params_)
    }

    print("📊 2025년 테스트 결과:", results)

    # =========================
    # 저장 및 시각화 (수정됨)
    # =========================
    save_path = os.path.join(base_path, f"{name}")
    os.makedirs(save_path, exist_ok=True)

    # 🔹 예측결과 시트 구성
    df_pred = test[['발전구분', '일시', '총량(KWh)']].copy()
    df_pred = df_pred.rename(columns={'총량(KWh)': '실제값'})
    df_pred['예측값'] = y_test_pred

    # 🔹 모델평가 시트 구성
    df_eval = pd.DataFrame([{
        'MAE(kWh)': results['MAE(kWh)'],
        'MAE(%)': results['MAE(%)'],
        'RMSE(kWh)': results['RMSE(kWh)'],
        'RMSE(%)': results['RMSE(%)'],
        'R2': results['R2'],
        'MAPE(%)': results['MAPE(%)'],
        'SMAPE(%)': results['SMAPE(%)'],
        'XGB파라미터': results['XGB_파라미터'],
        'LGB파라미터': results['LGB_파라미터'],
        '최적가중치_XGB': results['최적가중치_XGB'],
        '최적가중치_LGB': results['최적가중치_LGB']
    }])

    # 🔹 하나의 엑셀 파일에 시트 2개 저장
    output_path = os.path.join(save_path, "예측결과_및_모델평가.xlsx")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_pred.to_excel(writer, sheet_name='예측결과', index=False)
        df_eval.to_excel(writer, sheet_name='모델평가', index=False)

    print(f"📁 엑셀 파일 저장 완료: {output_path}")

    # 그래프 생성
    plt.figure(figsize=(14, 6))
    plt.plot(df_pred['일시'], df_pred['실제값'],
             label='실제값', marker='o', linewidth=1)
    plt.plot(df_pred['일시'], df_pred['예측값'],
             label='예측값', marker='x', linewidth=1)
    plt.title(f"'{name}'' - 최적가중치 앙상블 결과")
    plt.xlabel("일시")
    plt.ylabel("발전량(kWh)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "예측결과 그래프.png"))
    plt.close()

    # =========================
    # ✅ 전체기간(22~25년) 예측 추가
    # =========================
    X_all_scaled = scaler_X.transform(df[features])
    y_pred_xgb_all = scaler_y.inverse_transform(
        best_xgb.predict(X_all_scaled).reshape(-1, 1)).ravel()
    y_pred_lgb_all = scaler_y.inverse_transform(
        best_lgb.predict(X_all_scaled).reshape(-1, 1)).ravel()
    y_pred_all = best_w*y_pred_xgb_all + (1-best_w)*y_pred_lgb_all

    df_all = df.copy()
    df_all['예측값'] = y_pred_all
    df_all['실제값'] = df['총량(KWh)']

    # =========================
    # ✅ 22~25년 전체 그래프 추가
    # =========================
    plt.figure(figsize=(14, 6))
    plt.plot(df_all['일시'], df_all['실제값'], label='실제 발전량(kWh)',
             color='tab:blue', linewidth=1.5)
    plt.plot(df_all['일시'], df_all['예측값'], label='예측 발전량(kWh)',
             color='tab:orange', linewidth=1.5)
    plt.title("2022.01.01 ~ 2025.08.31 실제 vs 예측 발전량 비교", fontsize=13)
    plt.xlabel("일시")
    plt.ylabel("발전량(kWh)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(base_path, f"{name}")
    plt.savefig(os.path.join(save_path, "실제_vs_예측_전체기간_그래프.png"))
    plt.close()

    print("📊 전체 기간(2022~2025) 그래프 저장 완료")

    # =========================
    # 결과 반환
    # =========================
    return best_xgb, best_lgb, y_test_pred


# =========================
# 실행
# =========================
df_data = load_data()
best_xgb, best_lgb, test_results = train_test_optimal_ensemble(df_data)
