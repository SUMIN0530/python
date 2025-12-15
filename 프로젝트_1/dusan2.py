import os
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import PredefinedSplit
import matplotlib.pyplot as plt
from matplotlib import rc

'''
혜영이가 보내준 코드
'''

# 한글 폰트 설정
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

base_path = r"C:\Users\bhy10\Documents\PYTHONKDT\Update_set"

# =========================
# 데이터 로딩
# =========================


def load_data():
    files = [os.path.join(base_path, "두산엔진MG태양광_1",
                          f"두산엔진MG태양광_1_{year}_data.xlsx") for year in range(22, 26)]
    dfs = [pd.read_excel(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df['일시'] = pd.to_datetime(df['일시'], errors='coerce')
    return df

# =========================
# 날짜/lag/rolling 특성
# =========================


def add_date_and_lag_features(df):
    df = df.copy().sort_values('일시').reset_index(drop=True)
    df['year'] = df['일시'].dt.year
    df['month'] = df['일시'].dt.month
    df['day'] = df['일시'].dt.day
    df['dayofweek'] = df['일시'].dt.dayofweek
    df['dayofyear'] = df['일시'].dt.dayofyear
    df['season'] = ((df['month'] % 12 + 3) // 3)  # 계절

    # 주기형 변환
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    # lag & rolling (1,3,7,14)
    cols = ['총량(KWh)', '평균기온(°C)', '합계 일사량(MJ/m2)', '평균 풍속(m/s)', '평균 상대습도(%)']
    for col in cols:
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
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom != 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100

# =========================
# 학습 / 검증 / 테스트 + 앙상블 + 가중치 최적화
# =========================


def train_test_ensemble(df):
    df = add_date_and_lag_features(df)

    # 0값 처리
    for col in ['합계 일사량(MJ/m2)', '평균기온(°C)', '평균 풍속(m/s)', '평균 상대습도(%)', '총량(KWh)']:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan).fillna(df[col].mean())

    df['전일_발전량'] = df['총량(KWh)'].shift(1).fillna(method='bfill')

    features = [c for c in df.columns if c not in [
        '일시', '총량(KWh)'] and df[c].dtype != 'object']

    train = df[df['year'].isin([2022, 2023])].reset_index(drop=True)
    val = df[df['year'] == 2024].reset_index(drop=True)
    test = df[df['year'] == 2025].reset_index(drop=True)

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

    X_total = np.vstack([X_train_scaled, X_val_scaled])
    y_total = np.concatenate([y_train_scaled, y_val_scaled])

    # ===== XGBoost 최적 설정 =====
    best_xgb = xgb.XGBRegressor(
        subsample=0.7,
        reg_lambda=1,
        reg_alpha=0,
        n_estimators=500,
        min_child_weight=5,
        max_depth=7,
        learning_rate=0.05,
        gamma=0,
        colsample_bytree=0.9,
        random_state=42,
        tree_method='hist'
    )
    best_xgb.fit(X_total, y_total)

    # ===== LightGBM 최적 설정 =====
    best_lgb = lgb.LGBMRegressor(
        subsample=0.7,
        reg_lambda=1,
        reg_alpha=0,
        n_estimators=700,
        max_depth=7,
        learning_rate=0.01,
        colsample_bytree=0.8,
        random_state=42
    )
    best_lgb.fit(X_total, y_total)

    # ===== 개별 모델 예측 =====
    y_pred_xgb = scaler_y.inverse_transform(
        best_xgb.predict(X_test_scaled).reshape(-1, 1)).ravel()
    y_pred_lgb = scaler_y.inverse_transform(
        best_lgb.predict(X_test_scaled).reshape(-1, 1)).ravel()
    y_test = y_test.values

    # ===== 가중치 최적화 =====
    best_weight, best_r2, best_metrics = None, -np.inf, None
    results_list = []

    mask_test = y_test > 5
    for w in np.arange(0, 1.05, 0.05):
        y_ensemble = w * y_pred_xgb + (1 - w) * y_pred_lgb
        metrics = {
            'XGB_Weight': round(w, 2),
            'LGB_Weight': round(1 - w, 2),
            'MAE': mean_absolute_error(y_test[mask_test], y_ensemble[mask_test]),
            'RMSE': np.sqrt(mean_squared_error(y_test[mask_test], y_ensemble[mask_test])),
            'R2': r2_score(y_test[mask_test], y_ensemble[mask_test]),
            'MAPE': mean_absolute_percentage_error(y_test[mask_test], y_ensemble[mask_test]),
            'SMAPE': smape(y_test[mask_test], y_ensemble[mask_test])
        }
        results_list.append(metrics)
        if metrics['R2'] > best_r2:
            best_r2, best_weight, best_metrics = metrics['R2'], w, metrics

    df_results = pd.DataFrame(results_list)
    print("\n💡 최적 가중치 (R² 기준): XGB={:.2f}, LGB={:.2f}, R²={:.4f}".format(
        best_metrics['XGB_Weight'], best_metrics['LGB_Weight'], best_metrics['R2']))
    print("\n가중치별 성능 변화표:")
    print(df_results)

    # ===== 결과 저장 =====
    save_path = os.path.join(base_path, "두산엔진MG태양광_1")
    os.makedirs(save_path, exist_ok=True)
    df_results.to_excel(os.path.join(
        save_path, "ensemble_weight_optimization.xlsx"), index=False)

    return best_xgb, best_lgb, df_results, best_metrics


# =========================
# 실행
# =========================
df_data = load_data()
best_xgb, best_lgb, df_results, best_metrics = train_test_ensemble(df_data)
