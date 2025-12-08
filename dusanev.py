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

'''
혜영이가 보내준 코드
'''

# 한글 폰트 설정
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

base_path = r"C:/Users/alsl0/Documents/python/Update_set"

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
    df['season'] = ((df['month'] % 12 + 3)//3)  # 계절

    # 주기형 변환
    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)
    df['dayofweek_sin'] = np.sin(2*np.pi*df['dayofweek']/7)
    df['dayofweek_cos'] = np.cos(2*np.pi*df['dayofweek']/7)

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
# 지표
# =========================


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100


def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true)+np.abs(y_pred))/2
    mask = denom != 0
    return np.mean(np.abs(y_true[mask]-y_pred[mask])/denom[mask])*100

# =========================
# 학습 / 검증 / 테스트 + 2단계 랜덤서치 + 앙상블
# =========================


def train_test_ensemble(df, n_iter_stage1=50, n_iter_stage2=30):
    df = add_date_and_lag_features(df)

    # 0값 처리
    for col in ['합계 일사량(MJ/m2)', '평균기온(°C)', '평균 풍속(m/s)', '평균 상대습도(%)', '총량(KWh)']:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan).fillna(df[col].mean())

    df['전일_발전량'] = df['총량(KWh)'].shift(1).fillna(method='bfill')

    # 숫자형 feature만 사용 (문자형 제거)
    features = [c for c in df.columns if c not in [
        '일시', '총량(KWh)'] and df[c].dtype != 'object']

    # 학습 / 검증 / 테스트 분리
    train = df[df['year'].isin([2022, 2023])].reset_index(drop=True)
    val = df[df['year'] == 2024].reset_index(drop=True)
    test = df[df['year'] == 2025].reset_index(drop=True)

    X_train, y_train = train[features], train['총량(KWh)']
    X_val, y_val = val[features], val['총량(KWh)']
    X_test, y_test = test[features], test['총량(KWh)']

    # 스케일링
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(
        y_train.values.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()

    X_total = np.vstack([X_train_scaled, X_val_scaled])
    y_total = np.concatenate([y_train_scaled, y_val_scaled])
    test_fold = [-1]*len(X_train_scaled)+[0]*len(X_val_scaled)
    ps = PredefinedSplit(test_fold=test_fold)

    # ===== Stage1: XGBoost =====
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
    rs_xgb = RandomizedSearchCV(xgb_model, param_distributions=param_dist_xgb, n_iter=n_iter_stage1,
                                cv=ps, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=1, random_state=None)
    rs_xgb.fit(X_total, y_total)
    best_xgb = rs_xgb.best_estimator_

    # ===== Stage2: LightGBM =====
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
    rs_lgb = RandomizedSearchCV(lgb_model, param_distributions=param_dist_lgb, n_iter=n_iter_stage2,
                                cv=ps, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=1, random_state=None)
    rs_lgb.fit(X_total, y_total)
    best_lgb = rs_lgb.best_estimator_
    print("XGBoost 최적:", rs_xgb.best_params_)
    print("LightGBM 최적:", rs_lgb.best_params_)

    # ===== 앙상블 예측 =====
    y_pred_xgb = scaler_y.inverse_transform(
        best_xgb.predict(X_test_scaled).reshape(-1, 1)).ravel()
    y_pred_lgb = scaler_y.inverse_transform(
        best_lgb.predict(X_test_scaled).reshape(-1, 1)).ravel()
    y_test_pred = (y_pred_xgb + y_pred_lgb)/2

    mask_test = y_test > 5
    results = {
        'MAE': mean_absolute_error(y_test[mask_test], y_test_pred[mask_test]),
        'RMSE': np.sqrt(mean_squared_error(y_test[mask_test], y_test_pred[mask_test])),
        'R2': r2_score(y_test[mask_test], y_test_pred[mask_test]),
        'MAPE': mean_absolute_percentage_error(y_test[mask_test], y_test_pred[mask_test]),
        'SMAPE': smape(y_test[mask_test], y_test_pred[mask_test])
    }
    print("2025년 테스트 지표:", results)

    # 저장 및 시각화
    save_path = os.path.join(base_path, "두산엔진MG태양광_1")
    os.makedirs(save_path, exist_ok=True)
    df_pred = test[['일시', '총량(KWh)']].copy()
    df_pred['예측값'] = y_test_pred
    df_pred.to_excel(os.path.join(
        save_path, "pred_test_ensemble.xlsx"), index=False)

    # plt.figure(figsize=(14,6))
    # plt.plot(df_pred['일시'], df_pred['총량(KWh)'], label='실제값', marker='o',linewidth=1)
    # plt.plot(df_pred['일시'], df_pred['예측값'], label='예측값', marker='x',linewidth=1)
    # plt.title("두산엔진MG태양광_1호기 테스트 예측 결과 (XGB+LGB 앙상블)")
    # plt.xlabel("일시"); plt.ylabel("발전량 (kWh)")
    # plt.grid(True); plt.legend(); plt.tight_layout()
    # plt.savefig(os.path.join(save_path,"plot_test_ensemble.png"))
    # plt.close()

    def plot_feature_importance(model, feature_names, title="Feature Importance"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # plt.figure(figsize=(12,6))
        # plt.title(title)
        # plt.bar(range(len(importances)), importances[indices], align='center')
        # plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        # plt.ylabel("Importance")
        # plt.tight_layout()
        # plt.show()

    # XGB 특성 중요도
    plot_feature_importance(
        best_xgb, features, title="XGBoost Feature Importance")

    # LGB 특성 중요도
    plot_feature_importance(
        best_lgb, features, title="LightGBM Feature Importance")

    return best_xgb, best_lgb, results


# =========================
# 실행
# =========================
df_data = load_data()
best_xgb, best_lgb, test_results = train_test_ensemble(df_data)
