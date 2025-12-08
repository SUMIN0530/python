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

'''1대1 앙상블 어쩌고 + 누수제거 코드'''
# -----------------------------
# 한글 폰트
# -----------------------------
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# =========================
# 파일 경로
# =========================
base_path = r"C:\Users\alsl0\Documents\python\Update_set"
name = '경상대태양광_1'  # 지역명

# =========================
# 데이터 로딩
# =========================


def load_data():
    files = [os.path.join(
        base_path, f"{name}", f"{name}_{year}_data.xlsx") for year in range(22, 26)]
    dfs = [pd.read_excel(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df['일시'] = pd.to_datetime(df['일시'], errors='coerce')
    return df

# =========================
# 날짜, lag, rolling 생성 (누수 방지)
# =========================


def add_date_and_lag_features_no_leak(df):
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

    # lag 및 rolling (전일까지, 누수 방지)
    for col in ['총량(KWh)', '평균기온(°C)', '합계 일사량(MJ/m2)', '평균 풍속(m/s)', '평균 상대습도(%)']:
        # lag
        for lag in [1, 3, 7, 14]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
        # rolling
        for window in [3, 7, 14]:
            df[f'{col}_rolling{window}'] = df[col].shift(
                1).rolling(window=window, min_periods=1).mean()

    # 연도 경계에서 lag/rolling 누수 제거
    for c in df.columns:
        if 'lag' in c.lower() or 'rolling' in c.lower():
            df.loc[df['year'] != df['year'].shift(1), c] = np.nan

    # 전일 발전량
    df['전일_발전량'] = df['총량(KWh)'].shift(1)
    df.loc[df['year'] != df['year'].shift(1), '전일_발전량'] = np.nan

    return df.dropna().reset_index(drop=True)

# =========================
# 평가 지표
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
# 학습 / 테스트 / 누수 방지 앙상블
# =========================


def train_test_no_leak_ensemble(df, n_iter_search=50):
    df = add_date_and_lag_features_no_leak(df)

    # 결측/0 처리
    for col in ['합계 일사량(MJ/m2)', '평균기온(°C)', '평균 풍속(m/s)', '평균 상대습도(%)', '총량(KWh)']:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan).fillna(df[col].mean())

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
    # XGBoost 학습
    # =========================
    xgb_model = xgb.XGBRegressor(
        random_state=None, eval_metric='rmse', tree_method='hist')
    param_dist_xgb = {
        'n_estimators': [500, 700, 1000], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9], 'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.3], 'min_child_weight': [1, 3, 5], 'reg_alpha': [0, 0.5, 1], 'reg_lambda': [1, 1.2]
    }
    rs_xgb = RandomizedSearchCV(xgb_model, param_distributions=param_dist_xgb, n_iter=n_iter_search,
                                cv=ps, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=1, random_state=None)
    rs_xgb.fit(X_total, y_total)
    best_xgb = rs_xgb.best_estimator_

    # =========================
    # LightGBM 학습
    # =========================
    lgb_model = lgb.LGBMRegressor()
    param_dist_lgb = {
        'n_estimators': [500, 700, 1000], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9], 'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.5, 1], 'reg_lambda': [1, 1.2]
    }
    rs_lgb = RandomizedSearchCV(lgb_model, param_distributions=param_dist_lgb, n_iter=n_iter_search,
                                cv=ps, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=1, random_state=None)
    rs_lgb.fit(X_total, y_total)
    best_lgb = rs_lgb.best_estimator_

    # =========================
    # 최적 앙상블 가중치
    # =========================
    y_val_pred_xgb = scaler_y.inverse_transform(
        best_xgb.predict(X_val_scaled).reshape(-1, 1)).ravel()
    y_val_pred_lgb = scaler_y.inverse_transform(
        best_lgb.predict(X_val_scaled).reshape(-1, 1)).ravel()

    best_rmse, best_w = float('inf'), 0
    for w in np.linspace(0, 1, 101):
        y_ens_val = w*y_val_pred_xgb + (1-w)*y_val_pred_lgb
        rmse = np.sqrt(mean_squared_error(y_val, y_ens_val))
        if rmse < best_rmse:
            best_rmse, best_w = rmse, w

    # =========================
    # 테스트 예측
    # =========================
    y_test_pred_xgb = scaler_y.inverse_transform(
        best_xgb.predict(X_test_scaled).reshape(-1, 1)).ravel()
    y_test_pred_lgb = scaler_y.inverse_transform(
        best_lgb.predict(X_test_scaled).reshape(-1, 1)).ravel()
    y_test_pred = best_w*y_test_pred_xgb + (1-best_w)*y_test_pred_lgb

    # =========================
    # 평가 지표
    # =========================
    mask = y_test > 5
    mae = mean_absolute_error(y_test[mask], y_test_pred[mask])
    rmse = np.sqrt(mean_squared_error(y_test[mask], y_test_pred[mask]))
    r2 = r2_score(y_test[mask], y_test_pred[mask])
    mape = mean_absolute_percentage_error(y_test[mask], y_test_pred[mask])
    smape_val = smape(y_test[mask], y_test_pred[mask])
    mean_y = np.mean(y_test[mask])
    mae_pct, rmse_pct = (mae/mean_y)*100, (rmse/mean_y)*100

    results = {
        'MAE(kWh)': round(mae, 4), 'MAE(%)': round(mae_pct, 4),
        'RMSE(kWh)': round(rmse, 4), 'RMSE(%)': round(rmse_pct, 4),
        'R2': round(r2, 6), 'MAPE(%)': round(mape, 4), 'SMAPE(%)': round(smape_val, 4),
        '최적가중치_XGB': round(best_w, 4), '최적가중치_LGB': round(1-best_w, 4),
        'XGB_파라미터': str(rs_xgb.best_params_), 'LGB_파라미터': str(rs_lgb.best_params_)
    }

    # =========================
    # 지역별 폴더 생성 및 저장
    # =========================
    save_path = os.path.join(base_path, "누수방지", name)
    os.makedirs(save_path, exist_ok=True)

    # 엑셀 저장
    output_path = os.path.join(save_path, f"{name}_누수제거.xlsx")
    df_pred = test[['발전구분', '일시', '총량(KWh)']].copy()
    df_pred = df_pred.rename(columns={'총량(KWh)': '실제값'})
    df_pred['예측값'] = y_test_pred
    df_eval = pd.DataFrame([results])
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_pred.to_excel(writer, sheet_name='예측결과', index=False)
        df_eval.to_excel(writer, sheet_name='모델평가', index=False)

    # 그래프 저장: 실제 vs 예측
    plt.figure(figsize=(16, 6))
    plt.plot(test['일시'], test['총량(KWh)'], label='실제값', linewidth=1.5)
    plt.plot(test['일시'], y_test_pred, label='예측값', linewidth=1.5)
    plt.title('실제값 vs 예측값', fontsize=16)
    plt.xlabel('일시')
    plt.ylabel('발전량(KWh)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{name}_실제vs예측.png"))

    # 그래프 저장: 오차율
    error_pct = 100*(test['총량(KWh)'] - y_test_pred)/test['총량(KWh)']
    plt.figure(figsize=(16, 4))
    plt.plot(test['일시'], error_pct, color='red', linewidth=1)
    plt.title('예측 오차율 (%)', fontsize=16)
    plt.xlabel('일시')
    plt.ylabel('오차율 (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{name}_오차율.png"))

    print(f"✅ 누수 제거 예측결과 및 모델평가, 그래프 저장 완료: {save_path}")

    # =========================
    # 결과 출력
    # =========================
    for k, v in results.items():
        print(f"{k}: {v}")

    return best_xgb, best_lgb, y_test_pred, results


# =========================
# 실행
# =========================
df_data = load_data()
best_xgb, best_lgb, test_results, results = train_test_no_leak_ensemble(
    df_data)
