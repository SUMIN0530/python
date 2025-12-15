import os
import pandas as pd
import numpy as np
import xgboost as xgb
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

# =========================
# 파일 경로
# =========================
base_path = r"C:\Users\bhy10\Documents\PYTHONKDT\Update_set"

# =========================
# 데이터 로딩
# =========================


def load_data():
    files = [
        os.path.join(base_path, "두산엔진MG태양광_1", f"두산엔진MG태양광_1_{year}_data.xlsx") for year in range(22, 26)
    ]
    dfs = [pd.read_excel(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df['일시'] = pd.to_datetime(df['일시'], errors='coerce')
    return df

# =========================
# 날짜 및 lag 특성 생성
# =========================


def add_date_and_lag_features(df):
    df = df.copy().sort_values('일시').reset_index(drop=True)
    df['year'] = df['일시'].dt.year
    df['month'] = df['일시'].dt.month
    df['day'] = df['일시'].dt.day
    df['dayofweek'] = df['일시'].dt.dayofweek
    df['dayofyear'] = df['일시'].dt.dayofyear

    # 주기형 변환
    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)
    df['dayofweek_sin'] = np.sin(2*np.pi*df['dayofweek']/7)
    df['dayofweek_cos'] = np.cos(2*np.pi*df['dayofweek']/7)

    # lag, rolling 특성
    for col in ['총량(KWh)', '평균기온(°C)', '합계 일사량(MJ/m2)', '평균 풍속(m/s)', '평균 상대습도(%)']:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag3'] = df[col].shift(3)
        df[f'{col}_rolling3'] = df[col].rolling(window=3, min_periods=1).mean()

    df = df.dropna().reset_index(drop=True)
    return df

# =========================
# 지표 계산 함수
# =========================


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_mask = y_true != 0
    if nonzero_mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100


def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100

# =========================
# 학습 / 검증 / 테스트 + Randomized Search
# =========================


def train_validate_test_random(df, n_iter_search=100):
    df = add_date_and_lag_features(df)

    # 0값 처리
    for col in ['합계 일사량(MJ/m2)', '평균기온(°C)', '평균 풍속(m/s)', '평균 상대습도(%)', '총량(KWh)']:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan).fillna(df[col].mean())

    df['전일_발전량'] = df['총량(KWh)'].shift(1).fillna(method='bfill')

    features = [
        '합계 일사량(MJ/m2)', '평균기온(°C)', '평균 풍속(m/s)', '평균 상대습도(%)',
        'year', 'month', 'day', 'dayofweek', 'dayofyear',
        'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos',
        '총량(KWh)_lag1', '총량(KWh)_lag3', '총량(KWh)_rolling3', '전일_발전량'
    ]

    # 데이터 분리
    train = df[df['year'].isin([2022, 2023])].reset_index(drop=True)
    val = df[df['year'] == 2024].reset_index(drop=True)
    test = df[df['year'] == 2025].reset_index(drop=True)

    X_train = train[features]
    y_train = train['총량(KWh)']
    X_val = val[features]
    y_val = val['총량(KWh)']
    X_test = test[features]
    y_test = test['총량(KWh)']

    # 스케일링
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(
        y_train.values.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()

    # PredefinedSplit 정의 (train/val 단일 split)
    X_total = np.vstack([X_train_scaled, X_val_scaled])
    y_total = np.concatenate([y_train_scaled, y_val_scaled])
    test_fold = [-1]*len(X_train_scaled) + [0]*len(X_val_scaled)
    ps = PredefinedSplit(test_fold=test_fold)

    # =========================
    # Randomized Search 정의
    # =========================
    param_dist = {
        'n_estimators': [500, 700, 1000],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.3, 0.5],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 0.5, 1],
        'reg_lambda': [1, 1.2]
    }

    xgb_model = xgb.XGBRegressor(
        random_state=42, eval_metric='rmse', tree_method='hist')

    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=n_iter_search,
        scoring='neg_root_mean_squared_error',
        cv=ps,
        verbose=1,
        n_jobs=-1,
        random_state=None
    )

    # 학습
    random_search.fit(X_total, y_total)

    best_model = random_search.best_estimator_
    print("최적 하이퍼파라미터:", random_search.best_params_)

    # 테스트 예측
    y_test_pred = scaler_y.inverse_transform(
        best_model.predict(X_test_scaled).reshape(-1, 1)).ravel()

    # 테스트 지표
    mask_test = y_test > 5
    results = {
        'MAE': mean_absolute_error(y_test[mask_test], y_test_pred[mask_test]),
        'RMSE': np.sqrt(mean_squared_error(y_test[mask_test], y_test_pred[mask_test])),
        'R2': r2_score(y_test[mask_test], y_test_pred[mask_test]),
        'MAPE': mean_absolute_percentage_error(y_test[mask_test], y_test_pred[mask_test]),
        'SMAPE': smape(y_test[mask_test], y_test_pred[mask_test])
    }
    print("2025년 테스트 지표:", results)

    # 예측값 저장 및 시각화
    save_path = os.path.join(base_path, "두산엔진MG태양광_1")
    os.makedirs(save_path, exist_ok=True)

    df_pred = test[['일시', '총량(KWh)']].copy()
    df_pred['예측값'] = y_test_pred
    df_pred.to_excel(os.path.join(
        save_path, "pred_test_random.xlsx"), index=False)

    plt.figure(figsize=(14, 6))
    plt.plot(df_pred['일시'], df_pred['총량(KWh)'],
             label='실제값', marker='o', linewidth=1)
    plt.plot(df_pred['일시'], df_pred['예측값'],
             label='예측값', marker='x', linewidth=1)
    plt.title("두산 1호기 테스트 예측 결과 (Randomized Search)")
    plt.xlabel("일시")
    plt.ylabel("발전량 (kWh)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "plot_test_random.png"))
    plt.close()

    return best_model, results


# =========================
# 실행
# =========================
df_data = load_data()
best_model, test_results = train_validate_test_random(
    df_data, n_iter_search=50)
