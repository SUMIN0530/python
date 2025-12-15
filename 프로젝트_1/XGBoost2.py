import random
from sklearn.model_selection import ParameterSampler
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib import rc
import time
'''
grid 후보 / grid 모델링 + random 모델링
'''
# =============================
# 한글 폰트 설정
# =============================
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# =============================
# 데이터 불러오기
# =============================
train_path = r"C:/Users/alsl0/Documents/python/23년_train_data.xlsx"
test_path = r"C:/Users/alsl0/Documents/python/24년_test_data.xlsx"

train_df = pd.read_excel(train_path)
test_df = pd.read_excel(test_path)

X_train = train_df[['합계 일사량(MJ/m2)', '평균기온(°C)']]
y_train = train_df['총량(KWh)']
X_test = test_df[['합계 일사량(MJ/m2)', '평균기온(°C)']]
y_test = test_df['총량(KWh)']
dates_test = pd.to_datetime(test_df['일시'])

# =============================
# 결측치 / 0 처리
# =============================
X_train = X_train.replace(0, np.nan).interpolate(method='linear')
X_test = X_test.replace(0, np.nan).interpolate(method='linear')
y_train = y_train.replace(0, np.nan).interpolate(method='linear')
y_test = y_test.replace(0, np.nan).interpolate(method='linear')

# =============================
# 스케일링
# =============================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

# =============================
# 하이퍼파라미터 후보
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

xgb_model = xgb.XGBRegressor(random_state=42, eval_metric='rmse')

# =============================
# GridSearchCV
# =============================
start_time = time.time()
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='r2',
    cv=3,
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train_scaled, y_train_scaled)
grid_time = time.time() - start_time

best_grid_model = grid_search.best_estimator_
y_pred_scaled_grid = best_grid_model.predict(X_test_scaled)
y_pred_grid = scaler_y.inverse_transform(y_pred_scaled_grid.reshape(-1, 1))
y_orig = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))

mae_grid = mean_absolute_error(y_orig, y_pred_grid)
rmse_grid = np.sqrt(mean_squared_error(y_orig, y_pred_grid))
r2_grid = r2_score(y_orig, y_pred_grid)

print("===== GridSearch 결과 =====")
print(f"학습 시간: {grid_time:.2f}초")
print(f"MAE: {mae_grid:.2f}, RMSE: {rmse_grid:.2f}, R²: {r2_grid:.4f}")
print("최적 파라미터:", grid_search.best_params_)

# =============================
# RandomizedSearchCV
# =============================

start_time = time.time()
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=50,   # 랜덤 50번 샘플링
    scoring='r2',
    cv=3,
    n_jobs=-1,
    verbose=0,
    random_state=42
)
random_search.fit(X_train_scaled, y_train_scaled)
random_time = time.time() - start_time

best_random_model = random_search.best_estimator_
y_pred_scaled_rand = best_random_model.predict(X_test_scaled)
y_pred_rand = scaler_y.inverse_transform(y_pred_scaled_rand.reshape(-1, 1))

mae_rand = mean_absolute_error(y_orig, y_pred_rand)
rmse_rand = np.sqrt(mean_squared_error(y_orig, y_pred_rand))
r2_rand = r2_score(y_orig, y_pred_rand)

print("\n===== RandomizedSearch 결과 =====")
print(f"학습 시간: {random_time:.2f}초")
print(f"MAE: {mae_rand:.2f}, RMSE: {rmse_rand:.2f}, R²: {r2_rand:.4f}")
print("최적 파라미터:", random_search.best_params_)

# =============================
# 실제 vs 예측 꺾은선 그래프
# =============================
plt.figure(figsize=(15, 6))
plt.plot(dates_test, y_orig, label='실제 발전량', marker='o')
plt.plot(dates_test, y_pred_grid, label='GridSearch 예측', marker='x')
plt.plot(dates_test, y_pred_rand, label='RandomizedSearch 예측', marker='^')
plt.xlabel('일자 인덱스')
plt.ylabel('발전량(KWh)')
plt.title('실제 vs 예측 발전량 비교 (GridSearch vs RandomizedSearch)')
plt.legend()
plt.tight_layout()
plt.show()
