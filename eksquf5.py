import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib import rc
import time
import os  # 폴더 생성 및 경로 관리용

'''grid 후보 random 모델링 파일별 저장.'''
# =============================
# 한글 폰트 설정
# =============================
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# =============================
# train 데이터 여러 개 불러와 합치기
# =============================
train_files = [
    r"C:/Users/alsl0/Documents/python/지역별_발전량_비교/영흥태양광_1호/23_train_data.xlsx",
    r"C:/Users/alsl0/Documents/python/지역별_발전량_비교/영흥태양광_1호/24_test_data.xlsx"
]
test_path = r"C:/Users/alsl0/Documents/python/지역별_발전량_비교/영흥태양광_1호/25_test_data.xlsx"

train_df = pd.concat([pd.read_excel(f, engine="openpyxl")
                     for f in train_files], ignore_index=True)
test_df = pd.read_excel(test_path, engine="openpyxl")

X_train = train_df[['합계 일사량(MJ/m2)', '평균기온(°C)']]
y_train = train_df['총량(KWh)']
X_test = test_df[['합계 일사량(MJ/m2)', '평균기온(°C)']]
y_test = test_df['총량(KWh)']
dates_test = pd.to_datetime(test_df['일시'], errors='coerce')

# =============================
# 숫자형 변환 (interpolate 전에 필수)
# =============================
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')
y_train = pd.to_numeric(y_train, errors='coerce')
y_test = pd.to_numeric(y_test, errors='coerce')

# =============================
# 결측치 / 0 처리
# =============================
X_train = X_train.replace(0, np.nan).interpolate(
    method='linear').ffill().bfill()
X_test = X_test.replace(0, np.nan).interpolate(method='linear').ffill().bfill()
y_train = y_train.replace(0, np.nan).interpolate(
    method='linear').ffill().bfill()
y_test_interpolated = y_test.replace(0, np.nan).interpolate(
    method='linear').ffill().bfill()  # 보간값 별도 저장

# =============================
# 스케일링
# =============================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(
    y_test_interpolated.values.reshape(-1, 1)).ravel()  # 보간값으로 변환

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
# RandomizedSearchCV
# =============================
start_time = time.time()
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=50,
    scoring='r2',
    cv=3,
    n_jobs=-1,
    verbose=1,
    random_state=None  # 시드 랜덤으로 설정
)
random_search.fit(X_train_scaled, y_train_scaled)
random_time = time.time() - start_time

best_random_model = random_search.best_estimator_
y_pred_scaled_rand = best_random_model.predict(X_test_scaled)
y_pred_rand = scaler_y.inverse_transform(y_pred_scaled_rand.reshape(-1, 1))
y_orig = y_test.values.reshape(-1, 1)
y_interp = y_test_interpolated.values.reshape(-1, 1)

mae_rand = mean_absolute_error(y_interp, y_pred_rand)
rmse_rand = np.sqrt(mean_squared_error(y_interp, y_pred_rand))
r2_rand = r2_score(y_interp, y_pred_rand)

print("\n===== RandomizedSearch 결과 =====")
print(f"MAE: {mae_rand:.2f}, RMSE: {rmse_rand:.2f}, R²: {r2_rand:.4f}")
print("최적 파라미터:", random_search.best_params_)

# =============================
# 저장 폴더 경로 설정
# =============================
save_folder = r"C:/Users/alsl0/Documents/python/지역별_발전량_비교/영흥태양광_1호"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# =============================
# RandomizedSearch 결과 엑셀 저장
# =============================
results_df = test_df.copy()
results_df['보간발전량(KWh)'] = y_interp.ravel()
results_df['예측발전량(KWh)'] = y_pred_rand.ravel()

excel_save_path = os.path.join(save_folder, 'RandomizedSearch_결과.xlsx')
results_df.to_excel(excel_save_path, index=False)
print(f"✅ RandomizedSearch 결과 엑셀 저장 완료: {excel_save_path}")

# =============================
# 실제 vs 보간 vs 예측 그래프 저장
# =============================
plt.figure(figsize=(15, 6))
plt.plot(dates_test, y_orig, label='원래값', marker='o')
plt.plot(dates_test, y_interp, label='보간값', marker='^')
plt.plot(dates_test, y_pred_rand, label='예측값', marker='s')
plt.xlabel('일자')
plt.ylabel('발전량(KWh)')
plt.title('원래값 vs 보간값 vs 예측값')
plt.legend()
plt.tight_layout()

graph_save_path = os.path.join(save_folder, 'RandomizedSearch_3종그래프.png')
plt.savefig(graph_save_path, dpi=300)
plt.show()
print(f"✅ 3종 그래프 이미지 저장 완료: {graph_save_path}")
