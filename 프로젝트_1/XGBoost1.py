import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib import rc
'''xg 최적화롤 돌리는 코드'''
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

# =============================
# 결측치/0 처리
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
# XGBoost + GridSearchCV
# =============================
xgb_model = xgb.XGBRegressor(random_state=42, eval_metric='rmse')

param_grid = {
    'max_depth': [4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [200, 500, 800],
    'gamma': [0, 0.1, 0.3],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='r2',
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train_scaled)

best_model = grid_search.best_estimator_
print("최적 파라미터:", grid_search.best_params_)

# =============================
# 테스트 데이터 예측
# =============================
y_pred_scaled = best_model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_test_orig = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))

# =============================
# 평가
# =============================
mae = mean_absolute_error(y_test_orig, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
r2 = r2_score(y_test_orig, y_pred)

print(f"테스트 MAE: {mae:.2f}")
print(f"테스트 RMSE: {rmse:.2f}")
print(f"테스트 R²: {r2:.4f}")

# =============================
# 예측 결과 엑셀 저장
# =============================
result_df = test_df.copy()
result_df['Predicted_Total'] = y_pred
save_path = r"C:/Users/alsl0/Documents/python/Predicted_Test_Data1.xlsx"
result_df.to_excel(save_path, index=False)
print(f":흰색_확인_표시: 예측 결과를 엑셀로 저장했습니다: {save_path}")

# =============================
# 실제 vs 예측 꺾은선 그래프
# =============================
plt.figure(figsize=(15, 6))
plt.plot(y_test_orig, label='실제 발전량', marker='o')
plt.plot(y_pred, label='예측 발전량', marker='x')
plt.xlabel('시간 인덱스')
plt.ylabel('발전량(KWh)')
plt.title('실제 vs 예측 발전량')
plt.legend()
plt.tight_layout()
plt.show()
