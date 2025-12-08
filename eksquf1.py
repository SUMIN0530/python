import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import pandas as pd
import numpy as np
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
'''혜영이가 보낸준 코드 xg 최적으로 값 수정'''
# =============================
# :일: 데이터 불러오기
# =============================
train_path = r"C:/Users/alsl0/Documents/python/23년_train_data.xlsx"
test_path = r"C:/Users/alsl0/Documents/python/24년_test_data.xlsx"
train_df = pd.read_excel(train_path)
test_df = pd.read_excel(test_path)
# =============================
# :둘: 필요한 컬럼 선택
# =============================
# 입력(X) = 합계 일사량, 평균기온
# 출력(Y) = 총량
X_train = train_df[['합계 일사량(MJ/m2)', '평균기온(°C)']]
y_train = train_df['총량(KWh)']
X_test = test_df[['합계 일사량(MJ/m2)', '평균기온(°C)']]
y_test = test_df['총량(KWh)']
# =============================
# :셋: 결측치 및 0 값 처리 (선형 보간)
# =============================
# 0 → np.nan
X_train = X_train.replace(0, np.nan).interpolate(method='linear')
X_test = X_test.replace(0, np.nan).interpolate(method='linear')
y_train = y_train.replace(0, np.nan).interpolate(method='linear')
y_test = y_test.replace(0, np.nan).interpolate(method='linear')
# =============================
# :넷: 스케일링
# =============================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
# Train 데이터 기준으로 학습 후 Transform
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
# Test 데이터는 Train 기준 스케일 적용
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
# =============================
# :다섯: Train/Test 확인
# =============================
print("X_train_scaled shape:", X_train_scaled.shape)
print("y_train_scaled shape:", y_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)
print("y_test_scaled shape:", y_test_scaled.shape)

# =============================
# :일: XGBoost 모델 생성 (수정된 하이퍼파라미터 + early stopping 적용)
# =============================
xgb_model = xgb.XGBRegressor(
    n_estimators=1000,       # 트리 개수 늘림 -> 안정적 학습 # 수정됨
    max_depth=5,             # 트리 깊이 약간 증가 -> 복잡 패턴 학습 # 수정됨
    learning_rate=0.01,      # 낮춘 학습률 -> 안정적 수렴 # 수정됨
    subsample=0.8,           # 데이터 샘플링 비율 유지
    colsample_bytree=0.8,    # 피처 샘플링 유지
    gamma=0.1,               # 노드 분할 최소 손실 증가 -> 과적합 방지 # 수정됨
    min_child_weight=3,      # 자식 노드 최소 가중치 -> 과적합 완화 # 수정됨
    random_state=42,
    eval_metric='rmse',       # 수정됨
)

'''
# =============================
# :일: XGBoost 모델 생성
# =============================
xgb_model = xgb.XGBRegressor(
    n_estimators=500,   # 트리 개수
    max_depth=4,        # 트리 깊이
    learning_rate=0.05,  # 학습률
    subsample=0.8,      # 데이터 샘플링 비율 - 랜덤하게 선택 (고정이 가능하긴 함.)
    colsample_bytree=0.8,
    random_state=42
)  # 무슨 파라미터 그거 값
'''
# =============================
# :둘: 모델 학습 (callback으로 early stopping)
# =============================
# XGBoost 3.1.1에서는 fit()에서 early_stopping_rounds 직접 사용 불가
callbacks = [xgb.callback.EarlyStopping(rounds=50, save_best=True)]

xgb_model.fit(
    X_train_scaled,
    y_train_scaled.ravel(),
    eval_set=[(X_test_scaled, y_test_scaled.ravel())],  # 추가
    verbose=True,  # 추가
)
# =============================
# :셋: 예측
# =============================
y_pred_scaled = xgb_model.predict(X_test_scaled)
# =============================
# :넷: 스케일 원래대로 복원
# =============================
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_test_orig = scaler_y.inverse_transform(y_test_scaled)
# =============================
# :다섯: 예측 결과 엑셀 저장
# =============================
result_df = test_df.copy()
result_df['Predicted_Total'] = y_pred  # 수정된 부분
save_path = r"C:/Users/alsl0/Documents/python/Predicted_Test_Data.xlsx"
result_df.to_excel(save_path, index=False)
print(f":흰색_확인_표시: 예측 결과를 엑셀로 저장했습니다: {save_path}")
# =============================
# :여섯: 평가
# =============================
mae = mean_absolute_error(y_test_orig, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
r2 = r2_score(y_test_orig, y_pred)
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.4f}")


# =============================
# :일곱: 실제 vs 이론발전량 비교 그래프
# =============================
# 1️⃣ 한글 폰트 설정
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(15, 6))

# 꺾은선 그래프 (실제값과 예측값 비교)
plt.plot(y_test.index, y_test_orig, label='실제 발전량 (KWh)',
         color='tab:blue', linewidth=2)
plt.plot(y_test.index, y_pred, label='이론발전량 (예측, KWh)',
         color='tab:orange', linestyle='-', linewidth=2)

# 만약 test_df에 '일시' 컬럼이 있다면, x축을 날짜로 표시
if '일시' in test_df.columns:
    plt.xticks(
        ticks=range(0, len(test_df), max(1, len(test_df)//10)),
        labels=pd.to_datetime(test_df['일시']).dt.strftime(
            '%Y-%m-%d')[::max(1, len(test_df)//10)],
        rotation=45
    )
    plt.xlabel('일시')
else:
    plt.xlabel('Index')

plt.ylabel('발전량 (KWh)')
plt.title('24년도 실제 발전량 vs 예측 이론발전량')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
