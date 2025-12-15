import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from matplotlib import rc

# =============================
# 한글 폰트 설정
# =============================
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# -------------------------------------------------------
# ① 파일 경로 지정
# -------------------------------------------------------
train_files = [
    r"C:/Users/alsl0/Documents/python/Update_set/영흥태양광#3_3/영흥태양광#3_3_22_data.xlsx",
    r"C:/Users/alsl0/Documents/python/Update_set/영흥태양광#3_3/영흥태양광#3_3_23_data.xlsx",
    r"C:/Users/alsl0/Documents/python/Update_set/영흥태양광#3_3/영흥태양광#3_3_24_data.xlsx"
]
test_file = r"C:/Users/alsl0/Documents/python/Update_set/영흥태양광#3_3/영흥태양광#3_3_25_data.xlsx"

# -------------------------------------------------------
# ② 파일 읽기 및 train 데이터 병합
# -------------------------------------------------------
train = pd.concat([pd.read_excel(f) for f in train_files], ignore_index=True)
test = pd.read_excel(test_file)

# -------------------------------------------------------
# ③ 날짜 처리 및 1~8월 필터링
# -------------------------------------------------------
for df in [train, test]:
    df["일시"] = pd.to_datetime(df["일시"], errors="coerce")
    df["year"] = df["일시"].dt.year
    df["month"] = df["일시"].dt.month
    df["day"] = df["일시"].dt.day
    df["day_of_year"] = df["일시"].dt.dayofyear

train = train[train["month"] < 9]
test = test[test["month"] < 9]

# -------------------------------------------------------
# ✅ 정렬 보정 (오차율 불일치 방지 핵심)
# -------------------------------------------------------
train = train.sort_values("일시").reset_index(drop=True)
test = test.sort_values("일시").reset_index(drop=True)

# -------------------------------------------------------
# ④ 계절성 반영
# -------------------------------------------------------
for df in [train, test]:
    df["sin_day"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["cos_day"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)


# -------------------------------------------------------
# 🔧 실제 발전량이 0인 경우 평균 발전량으로 대체
# -------------------------------------------------------
for df in [train, test]:
    zero_count = (df["총량(KWh)"] == 0).sum()
    if zero_count > 0:
        mean_value = df.loc[df["총량(KWh)"] > 0, "총량(KWh)"].mean()
        df.loc[df["총량(KWh)"] == 0, "총량(KWh)"] = mean_value
        print(f"⚙️ {zero_count}개의 0값을 평균({mean_value:.2f})으로 대체했습니다.")


# -------------------------------------------------------
# ⑤ 입력 / 출력 변수 지정
# -------------------------------------------------------
X_cols = ["평균기온(°C)", "합계 일사량(MJ/m2)", "평균 풍속(m/s)",
          "평균 상대습도(%)", "sin_day", "cos_day"]
y_col = "총량(KWh)"

X_train, y_train = train[X_cols], train[y_col]
X_test, y_test = test[X_cols], test[y_col]

# -------------------------------------------------------
# ⑥ 성능 중심 Grid Search 후보
# -------------------------------------------------------
param_grid = {
    'n_estimators': [200, 500, 800],
    'max_depth': [4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'gamma': [0, 0.1, 0.3],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

xgb = XGBRegressor(random_state=42)
grid_search = GridSearchCV(estimator=xgb,
                           param_grid=param_grid,
                           scoring='neg_mean_absolute_error',
                           cv=3,
                           verbose=1,
                           n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# -------------------------------------------------------
# ⑦ 예측 (정렬 유지 상태에서 수행)
# -------------------------------------------------------
y_pred = best_model.predict(X_test)

# -------------------------------------------------------
# ⑧ 오차율 및 지표 계산
# -------------------------------------------------------
df_result = test.copy()
df_result["예측발전량"] = y_pred
df_result["오차율(%)"] = np.abs(y_pred - df_result["총량(KWh)"]) / \
    df_result["총량(KWh)"] * 100

'''
# 🔍 여기 아래에 바로 붙여 넣기!
# -------------------------------------------------------
# 🔍 특정 날짜 디버깅 및 원인 분석
# -------------------------------------------------------
debug_date = "2025-02-13"

target_row = df_result[df_result["일시"].dt.strftime("%Y-%m-%d") == debug_date]

if not target_row.empty:
    print(f"\n===== 🔎 {debug_date} 데이터 상세 분석 =====")
    print(target_row[["일시", "평균기온(°C)", "합계 일사량(MJ/m2)", "평균 풍속(m/s)",
                      "평균 상대습도(%)", "총량(KWh)", "예측발전량", "오차율(%)"]])

    y_pred_val = target_row["예측발전량"].values[0]
    y_true_val = target_row["총량(KWh)"].values[0]
    diff = y_pred_val - y_true_val
    sign = "과대예측" if diff > 0 else "과소예측"

    print("\n📊 예측 상세 분석")
    print(f"예측값 (y_pred): {y_pred_val:.3f}")
    print(f"실제값 (y_true): {y_true_val:.3f}")
    print(f"차이 (예측 - 실제): {diff:.3f} → {sign}")
    print(f"오차율: {abs(diff) / y_true_val * 100:.3f}%")

    # 📈 입력 특성 비교 (평균 대비)
    print("\n📈 입력 특성 비교 (해당일 vs 전체평균)")
    X_features = ["평균기온(°C)", "합계 일사량(MJ/m2)", "평균 풍속(m/s)", "평균 상대습도(%)"]
    means = df_result[X_features].mean()
    stds = df_result[X_features].std()
    target_values = target_row[X_features].iloc[0]

    compare_df = pd.DataFrame({
        "특성": X_features,
        "해당일 값": target_values.values,
        "전체 평균": means.values,
        "편차(해당일-평균)": (target_values - means).values,
        "표준화편차(Z-score)": ((target_values - means) / stds).values
    })

    print(compare_df.to_string(index=False, float_format="%.3f"))

    # 원인 추정
    print("\n🧠 원인 추정:")
    reasons = []
    if diff > 0:  # 과대예측
        if target_values["합계 일사량(MJ/m2)"] < means["합계 일사량(MJ/m2)"]:
            reasons.append("☁️ 일사량이 평소보다 낮아 실제 발전량이 줄었을 가능성")
        if target_values["평균 풍속(m/s)"] > means["평균 풍속(m/s)"]:
            reasons.append("💨 풍속이 높아 모듈 냉각이나 오염 영향 가능성")
        if target_values["평균기온(°C)"] > means["평균기온(°C)"]:
            reasons.append("🔥 온도가 높아 모듈 효율 저하로 실제 발전량이 낮아졌을 가능성")
        if target_values["평균 상대습도(%)"] > means["평균 상대습도(%)"]:
            reasons.append("💧 습도가 높아 산란광 비율이 커져 효율 하락 가능성")
    else:  # 과소예측
        if target_values["합계 일사량(MJ/m2)"] > means["합계 일사량(MJ/m2)"]:
            reasons.append("☀️ 일사량이 평소보다 높아 실제 발전량이 예상보다 많았을 가능성")
        if target_values["평균기온(°C)"] < means["평균기온(°C)"]:
            reasons.append("❄️ 온도가 낮아 모듈 효율이 올라 실제 발전량이 많았을 가능성")
        if target_values["평균 풍속(m/s)"] < means["평균 풍속(m/s)"]:
            reasons.append("🍃 풍속이 낮아 예측 모델이 냉각 효과를 과소평가했을 가능성")
        if target_values["평균 상대습도(%)"] < means["평균 상대습도(%)"]:
            reasons.append("🌤️ 습도가 낮아 일사 투과율이 높아 실제 발전량이 많았을 가능성")

    if reasons:
        for r in reasons:
            print("-", r)
    else:
        print("📊 입력 변수만으로 뚜렷한 원인 추정이 어려움 (기타 외부 요인 가능)")

else:
    print(f"⚠️ {debug_date} 날짜 데이터가 test 셋에 없습니다.")
'''

# 불필요한 컬럼 제거
df_result = df_result.drop(
    columns=["year", "month", "day", "day_of_year"], errors="ignore")

# 기본 지표
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
smape = np.mean(2 * np.abs(y_pred - y_test) /
                (np.abs(y_pred) + np.abs(y_test))) * 100

# % 변환 지표
y_mean = np.mean(y_test)
mae_pct = (mae / y_mean) * 100
rmse_pct = (rmse / y_mean) * 100

# 오차율 통계
error_mean = df_result["오차율(%)"].mean()
error_max = df_result["오차율(%)"].max()
error_std = df_result["오차율(%)"].std()

# -------------------------------------------------------
# ⑨ 꺾은선 그래프 (가시성 향상 버전)
# -------------------------------------------------------
plt.figure(figsize=(14, 6))
plt.plot(df_result["일시"], df_result["총량(KWh)"],
         label="실제발전량", color="blue", marker="o", linewidth=1.5)
plt.plot(df_result["일시"], df_result["예측발전량"],
         label="예측발전량", color="red", marker="s", linestyle="-", linewidth=1.5)
plt.xlabel("일시")
plt.ylabel("발전량 (KWh)")
plt.title("25년 1월~8월 발전량 예측 vs 실제")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# ⑩ 결과 엑셀 저장 (오차율 통계 포함)
# -------------------------------------------------------
df_result["오차율 평균(%)"] = error_mean
df_result["오차율 최대(%)"] = error_max
df_result["오차율 표준편차(%)"] = error_std

metrics_dict = {
    "MAE": [mae], "MAE(%)": [mae_pct],
    "RMSE": [rmse], "RMSE(%)": [rmse_pct],
    "R²": [r2],
    "MAPE(%)": [mape], "SMAPE(%)": [smape],
    "최적파라미터": [str(grid_search.best_params_)]
}

with pd.ExcelWriter(r"C:/Users/alsl0/Documents/python/25년_영흥#3_3_XGBoost_결과.xlsx") as writer:
    df_result.to_excel(writer, sheet_name="예측_비교_오차율", index=False)
    pd.DataFrame(metrics_dict).to_excel(writer, sheet_name="검증지표", index=False)

print("✅ 정렬 보정 + 오차율 통합 + 가시성 높은 그래프 적용 완료")
