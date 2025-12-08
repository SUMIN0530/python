# ============================================================
# 1️⃣ 엑셀 파일 불러오기 (한 달 기준 파일)
# ============================================================
import pandas as pd

irr = pd.read_excel("일사량.xlsx")
power = pd.read_excel("태양광 발전량.xlsx")
area = pd.read_excel("area.xlsx")  # 발전구분별 면적 정보

# ============================================================
# 2️⃣ 날짜 컬럼 정리 및 통일
# ============================================================
irr["일시"] = irr["일시"].astype(str).str.strip()
power["일자"] = power["일자"].astype(str).str.strip()

irr["date"] = pd.to_datetime(irr["일시"], errors="coerce").dt.date
power["date"] = pd.to_datetime(power["일자"], errors="coerce").dt.date

# ============================================================
# 1️⃣ irr와 power 병합 (발전구분 없는 경우)
# ============================================================
df = pd.merge(
    irr,
    power[["date", "발전구분", "총량(KW)"]],
    on="date",
    how="inner"
)

# ============================================================
# 2️⃣ 필요한 컬럼만 추출
# ============================================================
# irr에는 발전구분이 없으므로 date와 일사량만
irr = irr[["date", "합계 일사량(MJ/m2)"]].rename(
    columns={"합계 일사량(MJ/m2)": "daily_irradiance_MJ_m2"}
)
# power에는 date, 발전구분, 발전량
power = power[["date", "발전구분", "총량(KW)"]].rename(
    columns={"총량(KW)": "daily_generation_KW"}
)
# area 파일에는 발전구분과 면적
area = area[["발전구분", "면적(m2)"]].rename(columns={"면적(m2)": "panel_area"})

# ============================================================
# 3️⃣ irr와 power 병합 (irr에는 발전구분이 없으므로 date 기준)
# ============================================================
df = pd.merge(
    irr,
    power,
    on="date",
    how="inner"
)
# 발전 기준으로 면적 병합
df = pd.merge(df, area, on="발전구분", how="left")

# ============================================================
# 4️⃣ 단위 변환 및 이론 발전량 계산
# ============================================================
panel_efficiency = 0.175     # 17.5%
MJ_to_Wh = 277.778           # 변환계수

# 실제 발전량 (KW → kWh)
df["actual_generation_kWh"] = df["daily_generation_KW"]

# 이론 발전량 (MJ/m2 → kWh)
df["theoretical_generation_kWh"] = (
    df["daily_irradiance_MJ_m2"] * MJ_to_Wh *
    df["panel_area"] * panel_efficiency / 1000
)
'''
또 다른 식(직관적)
['예상 발전량(kWh)'] = ['합계 일사량(MJ/m2)'] * ['면적(㎡)'] * efficiency / 3.6
-> 변환계수 없음.
'''

# ===========================================================
# 공통 항목 제거
# ===========================================================
drop_daily_gen = df.drop(columns=['daily_generation_KW'])

# ============================================================
# 5️⃣ 병합 및 오차율 계산
# ============================================================
df["error_rate(%)"] = abs(df["actual_generation_kWh"] -
                          df["theoretical_generation_kWh"]) / df["theoretical_generation_kWh"] * 100

# 불량 판단 (15% 이상 오차 시 불량)
df["is_faulty"] = df["error_rate(%)"] > 15

# 전체 불량률
fault_rate = (df["is_faulty"].sum() / len(df)) * 100 if len(df) > 0 else 0

# ============================================================
# 6️⃣ 결과 저장
# ============================================================
df.to_excel("발전량_비교_결과2.xlsx", index=False)

# ============================================================
# 7️⃣ 결과 요약 출력
# ============================================================
print("=== ✅ 일별 발전량 비교 결과 ===")
print(df.head(10))
print(f"\n전체 불량률: {fault_rate:.2f}%")
print("\n📁 결과 파일: '발전량_비교_결과2.xlsx' 저장 완료")
