import os
import glob
import pandas as pd

update_set_path = r'C:\Users\alsl0\Documents\python\Update_set\누수방지'
summary_data = []

for region_folder in os.listdir(update_set_path):
    region_path = os.path.join(update_set_path, region_folder)
    if os.path.isdir(region_path):
        # 와일드카드 처리
        excel_files = glob.glob(os.path.join(region_path, '*누수제거.xlsx'))
        if excel_files:
            excel_file = excel_files[0]
            try:
                df = pd.read_excel(
                    excel_file, sheet_name='모델평가', engine='openpyxl')
                required_cols = ['MAE(%)', 'RMSE(%)', 'R2',
                                 'MAPE(%)', 'SMAPE(%)']
                row_data = {'지역명': region_folder}
                for col in required_cols:
                    row_data[col] = df[col].iloc[0] if col in df.columns else None
                summary_data.append(row_data)
            except Exception as e:
                print(f"{region_folder} 파일 읽기 실패: {e}")
        else:
            print(f"{region_folder} 폴더에 누수제거 엑셀 파일 없음")

summary_df = pd.DataFrame(summary_data, columns=[
                          '지역명', 'MAE(%)', 'RMSE(%)', 'R2', 'MAPE(%)', 'SMAPE(%)'])
output_file = os.path.join(update_set_path, '지역별_모델평가_누수제거_요약.xlsx')
summary_df.to_excel(output_file, index=False, engine='openpyxl')

print(f"✅ 요약 엑셀 파일 생성 완료: {output_file}")
