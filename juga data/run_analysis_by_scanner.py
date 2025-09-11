
import pandas as pd
import numpy as np
from scipy import stats

# 1. 데이터 로드 및 전처리 (이전 코드와 동일)
file_path = "c:\\python\\juga data\\scan data 2.csv"
df = pd.read_csv(file_path)
df = df.drop(0).reset_index(drop=True)
df.columns = ['OriginalIndex', 'Measurement', 'Rep', 'Value', 'Workflow']
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
df.dropna(subset=['Value'], inplace=True)
df[['Scanner', 'Method']] = df['Workflow'].str.split('\?', expand=True)
df['Measurement'] = df['Measurement'].str.replace('\?| ', '', regex=True)

reference_values = df[df['Scanner'] == 'reference'].set_index('Measurement')['Value'].to_dict()
df_analysis = df[df['Scanner'] != 'reference'].copy()

df_analysis['ReferenceValue'] = df_analysis['Measurement'].map(reference_values)
df_analysis['TruenessError'] = np.abs(df_analysis['Value'] - df_analysis['ReferenceValue'])

# --- 분석 수행 ---
results = []

# 스캔 방식별로 루프를 돌며 분석
for method_type in ['5-segments', 'full']:
    print(f"--- '{method_type}' 방식 내 스캐너 비교 ---")
    
    # 현재 방식에 해당하는 데이터 필터링
    method_data = df_analysis[df_analysis['Method'] == method_type]
    
    # wireless vs wired 데이터 분리
    wireless_data = method_data[method_data['Scanner'] == 'wireless']
    wired_data = method_data[method_data['Scanner'] == 'wired']

    # --- Trueness 분석 (t-test) ---
    trueness_ttest = stats.ttest_ind(wireless_data['TruenessError'], wired_data['TruenessError'], equal_var=False)
    
    # --- Precision 분석 (Levene test) ---
    precision_levene = stats.levene(wireless_data['Value'], wired_data['Value'])

    # 결과 저장
    results.append({
        'Comparison': f"Within '{method_type}' method",
        'Analysis': 'Trueness (Mean Error)',
        'Group1_Wireless': wireless_data['TruenessError'].mean(),
        'Group2_Wired': wired_data['TruenessError'].mean(),
        'p-value': trueness_ttest.pvalue,
        'Hypothesis': 'H0: 스캐너 간 Trueness에 차이가 없다.'
    })
    results.append({
        'Comparison': f"Within '{method_type}' method",
        'Analysis': 'Precision (Std Dev)',
        'Group1_Wireless': wireless_data['Value'].std(),
        'Group2_Wired': wired_data['Value'].std(),
        'p-value': precision_levene.pvalue,
        'Hypothesis': 'H0: 스캐너 간 Precision에 차이가 없다.'
    })

# 2. 결과 파일로 저장
results_df = pd.DataFrame(results)
output_path = 'C:\\python\\analysis_results_by_scanner.xlsx'
results_df.to_excel(output_path, index=False)

print(f"\n분석 완료. 결과가 '{output_path}' 파일에 저장되었습니다.")
print("결과 요약:")
print(results_df)
