
import pandas as pd
import numpy as np
from scipy import stats
import os

# --- 설정 ---
# 출력 디렉터리 설정
output_dir = "c:\\python\\juga data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_excel_path = os.path.join(output_dir, "exploratory_analysis_results.xlsx")

# --- 1. 데이터 로드 및 전처리 ---
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

# --- 2. 모든 조합에 대한 탐색적 분석 ---
all_results = []
measurement_types = df_analysis['Measurement'].unique()

# 비교 1: 스캐너 내에서 방식(full/5-seg) 비교 (측정 위치별 세분화)
for scanner in ['wireless', 'wired']:
    for dim in measurement_types:
        subset = df_analysis[(df_analysis['Scanner'] == scanner) & (df_analysis['Measurement'] == dim)]
        
        group1 = subset[subset['Method'] == 'full']
        group2 = subset[subset['Method'] == '5-segments']

        if len(group1) > 1 and len(group2) > 1:
            # Trueness
            ttest_trueness = stats.ttest_ind(group1['TruenessError'], group2['TruenessError'], equal_var=False)
            all_results.append({
                'Comparison': f'Method (in {scanner})',
                'Groups': 'full vs 5-seg',
                'Measurement': dim,
                'Analysis': 'Trueness',
                'p-value': ttest_trueness.pvalue
            })
            # Precision
            levene_precision = stats.levene(group1['Value'], group2['Value'])
            all_results.append({
                'Comparison': f'Method (in {scanner})',
                'Groups': 'full vs 5-seg',
                'Measurement': dim,
                'Analysis': 'Precision',
                'p-value': levene_precision.pvalue
            })

# 비교 2: 방식 내에서 스캐너(wired/wireless) 비교 (측정 위치별 세분화)
for method in ['full', '5-segments']:
    for dim in measurement_types:
        subset = df_analysis[(df_analysis['Method'] == method) & (df_analysis['Measurement'] == dim)]
        
        group1 = subset[subset['Scanner'] == 'wireless']
        group2 = subset[subset['Scanner'] == 'wired']

        if len(group1) > 1 and len(group2) > 1:
            # Trueness
            ttest_trueness = stats.ttest_ind(group1['TruenessError'], group2['TruenessError'], equal_var=False)
            all_results.append({
                'Comparison': f'Scanner (in {method})',
                'Groups': 'wireless vs wired',
                'Measurement': dim,
                'Analysis': 'Trueness',
                'p-value': ttest_trueness.pvalue
            })
            # Precision
            levene_precision = stats.levene(group1['Value'], group2['Value'])
            all_results.append({
                'Comparison': f'Scanner (in {method})',
                'Groups': 'wireless vs wired',
                'Measurement': dim,
                'Analysis': 'Precision',
                'p-value': levene_precision.pvalue
            })

# --- 3. 결과 정리 및 저장 ---
results_df = pd.DataFrame(all_results)
results_df['Significant (p<0.05)'] = results_df['p-value'] < 0.05

# p-value 기준으로 정렬하여 유의미한 결과를 쉽게 찾도록 함
results_df = results_df.sort_values(by='p-value', ascending=True)

results_df.to_excel(output_excel_path, index=False)

print(f"분석 완료. 결과가 다음 파일에 저장되었습니다: {output_excel_path}")
print("\n--- 유의미한 차이가 발견된 분석 (p < 0.05) ---")
significant_results = results_df[results_df['Significant (p<0.05)']]
if significant_results.empty:
    print("모든 세분화된 분석에서 통계적으로 유의미한 차이가 발견되지 않았습니다.")
else:
    print(significant_results.to_string())
