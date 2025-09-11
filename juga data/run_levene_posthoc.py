
import pandas as pd
from scipy import stats
import os
import itertools

# --- 설정 ---
output_dir = "c:\\python\\juga data"
output_excel_path = os.path.join(output_dir, "levene_posthoc_dim1_results.xlsx")

# --- 1. 데이터 로드 및 전처리 ---
file_path = "c:\\python\\juga data\\scan data 2.csv"
df = pd.read_csv(file_path)
df = df.drop(0).reset_index(drop=True)
df.columns = ['OriginalIndex', 'Measurement', 'Rep', 'Value', 'Workflow']
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
df.dropna(subset=['Value'], inplace=True)
df[['Scanner', 'Method']] = df['Workflow'].str.split('\\?', expand=True)
df['Measurement'] = df['Measurement'].str.replace('\\?| ', '', regex=True)
df_analysis = df[df['Scanner'] != 'reference'].copy()
df_analysis['Group'] = df_analysis['Scanner'] + "_" + df_analysis['Method']

# --- 2. LinearDim.1 데이터에 대한 Levene 사후분석 ---
print("--- Analyzing Precision Post-Hoc for LinearDim.1 ---")
dim1_data = df_analysis[df_analysis['Measurement'] == 'LinearDim.1']
groups = sorted(dim1_data['Group'].unique())

# 모든 그룹 쌍에 대한 조합 생성
pairs = list(itertools.combinations(groups, 2))
posthoc_results = []

for g1, g2 in pairs:
    data1 = dim1_data['Value'][dim1_data['Group'] == g1]
    data2 = dim1_data['Value'][dim1_data['Group'] == g2]
    
    stat, p_val = stats.levene(data1, data2)
    posthoc_results.append({
        'Group1': g1,
        'Group2': g2,
        'Statistic': stat,
        'Raw_p-value': p_val
    })

posthoc_df = pd.DataFrame(posthoc_results)

# 본페로니 보정 (p-value * 비교 횟수)
num_comparisons = len(pairs)
posthoc_df['Bonferroni_p-value'] = posthoc_df['Raw_p-value'] * num_comparisons
# 보정된 p-value가 1을 넘지 않도록 조정
posthoc_df['Bonferroni_p-value'] = posthoc_df['Bonferroni_p-value'].clip(upper=1.0)

posthoc_df['Significant (p<0.05)'] = posthoc_df['Bonferroni_p-value'] < 0.05

# --- 3. 결과 저장 및 출력 ---
posthoc_df = posthoc_df.sort_values(by='Bonferroni_p-value')
posthoc_df.to_excel(output_excel_path, index=False)

print(f"\n분석 완료. 결과가 다음 파일에 저장되었습니다: {output_excel_path}")
print("\n--- Levene 검정 사후분석 결과 (Bonferroni 보정) ---")
print(posthoc_df.to_string())
