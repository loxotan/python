
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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

# 분석을 위한 그룹 컬럼 생성
df_analysis['Group'] = df_analysis['Scanner'] + "_" + df_analysis['Method']

# --- 분석 수행 ---
results_summary = {}

# 2. Trueness 분석 (ANOVA)
groups_trueness = [df_analysis['TruenessError'][df_analysis['Group'] == g] for g in df_analysis['Group'].unique()]
f_stat_trueness, p_val_trueness = stats.f_oneway(*groups_trueness)
results_summary['Trueness_ANOVA'] = {'F-statistic': f_stat_trueness, 'p-value': p_val_trueness}

print(f"--- Trueness (정확도) 전체 비교 (ANOVA) ---")
print(f"P-value: {p_val_trueness:.4f}")

# ANOVA 결과가 유의미할 경우 사후 분석 수행
if p_val_trueness < 0.05:
    print("\nANOVA 결과가 유의미하므로, 그룹 간 차이를 확인하기 위해 Tukey HSD 사후 분석을 수행합니다.")
    tukey_results = pairwise_tukeyhsd(endog=df_analysis['TruenessError'], groups=df_analysis['Group'], alpha=0.05)
    print(tukey_results)
    results_summary['Trueness_TukeyHSD'] = str(tukey_results)
else:
    print("ANOVA 결과, 그룹 간 Trueness에 통계적으로 유의미한 차이가 없습니다.")


# 3. Precision 분석 (Levene Test)
groups_precision = [df_analysis['Value'][df_analysis['Group'] == g] for g in df_analysis['Group'].unique()]
stat_precision, p_val_precision = stats.levene(*groups_precision)
results_summary['Precision_Levene'] = {'Statistic': stat_precision, 'p-value': p_val_precision}

print(f"\n--- Precision (정밀도) 전체 비교 (Levene Test) ---")
print(f"P-value: {p_val_precision:.4f}")
if p_val_precision < 0.05:
    print("Levene 검정 결과, 그룹 간 Precision에 통계적으로 유의미한 차이가 있습니다.")
else:
    print("Levene 검정 결과, 그룹 간 Precision에 통계적으로 유의미한 차이가 없습니다.")


# 4. 결과 파일로 저장
output_path = 'C:\\python\\analysis_results_all_groups.xlsx'
with pd.ExcelWriter(output_path) as writer:
    pd.DataFrame.from_dict(results_summary, orient='index').to_excel(writer, sheet_name='Summary')
    if 'Trueness_TukeyHSD' in results_summary:
        pd.DataFrame(data=tukey_results._results_table.data[1:], columns=tukey_results._results_table.data[0]).to_excel(writer, sheet_name='Tukey_HSD_Details', index=False)

print(f"\n분석 완료. 결과가 '{output_path}' 파일에 저장되었습니다.")
