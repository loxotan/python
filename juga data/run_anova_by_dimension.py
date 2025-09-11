
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os

# --- 설정 ---
output_dir = "c:\\python\\juga data"
output_excel_path = os.path.join(output_dir, "anova_by_dimension_results.xlsx")

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
df_analysis['Group'] = df_analysis['Scanner'] + "_" + df_analysis['Method']

# --- 2. Dimension 별 ANOVA 및 Levene 검정 수행 ---
main_results = []
tukey_results_dict = {}

for dim in sorted(df_analysis['Measurement'].unique()):
    print(f"\n--- Analyzing Dimension: {dim} ---")
    dim_data = df_analysis[df_analysis['Measurement'] == dim]
    
    groups = sorted(dim_data['Group'].unique())
    
    # Trueness (ANOVA)
    trueness_data_groups = [dim_data['TruenessError'][dim_data['Group'] == g] for g in groups]
    f_stat_trueness, p_val_trueness = stats.f_oneway(*trueness_data_groups)
    main_results.append({
        'Measurement': dim,
        'Analysis': 'Trueness (ANOVA)',
        'Statistic': f_stat_trueness,
        'p-value': p_val_trueness,
        'Significant (p<0.05)': p_val_trueness < 0.05
    })

    # ANOVA가 유의미하면 Tukey HSD 사후검정 수행
    if p_val_trueness < 0.05:
        tukey = pairwise_tukeyhsd(endog=dim_data['TruenessError'], groups=dim_data['Group'], alpha=0.05)
        tukey_results_dict[dim + '_Trueness'] = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
        print(f"  -> Found significant difference in Trueness (p={p_val_trueness:.4f}). Tukey HSD results generated.")

    # Precision (Levene Test)
    precision_data_groups = [dim_data['Value'][dim_data['Group'] == g] for g in groups]
    stat_precision, p_val_precision = stats.levene(*precision_data_groups)
    main_results.append({
        'Measurement': dim,
        'Analysis': 'Precision (Levene)',
        'Statistic': stat_precision,
        'p-value': p_val_precision,
        'Significant (p<0.05)': p_val_precision < 0.05
    })
    if p_val_precision < 0.05:
        print(f"  -> Found significant difference in Precision (p={p_val_precision:.4f}).")

# --- 3. 결과 파일로 저장 ---
main_results_df = pd.DataFrame(main_results)

with pd.ExcelWriter(output_excel_path) as writer:
    main_results_df.to_excel(writer, sheet_name='ANOVA_Levene_Summary', index=False)
    for sheet_name, df_tukey in tukey_results_dict.items():
        df_tukey.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"\n분석 완료. 결과가 다음 파일에 저장되었습니다: {output_excel_path}")
print("\n--- 요약 ---")
print(main_results_df.to_string())
