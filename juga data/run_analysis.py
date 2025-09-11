
import pandas as pd
import numpy as np
from scipy import stats
import io

# 1. 데이터 로드 및 전처리
file_path = "c:\\python\\juga data\\scan data 2.csv"
df = pd.read_csv(file_path)

# 불필요한 첫 번째 빈 행 제거 및 컬럼 이름 정리
df = df.drop(0).reset_index(drop=True)
df.columns = ['OriginalIndex', 'Measurement', 'Rep', 'Value', 'Workflow']

# 데이터 타입 변환
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
df.dropna(subset=['Value'], inplace=True)

# 'Workflow' 컬럼 분리
df[['Scanner', 'Method']] = df['Workflow'].str.split('\?', expand=True)
df['Measurement'] = df['Measurement'].str.replace('\?| ', '', regex=True)


# 기준(reference) 데이터 추출
reference_values = df[df['Scanner'] == 'reference'].set_index('Measurement')['Value'].to_dict()
print("기준 값:", reference_values)

# 분석할 데이터만 필터링
df_analysis = df[df['Scanner'] != 'reference'].copy()

# 2. Trueness (정확도) 계산
# 각 측정값에 해당하는 기준값을 매핑하여 절대 오차 계산
df_analysis['ReferenceValue'] = df_analysis['Measurement'].map(reference_values)
df_analysis['TruenessError'] = np.abs(df_analysis['Value'] - df_analysis['ReferenceValue'])

# 3. RMS 오차 계산
df_analysis['SquaredError'] = (df_analysis['Value'] - df_analysis['ReferenceValue']) ** 2

# 4. 허용 오차 계산 (예: 기준값의 ±0.2%를 허용 오차 범위로 설정)
TOLERANCE_PERCENTAGE = 0.002
df_analysis['ToleranceLower'] = df_analysis['ReferenceValue'] * (1 - TOLERANCE_PERCENTAGE)
df_analysis['ToleranceUpper'] = df_analysis['ReferenceValue'] * (1 + TOLERANCE_PERCENTAGE)
df_analysis['WithinTolerance'] = (df_analysis['Value'] >= df_analysis['ToleranceLower']) & (df_analysis['Value'] <= df_analysis['ToleranceUpper'])


# 분석 결과 저장을 위한 리스트
results = []

# 스캐너 타입별로 분석 수행
for scanner_type in ['wireless', 'wired']:
    print(f"\n--- {scanner_type.upper()} 스캐너 분석 ---")
    
    # full-arch vs 5-segments 데이터 분리
    full_arch_data = df_analysis[(df_analysis['Scanner'] == scanner_type) & (df_analysis['Method'] == 'full')]
    segment_data = df_analysis[(df_analysis['Scanner'] == scanner_type) & (df_analysis['Method'] == '5-segments')]

    # --- 가설 1: Trueness 분석 (t-test) ---
    trueness_ttest = stats.ttest_ind(full_arch_data['TruenessError'], segment_data['TruenessError'], equal_var=False)
    
    # --- 가설 2: Precision 분석 (Levene test) ---
    precision_levene = stats.levene(full_arch_data['Value'], segment_data['Value'])

    # --- 가설 3: RMS 및 허용 오차 분석 ---
    rms_full = np.sqrt(full_arch_data['SquaredError'].mean())
    rms_segment = np.sqrt(segment_data['SquaredError'].mean())
    
    # 허용 오차 내 비율 계산
    tolerance_rate_full = full_arch_data['WithinTolerance'].mean()
    tolerance_rate_segment = segment_data['WithinTolerance'].mean()

    # 허용 오차 비율 차이 검정 (Chi-squared test)
    contingency_table = [
        [full_arch_data['WithinTolerance'].sum(), len(full_arch_data) - full_arch_data['WithinTolerance'].sum()],
        [segment_data['WithinTolerance'].sum(), len(segment_data) - segment_data['WithinTolerance'].sum()]
    ]
    chi2, p_val_tolerance, _, _ = stats.chi2_contingency(contingency_table)

    # 결과 저장
    results.append({
        'Scanner': scanner_type,
        'Analysis': 'Trueness (Mean Error)',
        'Full-Arch': full_arch_data['TruenessError'].mean(),
        '5-Segment': segment_data['TruenessError'].mean(),
        'p-value': trueness_ttest.pvalue,
        'Hypothesis': 'H0: 두 워크플로우 간 Trueness에 차이가 없다.'
    })
    results.append({
        'Scanner': scanner_type,
        'Analysis': 'Precision (Std Dev)',
        'Full-Arch': full_arch_data['Value'].std(),
        '5-Segment': segment_data['Value'].std(),
        'p-value': precision_levene.pvalue,
        'Hypothesis': 'H0: 두 워크플로우 간 Precision에 차이가 없다.'
    })
    results.append({
        'Scanner': scanner_type,
        'Analysis': 'RMS Error',
        'Full-Arch': rms_full,
        '5-Segment': rms_segment,
        'p-value': 'N/A',
        'Hypothesis': 'H0: 두 워크플로우 간 RMS에 차이가 없다.'
    })
    results.append({
        'Scanner': scanner_type,
        'Analysis': f'Tolerance Rate (±{TOLERANCE_PERCENTAGE*100}%)',
        'Full-Arch': tolerance_rate_full,
        '5-Segment': tolerance_rate_segment,
        'p-value': p_val_tolerance,
        'Hypothesis': 'H0: 두 워크플로우 간 허용오차 내 비율에 차이가 없다.'
    })

# 5. 결과 파일로 저장
results_df = pd.DataFrame(results)
output_path = 'C:\\python\\analysis_results.xlsx'
results_df.to_excel(output_path, index=False)

print(f"\n분석 완료. 결과가 '{output_path}' 파일에 저장되었습니다.")
print("결과 요약:")
print(results_df)
