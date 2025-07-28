from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 파일 경로 설정
Tk().withdraw()
input_path = askopenfilename(
    title="분석할 엑셀 파일을 선택하세요",
    filetypes=[("Excel Files", "*.xlsx *.xlsm *.xls")]
)
if not input_path:
    print("입력 파일이 선택되지 않았습니다.")
    exit()

output_file = asksaveasfilename(
    title="결과를 저장할 파일 이름을 지정하세요",
    defaultextension=".xlsx",
    filetypes=[("Excel Files", "*.xlsx")]
)
if not output_file:
    print("출력 파일 경로가 지정되지 않았습니다.")
    exit()

output_dir = os.path.dirname(output_file)

# 데이터 불러오기
df = pd.read_excel(input_path, sheet_name="정리된분석데이터")

# 이진화
df["mob_high"] = df["mob"].apply(lambda x: 1 if pd.to_numeric(x, errors="coerce") >= 2 else 0)
df["angle_high"] = df["angle"].apply(lambda x: 1 if pd.to_numeric(x, errors="coerce") > 30 else 0)

# 종속 변수와 독립 변수 정의
y_cols = ["dPPD", "dCAL", "dBDAC"]
base_x_cols = ["mob_high", "angle_high", "PPD_0m", "CAL_0m"]

results = []

for y in y_cols:
    df_valid = df[
        (df[y] != "") &
        (df["PPD_0m"] != "") &
        (df["CAL_0m"] != "")
    ].copy()

    # 숫자형 변환
    df_valid[y] = pd.to_numeric(df_valid[y], errors="coerce")
    for col in base_x_cols:
        df_valid[col] = pd.to_numeric(df_valid[col], errors="coerce")

    df_valid.dropna(subset=[y] + base_x_cols, inplace=True)

    # 다항항 생성
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(df_valid[base_x_cols])
    poly_feature_names = poly.get_feature_names_out(base_x_cols)
    X_df = pd.DataFrame(X_poly, columns=poly_feature_names)

    X_df.index = df_valid.index  # <- 이것 추가
    X_df = sm.add_constant(X_df)

    # 회귀 수행
    model = sm.OLS(df_valid[y], X_df).fit()


    # 회귀 계수 및 p-value 저장
    for var, coef, tval, pval, ci_low, ci_upp in zip(
        model.params.index,
        model.params.values,
        model.tvalues,
        model.pvalues,
        model.conf_int()[0],
        model.conf_int()[1]
    ):
        results.append({
            "Dependent": y,
            "Variable": var,
            "Coef": coef,
            "t-value": tval,
            "p-value": pval,
            "CI 2.5%": ci_low,
            "CI 97.5%": ci_upp,
            "R^2": model.rsquared
        })

    # VIF 저장
    vif_df = pd.DataFrame()
    vif_df["Variable"] = X_df.columns
    vif_df["VIF"] = [variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])]
    for _, row in vif_df.iterrows():
        results.append({
            "Dependent": y,
            "Variable": f"VIF_{row['Variable']}",
            "Coef": row["VIF"]
        })

    # 잔차 시각화
    residuals = model.resid
    fitted = model.fittedvalues
    plt.figure(figsize=(6, 4))
    sns.residplot(x=fitted, y=residuals, lowess=True, line_kws={"color": "red", "lw": 1})
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title(f"{y} - Residual Plot (Polynomial)")
    plt.axhline(0, color="gray", linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{y}_residual_plot_poly.png"))
    plt.close()

# 결과 저장
result_df = pd.DataFrame(results)
result_df.to_excel(output_file, index=False)

print(f"완료: {output_file}")
