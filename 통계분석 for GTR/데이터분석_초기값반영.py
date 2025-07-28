from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------- 파일 경로 선택 ----------------
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

# ---------------- 데이터 불러오기 ----------------
df = pd.read_excel(input_path, sheet_name="정리된분석데이터")

# ---------------- 이분형 변수 생성 ----------------
df["mob_high"] = pd.to_numeric(df["mob"], errors="coerce").apply(lambda x: 1 if x > 1 else 0)
df["angle_high"] = pd.to_numeric(df["angle"], errors="coerce").apply(lambda x: 1 if x > 30 else 0)

# ---------------- 회귀 분석 ----------------
results = []
y_cols = ["dPPD", "dCAL", "dBDAC"]
x_cols = ["mob_high", "angle_high", "PPD_0m", "CAL_0m"]

for y in y_cols:
    df_valid = df[
        (df[y] != "") &
        (df["PPD_0m"] != "") &
        (df["CAL_0m"] != "")
    ].copy()

    df_valid[y] = pd.to_numeric(df_valid[y], errors="coerce")
    for col in x_cols:
        df_valid[col] = pd.to_numeric(df_valid[col], errors="coerce")
    df_valid.dropna(subset=[y] + x_cols, inplace=True)

    Y = df_valid[y]
    X = sm.add_constant(df_valid[x_cols])
    model = sm.OLS(Y, X).fit()
    r2 = model.rsquared

    for var_name, coef, tval, pval, ci_low, ci_upp in zip(
        model.params.index, model.params, model.tvalues, model.pvalues,
        model.conf_int()[0], model.conf_int()[1]
    ):
        results.append({
            "Dependent": y,
            "Variable": var_name,
            "Coef": coef,
            "t-value": tval,
            "p-value": pval,
            "% CI Low": ci_low,
            "% CI Upp": ci_upp,
            "R^2": r2
        })

    vif_df = pd.DataFrame()
    vif_df["Variable"] = X.columns
    vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    for _, row in vif_df.iterrows():
        results.append({
            "Dependent": y,
            "Variable": f"VIF_{row['Variable']}",
            "Coef": row["VIF"]
        })

    # ---------------- 잔차 시각화 ----------------
    residuals = model.resid
    fitted = model.fittedvalues
    plt.figure(figsize=(6, 4))
    sns.residplot(x=fitted, y=residuals, lowess=True, line_kws={'color': 'red'})
    plt.axhline(0, color="gray", linestyle="--")
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title(f"{y} - Residual Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{y}_residual_plot.png"))
    plt.close()

# ---------------- 결과 저장 ----------------
result_df = pd.DataFrame(results)
result_df.to_excel(output_file, index=False)
print(f"분석 완료. 결과 저장 위치: {output_file}")
