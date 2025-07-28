from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns  

# ---------------- 파일 경로 선택 ----------------
Tk().withdraw()
input_path = askopenfilename(
    title="분석할 엑셀 파일을 선택하세요",
    filetypes=[("Excel Files", "*.xlsx *.xlsm *.xls")]
)
if not input_path:
    print("입력 파일이 선택되지 않았습니다.")
    exit()

output_path = asksaveasfilename(
    title="결과를 저장할 파일 이름을 지정하세요",
    defaultextension=".xlsx",
    filetypes=[("Excel Files", "*.xlsx")]
)
if not output_path:
    print("출력 파일 경로가 지정되지 않았습니다.")
    exit()

# (2) 엑셀 읽기
df = pd.read_excel(input_path, sheet_name="정리된분석데이터")

# (3) 그룹 설정
df["Group"] = df["ID"].str.extract(r"^([A-Z]+)")
df["BT"] = (df["Group"] == "BT").astype(int)
df["CO"] = (df["Group"] == "CO").astype(int)

# (4) 분석 대상 변수
y_cols = ["dPPD", "dCAL", "dBDAC"]
x_cols = ["mob", "angle", "BT", "CO"]

results = []
plot_dir = os.path.splitext(output_path)[0] + "_plots"
os.makedirs(plot_dir, exist_ok=True)

for y in y_cols:
    df_valid = df[(df[y] != "") & (df["angle"] != "-")].copy()
    df_valid[y] = pd.to_numeric(df_valid[y], errors="coerce")
    df_valid["angle"] = pd.to_numeric(df_valid["angle"], errors="coerce")
    df_valid = df_valid.dropna(subset=[y] + x_cols)

    X = df_valid[x_cols]
    X = sm.add_constant(X)
    y_data = df_valid[y]

    model = sm.OLS(y_data, X).fit()
    summary = model.summary2().tables[1]

    vif_df = pd.DataFrame()
    for i in range(X.shape[1]):
        vif_df.loc[i, "Variable"] = X.columns[i]
        vif_df.loc[i, "VIF"] = variance_inflation_factor(X.values, i)

    for idx in summary.index:
        results.append({
            "Dependent": y,
            "Variable": idx,
            "Coef": summary.loc[idx, "Coef."],
            "t-value": summary.loc[idx, "t"],
            "p-value": summary.loc[idx, "P>|t|"],
            "% CI Low": summary.loc[idx, "[0.025"],
            "% CI Upp": summary.loc[idx, "0.975]"],
            "R^2": model.rsquared if idx == "const" else ""
        })

    for _, row in vif_df.iterrows():
        results.append({
            "Dependent": y,
            "Variable": f"VIF_{row['Variable']}",
            "Coef": row["VIF"]
        })

    # 시각화 저장
    plt.figure(figsize=(6, 4))
    sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, line_kws={"color": "red"})
    plt.title(f"Residuals vs Fitted - {y}")
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.axhline(0, color="gray", linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{y}_residuals.png"))
    plt.close()

    sm.qqplot(model.resid, line='45')
    plt.title(f"Q-Q Plot - {y}")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{y}_qqplot.png"))
    plt.close()

# (5) 엑셀로 저장
pd.DataFrame(results).to_excel(output_path, index=False)
print(f"회귀 분석 완료. 결과 파일 저장됨: {output_path}")
print(f"시각화 PNG 파일 저장됨: {plot_dir}")
