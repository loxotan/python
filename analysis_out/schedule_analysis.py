import argparse
import os
import sys
import json
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# Config
# ------------------------------
# ---- Add after imports ----
from matplotlib import font_manager, rcParams
import os, sys

def _set_korean_font():
    # 우선순위: 맑은 고딕 → Malgun Gothic → 나눔고딕 → Apple SD Gothic Neo(맥) → DejaVuSans
    candidates = [
        "Malgun Gothic", "맑은 고딕",
        "NanumGothic", "Nanum Gothic",
        "Apple SD Gothic Neo",  # mac
        "DejaVu Sans"           # fallback
    ]
    # 시스템 폰트에서 검색
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            rcParams["font.family"] = name
            break
    # 마이너스 기호 깨짐 방지
    rcParams["axes.unicode_minus"] = False

_set_korean_font()
# ---- end patch ----


CODE_SET = {
    "FO","GTR","peri-implantitis","GBR","CL","APF","CTG","FGG","1st","2nd","sinus","impt","exp","reopen"
}
PRIORITY = ["1st","2nd","peri-implantitis","GBR","sinus","impt","GTR","FO","CL","APF","CTG","FGG","exp","reopen"]

CATEGORY_RULES = [
    ({"FO","GTR"}, "1. FO, GTR"),
    ({"peri-implantitis"}, "2. peri-implantitis"),
    ({"GBR","impt","1st","2nd","sinus"}, "3. GBR, impt, 1st/2nd, sinus"),
    ({"CL","APF","CTG","FGG"}, "4. CL, MGS"),
]
DEFAULT_CATEGORY = "5. etc"

def normalize_code(x: object):
    if not isinstance(x, str):
        return None
    s = x.strip()
    if not s:
        return None
    t = s.lower()
    if t.startswith("peri-"):
        return "peri-implantitis"
    mapping = {
        "fo":"FO","gtr":"GTR","gbr":"GBR","cl":"CL",
        "apf":"APF","ctg":"CTG","fgg":"FGG","1st":"1st",
        "2nd":"2nd","sinus":"sinus","impt":"impt","exp":"exp","reopen":"reopen"
    }
    return mapping.get(t, None)

def find_surgery_type(row: pd.Series):
    # Collect candidates from columns D~G (0-index: 4..7)
    candidates = []
    for col in ["Summary_Part_4","Summary_Part_5","Summary_Part_6","Summary_Part_7"]:
        nval = normalize_code(row.get(col, None))
        if nval is not None and nval in CODE_SET:
            candidates.append(nval)
    for p in PRIORITY:
        if p in candidates:
            return p
    return None

def is_digits(x):
    if isinstance(x, (int, np.integer)):
        return True
    if isinstance(x, str) and x.isdigit():
        return True
    return False

def parse_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    rows = []
    for _, r in df.iterrows():
        surgery_no = r.get("Summary_Part_1", np.nan)
        implant_no = np.nan
        name = r.get("Summary_Part_2", None)
        patient_id = r.get("Summary_Part_3", None)
        surgery_type = find_surgery_type(r)

        # Implant pattern:
        # B: SurgeryNo (exists anyway)
        # C: ImplantNo (digits)
        # D: Name (not digits)
        # E: PatientID (digits, usually 6~7)
        # F: Code (1st/2nd/…)
        cond_implant = (
            is_digits(r.get("Summary_Part_2"))
            and not is_digits(r.get("Summary_Part_3"))
            and is_digits(str(r.get("Summary_Part_4")))
            and (normalize_code(r.get("Summary_Part_5")) in CODE_SET)
        )
        if cond_implant:
            implant_no = int(r.get("Summary_Part_2"))
            name = r.get("Summary_Part_3")
            patient_id = r.get("Summary_Part_4")
            if surgery_type is None:
                surgery_type = normalize_code(r.get("Summary_Part_5"))
        else:
            if surgery_type is None:
                surgery_type = normalize_code(r.get("Summary_Part_4"))

        rows.append({
            "Date": r["Date"],
            "Year": r["Year"],
            "Month": r["Month"],
            "SurgeryNo": pd.to_numeric(surgery_no, errors="coerce"),
            "ImplantNo": (np.nan if (pd.isna(implant_no)) else int(implant_no)),
            "Name": name,
            "PatientID": (None if pd.isna(patient_id) else str(patient_id)),
            "SurgeryType": surgery_type if surgery_type is not None else "etc"
        })

    out = pd.DataFrame(rows)
    out["ImplantCase"] = out["ImplantNo"].notna()
    out["Is1st"] = out["SurgeryType"].eq("1st")
    out["Is2nd"] = out["SurgeryType"].eq("2nd")
    return out

def categorize(surgery_type: str) -> str:
    for keys, label in CATEGORY_RULES:
        if surgery_type in keys:
            return label
    return DEFAULT_CATEGORY

def compute_tables(dfp: pd.DataFrame):
    monthly_all = dfp.groupby(["Year","Month"]).size().reset_index(name="Count")
    monthly_by_type = dfp.groupby(["Year","Month","SurgeryType"]).size().reset_index(name="Count")

    tmp = dfp.copy()
    tmp["Category"] = tmp["SurgeryType"].map(categorize)
    monthly_by_cat = tmp.groupby(["Year","Month","Category"]).size().reset_index(name="Count")

    top_names = dfp["Name"].value_counts().head(20).reset_index()
    top_names.columns = ["Name","Count"]

    top_patients = dfp["PatientID"].value_counts().head(20).reset_index()
    top_patients.columns = ["PatientID","Count"]

    implant_hist = dfp[dfp["ImplantCase"]]["ImplantNo"].value_counts().sort_index().reset_index()
    implant_hist.columns = ["ImplantNo","Count"]

    yearly_1st2nd = dfp[dfp["SurgeryType"].isin(["1st","2nd"])].groupby(["Year","SurgeryType"]).size().reset_index(name="Count")
    monthly_1st2nd = dfp[dfp["SurgeryType"].isin(["1st","2nd"])].groupby(["Year","Month","SurgeryType"]).size().reset_index(name="Count")

    return {
        "dfp": dfp,
        "monthly_all": monthly_all,
        "monthly_by_type": monthly_by_type,
        "monthly_by_cat": monthly_by_cat,
        "top_names": top_names,
        "top_patients": top_patients,
        "implant_hist": implant_hist,
        "yearly_1st2nd": yearly_1st2nd,
        "monthly_1st2nd": monthly_1st2nd,
    }

def save_tables_excel(tables: dict, out_xlsx: Path):
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as w:
        for k, v in tables.items():
            if isinstance(v, pd.DataFrame):
                v.to_excel(w, sheet_name=k[:31], index=False)

# =====================================================
# 월 0건인 달도 그래프에 0으로 표시하기 위한 헬퍼들
# =====================================================

def _fill_monthly_all(monthly_all: pd.DataFrame) -> pd.DataFrame:
    """Year별로 Month 1~12를 모두 포함하도록 Count 없는 달은 0으로 채움."""
    years = sorted(monthly_all["Year"].dropna().unique())
    full = []
    for y in years:
        for m in range(1, 13):
            full.append({"Year": y, "Month": m})
    full = pd.DataFrame(full)
    merged = full.merge(monthly_all, on=["Year","Month"], how="left")
    merged["Count"] = merged["Count"].fillna(0)
    return merged.sort_values(["Year","Month"])

def _fill_monthly_by_cat(monthly_by_cat: pd.DataFrame) -> pd.DataFrame:
    """Year × Month(1~12) × Category 전체 조합을 만들고 Count 없는 곳은 0."""
    years = sorted(monthly_by_cat["Year"].dropna().unique())
    cats = sorted(monthly_by_cat["Category"].dropna().unique())
    full = []
    for y in years:
        for m in range(1, 13):
            for c in cats:
                full.append({"Year": y, "Month": m, "Category": c})
    full = pd.DataFrame(full)
    merged = full.merge(monthly_by_cat, on=["Year","Month","Category"], how="left")
    merged["Count"] = merged["Count"].fillna(0)
    return merged.sort_values(["Year","Month","Category"])

def _fill_monthly_1st2nd(monthly_1st2nd: pd.DataFrame) -> pd.DataFrame:
    """Year × Month(1~12) × SurgeryType(1st/2nd) 전체 조합을 만들고 Count 없는 곳은 0."""
    if monthly_1st2nd.empty:
        return monthly_1st2nd
    years = sorted(monthly_1st2nd["Year"].dropna().unique())
    stypes = sorted(monthly_1st2nd["SurgeryType"].dropna().unique())
    full = []
    for y in years:
        for m in range(1, 13):
            for st in stypes:
                full.append({"Year": y, "Month": m, "SurgeryType": st})
    full = pd.DataFrame(full)
    merged = full.merge(monthly_1st2nd, on=["Year","Month","SurgeryType"], how="left")
    merged["Count"] = merged["Count"].fillna(0)
    return merged.sort_values(["Year","Month","SurgeryType"])

# =====================================================
# Plot functions (월 0건도 0으로 표현하도록 수정)
# =====================================================

def plot_monthly_totals(monthly_all: pd.DataFrame, outdir: Path):
    # Month 1~12 강제 포함
    monthly_all_filled = _fill_monthly_all(monthly_all)

    plt.figure(figsize=(9,5))
    for y, sub in monthly_all_filled.groupby("Year"):
        plt.plot(sub["Month"], sub["Count"], marker="o", label=str(y))
    plt.title("월별 총 수술 건수 추세")
    plt.xlabel("월")
    plt.ylabel("건수")
    plt.legend(title="연도")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "trend_monthly_total.png", dpi=150)
    plt.close()

def plot_monthly_category(monthly_by_cat: pd.DataFrame, outdir: Path):
    # Year × Month(1~12) × Category 전체 조합 생성
    monthly_by_cat_filled = _fill_monthly_by_cat(monthly_by_cat)

    pivot = (
        monthly_by_cat_filled
        .pivot_table(index=["Year","Month"], columns="Category", values="Count", fill_value=0)
        .reset_index()
    )
    for y, sub in pivot.groupby("Year"):
        ax = (
            sub.set_index("Month")
            .sort_index()
            .drop(columns=["Year"])
            .plot(kind="bar", stacked=True, figsize=(10,6))
        )
        ax.set_title(f"{y}년 월별 카테고리 분포")
        ax.set_xlabel("월")
        ax.set_ylabel("건수")
        ax.legend(title="Category")
        plt.tight_layout()
        plt.savefig(outdir / f"stacked_category_{y}.png", dpi=150)
        plt.close()

def plot_top_types(dfp: pd.DataFrame, outdir: Path):
    type_counts = dfp["SurgeryType"].value_counts().head(12)
    plt.figure(figsize=(8,5))
    plt.bar(type_counts.index.astype(str), type_counts.values)
    plt.title("술식 Top 12")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("건수")
    plt.tight_layout()
    plt.savefig(outdir / "top_types.png", dpi=150)
    plt.close()

def plot_1st2nd(monthly_1st2nd: pd.DataFrame, outdir: Path):
    if monthly_1st2nd.empty:
        return

    # Year × Month(1~12) × SurgeryType 전체 조합 생성
    monthly_1st2nd_filled = _fill_monthly_1st2nd(monthly_1st2nd)

    for stype, sub in monthly_1st2nd_filled.groupby("SurgeryType"):
        plt.figure(figsize=(9,5))
        for y, sub2 in sub.groupby("Year"):
            plt.plot(sub2["Month"], sub2["Count"], marker="o", label=str(y))
        plt.title(f"월별 {stype} 건수 추세")
        plt.xlabel("월")
        plt.ylabel("건수")
        plt.legend(title="연도")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(outdir / f"trend_{stype}.png", dpi=150)
        plt.close()


def main():
    p = argparse.ArgumentParser(description="Schedule analytics from count.xlsx")
    p.add_argument("--input", default="c:/Users/최수영/Desktop/술식별/count.xlsx", help="엑셀 파일 경로 (기본: count.xlsx)")
    p.add_argument("--sheet", default="Sheet1", help="시트명 (기본: Sheet1)")
    p.add_argument("--outdir", default="analysis_out", help="출력 폴더")
    args = p.parse_args()

    infile = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load
    df = pd.read_excel(infile, sheet_name=args.sheet)

    # Parse
    dfp = parse_dataframe(df)
    tables = compute_tables(dfp)

    # Save CSV + Excel
    dfp.to_csv(outdir / "parsed_schedule.csv", index=False, encoding="utf-8-sig")
    save_tables_excel(tables, outdir / "summary_tables.xlsx")

    # Plots
    plot_monthly_totals(tables["monthly_all"], outdir)
    plot_monthly_category(tables["monthly_by_cat"], outdir)
    plot_top_types(dfp, outdir)
    plot_1st2nd(tables["monthly_1st2nd"], outdir)

    # Meta
    with open(outdir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": dt.datetime.now().isoformat(),
            "input": str(infile.resolve()),
            "sheet": args.sheet
        }, f, ensure_ascii=False, indent=2)

    print(f"[OK] Parsed rows: {len(dfp)}")
    print(f"[OK] Output folder: {outdir.resolve()}")

if __name__ == "__main__":
    main()
