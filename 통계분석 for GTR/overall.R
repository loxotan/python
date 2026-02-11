# =========================================================
# Overall (no Group) 0m vs 6m summary + LMM(Time only)
# =========================================================
options(stringsAsFactors = FALSE, warn = 1)

pkgs <- c("dplyr","tibble","purrr","tidyr","openxlsx","stringr","lme4","lmerTest","emmeans","performance")
to_install <- pkgs[!pkgs %in% installed.packages()[,"Package"]]
if (length(to_install)) install.packages(to_install, dependencies = TRUE)

suppressPackageStartupMessages({
  library(dplyr); library(tibble); library(purrr); library(tidyr)
  library(openxlsx); library(stringr)
  library(lme4); library(lmerTest); library(emmeans); library(performance)
})

BOM  <- "\uFEFF"
fix_names <- function(nm){
  nm <- enc2utf8(nm); nm <- gsub(BOM,"",nm,fixed=TRUE)
  nm <- gsub("\\x{00A0}", "", nm, perl=TRUE); trimws(nm)
}
num <- function(x) suppressWarnings(as.numeric(gsub("[^0-9.+-]", "", as.character(x))))
fmt_p <- function(p) ifelse(is.na(p),"", ifelse(p<0.001,"<0.001", sprintf("%.3f", p)))
fmt_ms <- function(x) if (length(x)==0 || all(is.na(x))) "" else sprintf("%.1f \u00B1 %.1f", mean(x,na.rm=TRUE), sd(x,na.rm=TRUE))
safe <- function(expr, fallback) tryCatch(expr, error=function(e) fallback)

stopifnot(file.exists("RM_ready.csv"))
raw0 <- read.csv("RM_ready.csv", na.strings=c("","NA"), check.names = FALSE, fileEncoding="UTF-8")
names(raw0) <- fix_names(names(raw0))

has_wide <- all(c("ID","Time","PPD","CAL","BDAC") %in% names(raw0))
has_long <- all(c("ID","DV","Time","Value") %in% names(raw0))
if (!has_wide && !has_long) stop("RM_ready.csv 형식을 인식 불가.")

raw <- if (has_wide) raw0 else {
  raw0 %>% pivot_wider(names_from = DV, values_from = Value)
}

# 필요한 컬럼 없으면 생성
for (nm in c("PPD","CAL","BDAC","MOB","ANGLE","PI","GI"))
  if (!nm %in% names(raw)) raw[[nm]] <- NA

raw$ID <- factor(raw$ID)

raw$Time  <- enc2utf8(as.character(raw$Time))
raw$Time  <- gsub("\\s+","", raw$Time) |> tolower() |>
  gsub("^0m$|^0month(s)?$","0m", x=_) |>
  gsub("^3m$|^3month(s)?$","3m", x=_) |>
  gsub("^6m$|^6month(s)?$","6m", x=_)
raw$Time  <- factor(raw$Time, levels=c("0m","3m","6m"))

if (!"PATIENT" %in% names(raw)) stop("RM_ready.csv 에 'PATIENT' 열이 없습니다.")
raw$PATIENT <- factor(raw$PATIENT)

# 0m vs 6m만
t0 <- "0m"; t2 <- "6m"
dat_b6 <- raw %>% filter(Time %in% c(t0,t2))

# =========================
# 1) Overall summary table
# =========================
vars <- c("PPD","CAL","MOB","PI","GI","BDAC","ANGLE")
vars <- vars[vars %in% names(dat_b6)]

paired_p <- function(df, var){
  wide <- df %>%
    transmute(ID, Time, v=num(.data[[var]])) %>%
    pivot_wider(names_from=Time, values_from=v, values_fn=list(v=~mean(.,na.rm=TRUE)))
  if (!all(c(t0,t2) %in% names(wide))) return(NA_real_)
  a <- wide[[t0]]; b <- wide[[t2]]
  ok <- is.finite(a) & is.finite(b)
  if (!any(ok)) return(NA_real_)
  suppressWarnings(wilcox.test(a[ok], b[ok], paired=TRUE)$p.value)
}

delta_ms <- function(df, var){
  wide <- df %>%
    transmute(ID, Time, v=num(.data[[var]])) %>%
    pivot_wider(names_from=Time, values_from=v, values_fn=list(v=~mean(.,na.rm=TRUE)))
  if (!all(c(t0,t2) %in% names(wide))) return(c(NA_real_, NA_real_))
  a <- wide[[t0]]; b <- wide[[t2]]
  ok <- is.finite(a) & is.finite(b)
  if (!any(ok)) return(c(NA_real_, NA_real_))
  d <- a[ok] - b[ok]
  c(mean(d,na.rm=TRUE), sd(d,na.rm=TRUE))
}

label_for <- function(v){
  switch(v,
    PPD="PPD (mm)", CAL="CAL (mm)", MOB="Mobility",
    PI="PI", GI="GI", BDAC="LDF (BD-AC) (mm)", ANGLE="Defect angle (°)",
    v
  )
}

Overall_0m_6m <- map_dfr(vars, function(v){
  v0 <- dat_b6 %>% filter(Time==t0) %>% pull(!!sym(v)) %>% num()
  v6 <- dat_b6 %>% filter(Time==t2) %>% pull(!!sym(v)) %>% num()
  d  <- delta_ms(dat_b6, v)
  tibble(
    Variable = label_for(v),
    N_pairs  = {
      wide <- dat_b6 %>% transmute(ID, Time, v=num(.data[[v]])) %>%
        pivot_wider(names_from=Time, values_from=v, values_fn=list(v=~mean(.,na.rm=TRUE)))
      if (!all(c(t0,t2) %in% names(wide))) 0 else sum(is.finite(wide[[t0]]) & is.finite(wide[[t2]]))
    },
    `0m (mean±SD)` = fmt_ms(v0),
    `6m (mean±SD)` = fmt_ms(v6),
    `Δ(0m−6m) (mean±SD)` = ifelse(is.finite(d[1]), sprintf("%.1f \u00B1 %.1f", d[1], d[2]), ""),
    `paired p` = fmt_p(paired_p(dat_b6, v))
  )
})

# ======================================
# 2) LMM (Time only) optional summary
# ======================================
# long
df_long <- dat_b6 %>%
  select(ID, PATIENT, Time, all_of(vars)) %>%
  pivot_longer(all_of(vars), names_to="DV", values_to="Value") %>%
  mutate(Value=num(Value)) %>%
  filter(is.finite(Value)) %>%
  mutate(Time=factor(Time, levels=c("0m","6m"))) # LMM은 0m vs 6m만

fit_lmm_time <- function(dv){
  dd <- df_long %>% filter(DV==dv)
  if (n_distinct(dd$Time) < 2 || nrow(dd) < 5) return(NULL)
  m <- safe(lmer(Value ~ Time + (1|PATIENT) + (1|ID), data=dd, REML=TRUE), NULL)
  if (is.null(m)) return(NULL)

  # 0m-6m contrast (emmeans)
  emm <- emmeans(m, ~ Time)
  ct  <- as.data.frame(contrast(emm, method=list("0m-6m"=c(1,-1)), adjust="none"))
  # R2 / ICC
  r2  <- safe(performance::r2_nakagawa(m), NULL)
  vc  <- safe(as.data.frame(VarCorr(m)), NULL)

  var_pat <- if (!is.null(vc) && any(vc$grp=="PATIENT")) vc$vcov[vc$grp=="PATIENT"][1] else NA_real_
  var_id  <- if (!is.null(vc) && any(vc$grp=="ID"))      vc$vcov[vc$grp=="ID"][1]      else NA_real_
  var_res <- if (!is.null(vc) && any(vc$grp=="Residual"))vc$vcov[vc$grp=="Residual"][1] else NA_real_
  var_tot <- sum(c(var_pat,var_id,var_res), na.rm=TRUE)
  icc_pat <- ifelse(is.finite(var_tot) && var_tot>0, var_pat/var_tot, NA_real_)
  icc_id  <- ifelse(is.finite(var_tot) && var_tot>0, var_id/var_tot, NA_real_)

  tibble(
    DV = label_for(dv),
    `Δ0–6m (EMM)` = ifelse(nrow(ct), sprintf("%.2f", ct$estimate[1]), ""),
    `95% CI`      = ifelse(nrow(ct), sprintf("%.2f to %.2f", ct$lower.CL[1], ct$upper.CL[1]), ""),
    `p`           = ifelse(nrow(ct), fmt_p(ct$p.value[1]), ""),
    R2_marginal   = ifelse(!is.null(r2), sprintf("%.3f", r2$R2_marginal), ""),
    R2_conditional= ifelse(!is.null(r2), sprintf("%.3f", r2$R2_conditional), ""),
    ICC_patient   = ifelse(is.finite(icc_pat), sprintf("%.3f", icc_pat), ""),
    ICC_ID        = ifelse(is.finite(icc_id),  sprintf("%.3f", icc_id),  "")
  )
}

LMM_Time_only <- map_dfr(vars, fit_lmm_time)
if (!nrow(LMM_Time_only)) LMM_Time_only <- tibble(note="(insufficient data for LMM)")

# =========================
# 3) Write one Excel file
# =========================
out_xlsx <- "overall_change.xlsx"
wb <- createWorkbook()

addWorksheet(wb, "Overall_0m_6m")
writeData(wb, "Overall_0m_6m", Overall_0m_6m, withFilter=FALSE)
setColWidths(wb, "Overall_0m_6m", cols=1:6, widths=c(22,10,18,18,22,10))
addStyle(wb, "Overall_0m_6m", createStyle(textDecoration="bold", halign="center"),
         rows=1, cols=1:6, gridExpand=TRUE)

addWorksheet(wb, "LMM_Time_only")
writeData(wb, "LMM_Time_only", LMM_Time_only, withFilter=FALSE)
setColWidths(wb, "LMM_Time_only", cols=1:8, widths=c(22,12,18,10,12,14,12,10))
addStyle(wb, "LMM_Time_only", createStyle(textDecoration="bold", halign="center"),
         rows=1, cols=1:8, gridExpand=TRUE)

saveWorkbook(wb, out_xlsx, overwrite=TRUE)
message("Saved: ", out_xlsx)


