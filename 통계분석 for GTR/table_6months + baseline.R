# =========================================================
# RM_ready.csv → 분석 + 표 생성 (개선판)
# - LMM 결과(results_ppd_cal_bdac.xlsx)  [기능 유지]
# - 0m vs 6m 통합표(table_baseline_6m.xlsx)  [CAL,PPD,BD-AC,MOB,ANGLE,PI,GI]
# - Multiple_regression (ΔCAL 예측 회귀)      [별도 시트]
# - LMM Time×Group 시트                      [기능 유지]
# - Baseline characteristics (Table 1)       [신규 시트]
# - 모든 표에서 그룹 순서: BT–CO–BO
# =========================================================

options(stringsAsFactors = FALSE, warn = 1)

## ===== 0) 패키지 =====
pkgs <- c("dplyr","tibble","purrr","tidyr",
          "lme4","lmerTest","emmeans","performance",
          "openxlsx","stringr")
to_install <- pkgs[!pkgs %in% installed.packages()[,"Package"]]
if (length(to_install)) install.packages(to_install, dependencies = TRUE)

suppressPackageStartupMessages({
  library(dplyr); library(tibble); library(purrr); library(tidyr)
  library(lme4);  library(lmerTest); library(emmeans); library(performance)
  library(openxlsx); library(stringr)
})

`%||%` <- function(x, y) if (is.null(x) || length(x)==0) y else x

## ===== 공용 유틸 =====
BOM  <- "\uFEFF"; NBSP <- "\u00A0"
fix_names <- function(nm){
  nm <- enc2utf8(nm); nm <- gsub(BOM,"",nm,fixed=TRUE)
  nm <- gsub("\\x{00A0}", "", nm, perl=TRUE); trimws(nm)
}
# 안전한 숫자 변환(문자/기호 제거)
num <- function(x) suppressWarnings(as.numeric(gsub("[^0-9.+-]", "", as.character(x))))
to_bin01 <- function(x){
  s <- enc2utf8(as.character(x))
  s <- gsub(BOM,"",s,fixed=TRUE); s <- gsub("\\x{00A0}","",s,perl=TRUE)
  s <- gsub("[[:space:]\r\n\t]","",s,perl=TRUE); s[s=="+"]="1"; s[s=="-"]="0"
  s <- gsub("[^01]","",s,perl=TRUE); s[s==""] <- NA
  v <- suppressWarnings(as.integer(s)); v[!(v %in% c(0L,1L))] <- NA; v
}
make_sheet <- function(x){
  if (is.null(x)) return(data.frame(note="(no data)"))
  if (is.data.frame(x)) {
    if (ncol(x)==0) return(data.frame(note="(no data)"))
    return(x)
  }
  as.data.frame(x)
}
.safe <- function(expr, fallback) tryCatch(expr, error=function(e){ message("WARN: ", conditionMessage(e)); fallback })

## ===== 1) 파일 읽기 & 정리 =====
stopifnot(file.exists("RM_ready.csv"))
raw0 <- read.csv("RM_ready.csv", na.strings=c("","NA"),
                 check.names = FALSE, fileEncoding="UTF-8")
names(raw0) <- fix_names(names(raw0))

has_wide <- all(c("ID","Group","Time","PPD","CAL","BDAC") %in% names(raw0))
has_long <- all(c("ID","DV","Time","Value") %in% names(raw0))
if (!has_wide && !has_long) stop("RM_ready.csv 형식을 인식 불가.")

raw <- if (has_wide) raw0 else {
  raw0 %>% mutate(Group = factor(str_extract(ID, "^[A-Za-z]{2}"))) %>%
    pivot_wider(names_from = DV, values_from = Value)
}

for (nm in c("PPD","CAL","BDAC","MOB","ANGLE","COM","PC","PI","GI","WALLS"))
  if (!nm %in% names(raw)) raw[[nm]] <- NA

raw$ID    <- factor(raw$ID)
raw$Group <- factor(
  raw$Group %||% sub("^([A-Za-z]{2}).*$","\\1", as.character(raw$ID)),
  levels = c("BT","CO","BO")                    # ← 그룹 순서 BT–CO–BO 고정
)
raw$Time  <- enc2utf8(as.character(raw$Time))
raw$Time  <- gsub("\\s+","", raw$Time) |> tolower() |>
  gsub("^0m$|^0month(s)?$","0m", x=_) |>
  gsub("^3m$|^3month(s)?$","3m", x=_) |>
  gsub("^6m$|^6month(s)?$","6m", x=_)
raw$Time  <- factor(raw$Time, levels=c("0m","3m","6m"))

## ===== PATIENT (환자 ID) 처리 =====
if (!"PATIENT" %in% names(raw)) {
  stop("RM_ready.csv 에 'PATIENT' 열이 없습니다. 정리된분석데이터에서 PATIENT를 추가해 주세요.")
}
raw$PATIENT <- factor(raw$PATIENT)

## ===== 2) LMM 준비/적합 (기능 유지) =====
cont_cols <- intersect(c("PPD","CAL","BDAC","MOB","ANGLE","PI","GI"), names(raw))
df_keep <- raw %>%
  pivot_longer(all_of(cont_cols), names_to="DV", values_to="Value") %>%
  mutate(Value = num(Value)) %>%
  filter(!is.na(Value))

fit_lmm <- function(dv_name, dat=df_keep, tol_r2=1e-8){
  dat_dv <- dplyr::filter(dat, DV==dv_name)
  if (n_distinct(dat_dv$Time) < 2 || nrow(dat_dv) < 5) return(NULL)

  .std_names <- function(d){
    nm <- names(d); nm <- gsub("[[:space:]]+","_",nm)
    nm <- gsub("\\.","_",nm); nm <- gsub("\\(|\\)|>|<|=","",nm)
    names(d) <- nm; d
  }
  .pick <- function(nm, p){
    hit <- grep(p, nm, value=TRUE, ignore.case=TRUE)
    if (length(hit)) hit[1] else NA
  }

  m <- .safe(
    lmer(
      Value ~ Group * Time + (1 | PATIENT) + (1 | ID),
      data   = dat_dv,
      REML   = TRUE,
      control = lme4::lmerControl(check.conv.singular = "message")
    ),
    NULL
  )
  if (is.null(m)) return(NULL)

  # --- AIC/BIC/logLik/df 계산 ---
  ll  <- tryCatch(logLik(m), error = function(e) NA)
  aic <- tryCatch(AIC(m),    error = function(e) NA_real_)
  bic <- tryCatch(BIC(m),    error = function(e) NA_real_)
  dfm <- if (inherits(ll, "logLik")) attr(ll, "df") else NA_real_
  llv <- if (inherits(ll, "logLik")) as.numeric(ll) else NA_real_

  an_raw <- .safe(anova(m, type=3) |> as.data.frame() |> tibble::rownames_to_column("Effect"), NULL)
  if (is.null(an_raw)) return(NULL)
  an <- .std_names(an_raw); nm <- names(an)
  cs <- .pick(nm,"^Sum_?Sq$"); cm <- .pick(nm,"^Mean_?Sq$")
  cN <- .pick(nm,"^NumDF$");   cD <- .pick(nm,"^DenDF$")
  cF <- .pick(nm,"^F_?value$|^F$"); cp <- .pick(nm,"^Pr|^p_?value$")
  if (any(is.na(c(cs,cm,cN,cD,cF,cp)))) return(NULL)

  anova_tbl <- transmute(an, Effect,
                         `Sum Sq`=.data[[cs]], `Mean Sq`=.data[[cm]],
                         NumDF=.data[[cN]],    DenDF=.data[[cD]],
                         `F value`=.data[[cF]], `Pr(>F)`=.data[[cp]])

  effsize <- transmute(anova_tbl, Effect,
                       Eta2_partial = ifelse(is.finite(`F value`),
                                             (`F value`*NumDF)/((`F value`*NumDF)+DenDF), NA_real_),
                       CI_low=NA_real_, CI_high=NA_real_)

  r2 <- tryCatch(performance::r2_nakagawa(m, tolerance=tol_r2), error=function(e) NULL)
  r2_tbl <- tibble(Metric=c("Conditional R2","Marginal R2"),
                   Value=c(r2$R2_conditional %||% NA_real_, r2$R2_marginal %||% NA_real_))

  ## ----- ICC 계산 (PATIENT, ID 기준) -----
  vc <- tryCatch(as.data.frame(lme4::VarCorr(m)), error = function(e) NULL)

  if (!is.null(vc)) {
    var_pat <- if (any(vc$grp == "PATIENT")) vc$vcov[vc$grp == "PATIENT"][1] else NA_real_
    var_id  <- if (any(vc$grp == "ID"))       vc$vcov[vc$grp == "ID"][1]       else NA_real_
    var_res <- if (any(vc$grp == "Residual")) vc$vcov[vc$grp == "Residual"][1] else NA_real_

    var_tot <- if (is.finite(var_pat) | is.finite(var_id) | is.finite(var_res)) {
      sum(c(var_pat, var_id, var_res), na.rm = TRUE)
    } else NA_real_

    icc_pat <- if (!is.na(var_tot) && var_tot > 0) var_pat / var_tot else NA_real_
    icc_id  <- if (!is.na(var_tot) && var_tot > 0) var_id  / var_tot else NA_real_
  } else {
    var_pat <- var_id <- var_res <- var_tot <- icc_pat <- icc_id <- NA_real_
  }

  emm_g_t_obj <- emmeans(m, ~ Group | Time)
  emm_t_g_obj <- emmeans(m, ~ Time  | Group)

  list(
    model   = m,
    singular= lme4::isSingular(m, tol=1e-5),
    anova   = anova_tbl,
    effsize = effsize,
    r2      = r2_tbl,
    emm_g_t = as.data.frame(emm_g_t_obj),
    emm_t_g = as.data.frame(emm_t_g_obj),
    post_g_t= as.data.frame(pairs(emm_g_t_obj, adjust="bonferroni")),
    post_t_g= as.data.frame(contrast(emm_t_g_obj, method=list("0m-6m"=c(1,0,-1)), adjust="bonferroni")),
    aic     = aic,
    bic     = bic,
    logLik  = llv,
    df      = dfm,
    var_patient = var_pat,
    var_ID      = var_id,
    var_resid   = var_res,
    var_total   = var_tot,
    ICC_patient = icc_pat,
    ICC_ID      = icc_id
  )
}

dv_list  <- intersect(c("PPD","CAL","BDAC","MOB","ANGLE","PI","GI"), unique(df_keep$DV))
res_list <- setNames(lapply(dv_list, fit_lmm), dv_list)
res_list <- res_list[!sapply(res_list, is.null)]

# (요약 엑셀 저장 – 기존 기능 유지)
fmt_num <- function(x,d=4) if (is.numeric(x)) round(x,d) else x
ANOVA_all <- bind_rows(lapply(names(res_list), \(dv) res_list[[dv]]$anova %>% mutate(DV=dv, .before=1))) %>%
  mutate(across(where(is.numeric), ~round(.x,4)))
EffectSize_all <- bind_rows(lapply(names(res_list), \(dv) res_list[[dv]]$effsize %>% mutate(DV=dv, .before=1))) %>%
  mutate(across(where(is.numeric), ~round(.x,4)))
R2_all <- bind_rows(lapply(names(res_list), \(dv) res_list[[dv]]$r2 %>% mutate(DV=dv, .before=1))) %>%
  mutate(across(where(is.numeric), ~round(.x,4)))

# --- AIC/BIC/logLik/df 요약 ---
AIC_all <- bind_rows(lapply(names(res_list), function(dv){
  x <- res_list[[dv]]
  tibble(
    DV     = dv,
    AIC    = x$aic %||% NA_real_,
    BIC    = x$bic %||% NA_real_,
    logLik = x$logLik %||% NA_real_,
    df     = x$df %||% NA_real_
  )
})) %>% mutate(across(where(is.numeric), ~round(.x,4)))

# --- ICC 요약 ---
ICC_all <- bind_rows(lapply(names(res_list), function(dv){
  x <- res_list[[dv]]
  tibble(
    DV          = dv,
    var_patient = x$var_patient %||% NA_real_,
    var_ID      = x$var_ID      %||% NA_real_,
    var_resid   = x$var_resid   %||% NA_real_,
    var_total   = x$var_total   %||% NA_real_,
    ICC_patient = x$ICC_patient %||% NA_real_,
    ICC_ID      = x$ICC_ID      %||% NA_real_
  )
})) %>% mutate(across(where(is.numeric), ~round(.x,4)))

PostHoc_Group_at_Time <- bind_rows(lapply(names(res_list), \(dv) res_list[[dv]]$post_g_t %>% mutate(DV=dv, .before=1)))
PostHoc_Time_in_Group <- bind_rows(lapply(names(res_list), \(dv) res_list[[dv]]$post_t_g %>% mutate(DV=dv, .before=1)))
EMM_Group_at_Time     <- bind_rows(lapply(names(res_list), \(dv) res_list[[dv]]$emm_g_t %>% mutate(DV=dv, .before=1)))
EMM_Time_in_Group     <- bind_rows(lapply(names(res_list), \(dv) res_list[[dv]]$emm_t_g %>% mutate(DV=dv, .before=1)))
Singularity_flag      <- tibble(DV=names(res_list),
                                Singular=vapply(res_list, function(x) x$singular, logical(1)))

xlsx_path_results <- "results_ppd_cal_bdac.xlsx"
csv_path_master   <- "MASTER_summary_table.csv"
sheets <- list(
  ANOVA_TypeIII        = make_sheet(ANOVA_all),
  EffectSize_eta2p     = make_sheet(EffectSize_all),
  R2_Nakagawa          = make_sheet(R2_all),
  AIC_LMM              = make_sheet(AIC_all),
  ICC_LMM              = make_sheet(ICC_all),
  PostHoc_G_at_T       = make_sheet(PostHoc_Group_at_Time),
  PostHoc_T_in_G       = make_sheet(PostHoc_Time_in_Group),
  EMM_G_at_T           = make_sheet(EMM_Group_at_Time),
  EMM_T_in_G           = make_sheet(EMM_Time_in_Group),
  Singularity          = make_sheet(Singularity_flag)
)
.safe(openxlsx::write.xlsx(x=sheets, file=xlsx_path_results, asTable=TRUE, overwrite=TRUE),
      message("엑셀 저장 실패: ", xlsx_path_results))
.safe(write.csv(EMM_Time_in_Group, csv_path_master, row.names=FALSE),
      message("CSV 저장 실패: ", csv_path_master))

## ===== 3) 0m vs 6m 표 구성 (임상/방사선 분리, 스샷 배열) =====
t0 <- "0m"; t2 <- "6m"
raw_b6 <- raw %>%
  filter(Time %in% c(t0, t2)) %>%
  mutate(Group = factor(Group, levels = c("BT","CO","BO")))  # 그룹 순서 BT–CO–BO

fmt_p   <- function(p) ifelse(is.na(p),"", ifelse(p<0.001,"<0.001", sprintf("%.3f", p)))
fmt_ms  <- function(x) if (length(x)==0 || all(is.na(x))) "" else
  sprintf("%.1f \u00B1 %.1f", mean(x,na.rm=TRUE), sd(x,na.rm=TRUE))
fmt_pct <- function(x) if (length(x)==0 || all(is.na(x))) "" else
  sprintf("%.1f", 100*mean(x>0, na.rm=TRUE))

## ----- 군간 p (연속) -----
p_kw <- function(dat, var, time){
  sub <- dat %>% filter(Time==time) %>%
    mutate(v = num(.data[[var]])) %>%
    select(Group, v) %>% filter(!is.na(v))
  if (nrow(sub)<2 || dplyr::n_distinct(sub$Group)<2) return(NA_real_)
  kruskal.test(v ~ Group, data=sub)$p.value
}

## ===== 3-1) Baseline characteristics table (Table 1) =====
# 그룹 순서
grp_ord <- c("BT","CO","BO")

# baseline 0m 데이터만
base0 <- raw_b6 %>%
  filter(Time == t0) %>%
  mutate(Group = factor(Group, levels = grp_ord))

# 공통: 그룹별 값 벡터 뽑기
get_vec <- function(df, value_col){
  v <- setNames(df[[value_col]], as.character(df$Group))
  sapply(grp_ord, function(g){
    val <- v[[g]]
    if (is.null(val) || is.na(val)) "" else as.character(val)
  })
}

## ---- 1) N (patients) ----
tmpN <- base0 %>%
  group_by(Group) %>%
  summarise(N_pat = n_distinct(PATIENT), .groups = "drop")
N_pat <- get_vec(tmpN, "N_pat")

## ---- 2) Age (있을 때만) ----
age_cols <- intersect(c("Age","AGE","age"), names(base0))
age_row  <- NULL
if (length(age_cols)) {
  age_col <- age_cols[1]
  age_pat <- base0 %>%
    mutate(age = num(.data[[age_col]])) %>%
    group_by(Group, PATIENT) %>%
    summarise(age = age[which(!is.na(age))[1]], .groups = "drop")

  tmpA <- age_pat %>%
    group_by(Group) %>%
    summarise(ms = fmt_ms(age), .groups = "drop")
  age_ms <- get_vec(tmpA, "ms")

  p_age <- {
    df <- age_pat %>% filter(!is.na(age))
    if (nrow(df) < 3 || dplyr::n_distinct(df$Group) < 2) NA_real_
    else suppressWarnings(kruskal.test(age ~ Group, data = df)$p.value)
  }

  age_row <- tibble(
    Characteristic  = "Age (years, mean ± SD)",
    BT              = age_ms[1],
    CO              = age_ms[2],
    BO              = age_ms[3],
    `intra-group p` = fmt_p(p_age)
  )
}

## ---- 3) Defect walls (2 / 3) ----
walls_row <- NULL
if ("WALLS" %in% names(base0)) {
  walls_dat <- base0 %>% mutate(WALLS = num(WALLS))
  tab_w <- walls_dat %>%
    group_by(Group) %>%
    summarise(
      w2 = sum(WALLS == 2, na.rm = TRUE),
      w3 = sum(WALLS == 3, na.rm = TRUE),
      .groups = "drop"
    )

  walls_str <- setNames(sprintf("%d / %d", tab_w$w2, tab_w$w3),
                        as.character(tab_w$Group))
  walls_vec <- sapply(grp_ord, function(g){
    val <- walls_str[[g]]
    if (is.null(val)) "" else val
  })

  tab_f <- xtabs(~ Group + factor(WALLS, levels = c(2,3)), data = walls_dat)
  p_walls <- if (all(dim(tab_f) >= c(2,2))) {
    suppressWarnings(fisher.test(tab_f)$p.value)
  } else NA_real_

  walls_row <- tibble(
    Characteristic  = "Defect walls (2 / 3)",
    BT              = walls_vec[1],
    CO              = walls_vec[2],
    BO              = walls_vec[3],
    `intra-group p` = fmt_p(p_walls)
  )
}

## --- Tooth position (premolar / molar) ---
tooth_row <- NULL
tooth_cols <- intersect(c("TOOTH_POS","Tooth_position","tooth_position","TOOTH","Tooth"), names(base0))
if (length(tooth_cols)) {
  tcol <- tooth_cols[1]
  tvec <- tolower(as.character(base0[[tcol]]))
  tbin <- ifelse(tvec %in% c("premolar","pm","소구치"), "premolar",
                 ifelse(tvec %in% c("molar","m","대구치"), "molar", NA))

  tab_t <- base0 %>%
    mutate(pos = tbin) %>%
    group_by(Group) %>%
    summarise(
      prem  = sum(pos=="premolar", na.rm=TRUE),
      molar = sum(pos=="molar",   na.rm=TRUE),
      .groups="drop"
    )
  tooth_str <- setNames(sprintf("%d / %d", tab_t$prem, tab_t$molar),
                        as.character(tab_t$Group))
  tooth_vec <- sapply(grp_ord, function(g){
    val <- tooth_str[[g]]; if (is.null(val)) "" else val
  })

  tab_ft <- table(base0$Group, tbin)
  p_tooth <- if (nrow(tab_ft) >= 2 && ncol(tab_ft) >= 2) {
    suppressWarnings(fisher.test(tab_ft)$p.value)
  } else NA_real_

  tooth_row <- tibble(
    Characteristic = "Tooth position (premolar / molar)",
    BT = tooth_vec[1], CO = tooth_vec[2], BO = tooth_vec[3],
    `intra-group p` = fmt_p(p_tooth)
  )
}

## ---- 4) Baseline measurements 7개 (CAL~GI) ----
vars_base   <- c("CAL","PPD","MOB","BDAC","ANGLE","PI","GI")
labels_base <- c(
  CAL   = "CAL (mm, mean ± SD)",
  PPD   = "PPD (mm, mean ± SD)",
  MOB   = "Mobility (mean ± SD)",
  BDAC  = "LDF (mm, mean ± SD)",
  ANGLE = "Defect angle (°, mean ± SD)",
  PI    = "PI (mean ± SD)",
  GI    = "GI (mean ± SD)"
)

build_baseline_row <- function(var){
  if (!var %in% names(raw_b6)) return(NULL)

  sub0 <- raw_b6 %>%
    filter(Time == t0) %>%
    mutate(
      Group = factor(Group, levels = grp_ord),
      v     = num(.data[[var]])
    )

  tmp <- sub0 %>%
    group_by(Group) %>%
    summarise(ms = fmt_ms(v), .groups = "drop")

  ms_named <- setNames(tmp$ms, as.character(tmp$Group))
  vec <- sapply(grp_ord, function(g){
    val <- ms_named[[g]]
    if (is.null(val) || is.na(val)) "" else val
  })

  p0 <- p_kw(raw_b6, var, t0)

  tibble(
    Characteristic  = labels_base[[var]],
    BT              = vec[1],
    CO              = vec[2],
    BO              = vec[3],
    `intra-group p` = fmt_p(p0)
  )
}

rows_baseline <- vars_base |>
  intersect(names(raw_b6)) |>
  lapply(build_baseline_row) |>
  Filter(Negate(is.null), x = _) |>
  bind_rows()

## ---- 최종 Table 1 조립 ----
TABLE_baseline <- bind_rows(
  tibble(Characteristic = "Patient characteristic",
         BT = "", CO = "", BO = "", `intra-group p` = ""),
  tibble(Characteristic = "N (patients)",
         BT = N_pat[1], CO = N_pat[2], BO = N_pat[3],
         `intra-group p` = ""),
  age_row,
  tibble(Characteristic = "Defect characteristic",
         BT = "", CO = "", BO = "", `intra-group p` = ""),
  walls_row,
  tooth_row,
  tibble(Characteristic = "Baseline measurements",
         BT = "", CO = "", BO = "", `intra-group p` = ""),
  rows_baseline
)

## ----- 이하 0–6m 통합표용 함수들 ----- 

## 이분형 변환 (>0)
bin_gt0 <- function(x){
  v <- suppressWarnings(as.numeric(gsub("[^0-9.+-]", "", as.character(x))))
  ifelse(is.na(v), NA_integer_, as.integer(v > 0))
}

## 군내 p (연속; paired Wilcoxon)
p_wil <- function(dat, var, grp){
  wide <- dat %>%
    filter(Group==grp) %>%
    mutate(v = num(.data[[var]])) %>%
    select(ID, Time, v) %>%
    tidyr::pivot_wider(
      names_from  = Time,
      values_from = v,
      values_fn   = list(v = ~mean(., na.rm=TRUE))
    )
  if (!all(c(t0,t2) %in% names(wide))) return(NA_real_)
  a <- wide[[t0]]; b <- wide[[t2]]
  ok <- !is.na(a) & !is.na(b)
  if (!any(ok)) return(NA_real_)
  suppressWarnings(wilcox.test(a[ok], b[ok], paired=TRUE)$p.value)
}

## Δ(0m−6m)
delta_ms <- function(dat, var, grp){
  wide <- dat %>%
    filter(Group==grp) %>%
    mutate(v = num(.data[[var]])) %>%
    select(ID, Time, v) %>%
    tidyr::pivot_wider(
      names_from  = Time,
      values_from = v,
      values_fn   = list(v = ~mean(., na.rm=TRUE))
    )
  if (!all(c(t0,t2) %in% names(wide))) return(NULL)
  a <- wide[[t0]]; b <- wide[[t2]]
  ok <- !is.na(a) & !is.na(b)
  if (!any(ok)) return(NULL)
  d <- a[ok] - b[ok]
  list(mean = mean(d, na.rm=TRUE), sd = sd(d, na.rm=TRUE))
}

## Δ inter-group p
p_kw_delta <- function(dat, var){
  sub <- dat %>%
    mutate(v = num(.data[[var]])) %>%
    select(ID, Group, Time, v) %>%
    tidyr::pivot_wider(
      names_from  = Time,
      values_from = v,
      values_fn   = list(v = ~mean(., na.rm=TRUE))
    )
  if (!all(c(t0,t2) %in% names(sub))) return(NA_real_)
  ok <- !is.na(sub[[t0]]) & !is.na(sub[[t2]])
  if (!any(ok)) return(NA_real_)
  df <- data.frame(
    d = sub[[t0]][ok] - sub[[t2]][ok],
    g = droplevels(sub$Group[ok])
  )
  df <- df[is.finite(df$d) & !is.na(df$g), ]
  if (nrow(df) < 3 || dplyr::n_distinct(df$g) < 2) return(NA_real_)
  suppressWarnings(kruskal.test(d ~ g, data=df)$p.value)
}

## 군내 p (이분형; McNemar)
p_mcn <- function(dat, var, grp){
  wide <- dat %>%
    filter(Group==grp) %>%
    mutate(v = bin_gt0(.data[[var]])) %>%
    select(ID, Time, v) %>%
    tidyr::pivot_wider(names_from=Time, values_from=v)
  if (!all(c(t0,t2) %in% names(wide))) return(NA_real_)
  a <- wide[[t0]]; b <- wide[[t2]]
  ok <- !is.na(a) & !is.na(b); a <- a[ok]; b <- b[ok]
  if (length(a)==0) return(NA_real_)
  tab <- table(a,b)
  if (nrow(tab)<2 || ncol(tab)<2 || (tab[1,2]+tab[2,1])==0) return(NA_real_)
  mcnemar.test(tab)$p.value
}

## 군간 p (이분형; 시점별)
p_inter_bin <- function(dat, var, time){
  sub <- dat %>%
    filter(Time==time) %>%
    mutate(b = bin_gt0(.data[[var]])) %>%
    select(Group, b) %>%
    filter(!is.na(b))
  if (nrow(sub)<2 || dplyr::n_distinct(sub$Group)<2 || length(unique(sub$b))<2)
    return(NA_real_)
  tab <- table(sub$Group, sub$b)
  ok <- tryCatch(all(suppressWarnings(chisq.test(tab))$expected >= 5),
                 error=function(e) FALSE)
  if (ok) suppressWarnings(chisq.test(tab))$p.value else fisher.test(tab)$p.value
}

## 블록 빌더 (연속형)
build_block_cont2 <- function(dat, var, label){
  grps <- levels(dat$Group)   # BT–CO–BO 순서 유지
  rows <- lapply(seq_along(grps), function(i){
    g <- grps[i]
    d_res <- delta_ms(dat, var, g)
    tibble(
      `Clinical measurements` = if (i==1) label else "",
      Group    = g,
      Baseline = fmt_ms(num(dat %>% filter(Group==g, Time==t0) %>% pull(!!sym(var)))),
      `6 months` = fmt_ms(num(dat %>% filter(Group==g, Time==t2) %>% pull(!!sym(var)))),
      `Δ baseline to 6 months` =
        if (!is.null(d_res)) sprintf("%.1f ± %.1f", d_res$mean, d_res$sd) else "",
      `p (baseline to 6 months) intra-group` = fmt_p(p_wil(dat, var, g))
    )
  }) %>% bind_rows()

  dp <- p_kw_delta(dat, var)
  tibble(
    `Clinical measurements` = "p inter-group",
    Group = "",
    Baseline  = fmt_p(p_kw(dat, var, t0)),
    `6 months`= fmt_p(p_kw(dat, var, t2)),
    `Δ baseline to 6 months` = ifelse(is.na(dp),"",fmt_p(dp)),
    `p (baseline to 6 months) intra-group` = ""
  ) %>% bind_rows(rows, .)
}

## 블록 빌더 (이분형 %)
build_block_pct2 <- function(dat, var, label){
  grps <- levels(dat$Group)
  rows <- lapply(seq_along(grps), function(i){
    g <- grps[i]
    tibble(
      `Clinical measurements` = if (i==1) label else "",
      Group = g,
      Baseline  = fmt_pct(num(dat %>% filter(Group==g, Time==t0) %>% pull(!!sym(var)))),
      `6 months`= fmt_pct(num(dat %>% filter(Group==g, Time==t2) %>% pull(!!sym(var)))),
      `Δ baseline to 6 months` = {
        wide <- dat %>%
          filter(Group==g) %>%
          select(ID, Time, v = !!sym(var)) %>%
          tidyr::pivot_wider(names_from=Time, values_from=v)
        if (!all(c(t0,t2) %in% names(wide))) "" else {
          a <- num(wide[[t0]]); b <- num(wide[[t2]])
          ok <- !is.na(a) & !is.na(b)
          if (!any(ok)) "" else
            sprintf("%.1f", 100*(mean(a[ok] > 0) - mean(b[ok] > 0)))
        }
      },
      `p (baseline to 6 months) intra-group` = fmt_p(p_mcn(dat, var, g))
    )
  }) %>% bind_rows()

  tibble(
    `Clinical measurements` = "p inter-group",
    Group = "",
    Baseline  = fmt_p(p_inter_bin(dat, var, t0)),
    `6 months`= fmt_p(p_inter_bin(dat, var, t2)),
    `Δ baseline to 6 months` = "",
    `p (baseline to 6 months) intra-group` = ""
  ) %>% bind_rows(rows, .)
}

## 임상 블록
blocks_clin <- list(
  build_block_cont2(raw_b6, "PPD", "PPD (mm; mean ± SD)"),
  build_block_cont2(raw_b6, "CAL", "CAL (mm; mean ± SD)"),
  if ("MOB" %in% names(raw_b6))
    build_block_cont2(raw_b6, "MOB", "Mobility (mean ± SD)"),
  if ("PI" %in% names(raw_b6))
    build_block_pct2(raw_b6, "PI", "PI (%)"),
  if ("GI" %in% names(raw_b6))
    build_block_pct2(raw_b6, "GI (%)")
)
blocks_clin <- Filter(Negate(is.null), blocks_clin)

spacer <- tibble(
  `Clinical measurements` = "",
  Group = "",
  Baseline = "",
  `6 months` = "",
  `Δ baseline to 6 months` = "",
  `p (baseline to 6 months) intra-group` = ""
)

TABLE_clin <- purrr::imap_dfr(
  blocks_clin,
  \(x,i) bind_rows(x, if (i < length(blocks_clin)) spacer else NULL)
)

## 방사선 블록
blocks_rad <- list(
  build_block_cont2(raw_b6, "BDAC",
                    "Linear defect fill (BD-AC) (mm; mean ± SD)"),
  if ("ANGLE" %in% names(raw_b6))
    build_block_cont2(raw_b6, "ANGLE",
                      "Defect angle modification (°; mean ± SD)")
)
blocks_rad <- Filter(Negate(is.null), blocks_rad)

if (length(blocks_rad)) {
  TABLE_rad_core <- purrr::imap_dfr(
    blocks_rad,
    \(x,i) bind_rows(x, if (i < length(blocks_rad)) spacer else NULL)
  )

  header_rad <- tibble(
    `Clinical measurements` = "Radiographical measurements",
    Group = "Group",
    Baseline = "Baseline",
    `6 months` = "6 months",
    `Δ baseline to 6 months` = "Δ baseline to 6 months",
    `p (baseline to 6 months) intra-group` =
      "(baseline to 6 months) intra-group"
  )

  TABLE_base_6m <- bind_rows(
    TABLE_clin,
    spacer,            # GI 블록 뒤 빈 줄
    header_rad,
    TABLE_rad_core
  )
} else {
  TABLE_base_6m <- TABLE_clin
}

## ===== 4) ΔCAL 회귀 (별도 시트) =====
walls_by_id <- raw %>%
  mutate(WALLS=num(WALLS)) %>%
  group_by(ID) %>%
  summarise(WALLS = na.omit(WALLS)[1], .groups="drop")

wide0 <- raw %>%
  select(ID, Time, PPD, CAL, MOB, ANGLE) %>%
  mutate(across(c(PPD,CAL,MOB,ANGLE), num)) %>%
  pivot_wider(names_from=Time, values_from=c(PPD,CAL,MOB,ANGLE),
              values_fn=list(PPD=~mean(.,na.rm=TRUE),
                             CAL=~mean(.,na.rm=TRUE),
                             MOB=~mean(.,na.rm=TRUE),
                             ANGLE=~mean(.,na.rm=TRUE))) %>%
  left_join(walls_by_id, by="ID") %>%
  mutate(dCAL = CAL_0m - CAL_6m,
         PPD0 = PPD_0m, MOB0 = MOB_0m, ANGLE0 = ANGLE_0m,
         WALLS_3vs2 = ifelse(WALLS==3,1L, ifelse(WALLS==2,0L, NA_integer_))) %>%
  filter(complete.cases(dCAL,PPD0,MOB0,ANGLE0,WALLS_3vs2))

if (nrow(wide0) >= 5) {
  fit <- lm(dCAL ~ PPD0 + MOB0 + ANGLE0 + WALLS_3vs2, data=wide0)
  co  <- summary(fit)$coefficients; ci <- confint(fit)
  get_row <- function(lbl, nm){
    b <- co[nm,"Estimate"]; p <- co[nm,"Pr(>|t|)"]
    tibble(Variable=lbl,
           `β`=sprintf("%.2f",b),
           `95% CI`=sprintf("%.2f to %.2f", ci[nm,1], ci[nm,2]),
           p=ifelse(p<0.001,"<0.001",sprintf("%.4f",p)))
  }
  REG_table <- bind_rows(
    get_row("PPD baseline (mm)","PPD0"),
    get_row("Mobility","MOB0"),
    get_row("Defect angle (°)","ANGLE0"),
    get_row("Defect morphology (3 vs 2 walls)","WALLS_3vs2")
  )
} else {
  REG_table <- tibble(Variable="Insufficient complete cases for regression", `β`="", `95% CI`="", p="")
}

## ===== 5) 엑셀 작성 (table_baseline_6m.xlsx) =====
xlsx_path <- "table_baseline_6m.xlsx"
wb <- openxlsx::createWorkbook()

# 5-1) 0–6m 메인 테이블
addWorksheet(wb, "TABLE_base_6m")
writeData(wb, "TABLE_base_6m", TABLE_base_6m,
          startRow=1, startCol=1, withFilter=FALSE, na.string="")
setColWidths(wb, "TABLE_base_6m", cols=1:6,
             widths=c(30,8,18,18,22,26))
sty_hdr <- createStyle(textDecoration="bold", halign="center", valign="center")
addStyle(wb, "TABLE_base_6m", sty_hdr, rows=1, cols=1:6, gridExpand=TRUE)
lab_rows <- which(TABLE_base_6m$`Clinical measurements`!="" &
                    TABLE_base_6m$`Clinical measurements`!="p inter-group")+1
if (length(lab_rows))
  addStyle(wb, "TABLE_base_6m", createStyle(textDecoration="bold"),
           rows=lab_rows, cols=1, gridExpand=TRUE)
conditionalFormatting(wb, "TABLE_base_6m", cols=6,
                      rows=2:(nrow(TABLE_base_6m)+1),
  type="expression",
  rule='OR(LEFT($F2,1)="<", VALUE($F2)<0.05)',
  style=createStyle(bgFill="#C6EFCE"))
rows_pin <- which(TABLE_base_6m$`Clinical measurements`=="p inter-group")+1
if (length(rows_pin)){
  conditionalFormatting(wb, "TABLE_base_6m", cols=3:5, rows=rows_pin,
    type="expression",
    rule='OR(LEFT(CELL("contents",INDIRECT(ADDRESS(ROW(),COLUMN()))),1)="<", VALUE(INDIRECT(ADDRESS(ROW(),COLUMN())))<0.05)',
    style=createStyle(bgFill="#C6EFCE"))
}

# 5-2) Baseline characteristics (Table 1)
addWorksheet(wb, "Baseline_characteristics")
writeData(wb, "Baseline_characteristics", TABLE_baseline,
          startRow=1, startCol=1, withFilter=FALSE, na.string="")
setColWidths(wb, "Baseline_characteristics", cols=1:5,
             widths=c(35,10,10,10,14))
addStyle(wb, "Baseline_characteristics",
         createStyle(textDecoration="bold", halign="center"),
         rows=1, cols=1:5, gridExpand=TRUE)
hdr_rows1 <- which(TABLE_baseline$Characteristic %in%
                     c("Patient characteristic",
                       "Defect characteristic",
                       "Baseline measurements")) + 1L
if (length(hdr_rows1))
  addStyle(wb, "Baseline_characteristics",
           createStyle(textDecoration="bold"),
           rows=hdr_rows1, cols=1, gridExpand=TRUE)

# 5-3) 회귀 시트
addWorksheet(wb, "Multiple_regression")
writeData(wb, "Multiple_regression", REG_table,
          withFilter=FALSE, na.string="")
setColWidths(wb, "Multiple_regression", cols=1:4,
             widths=c(36,10,22,10))
addStyle(wb, "Multiple_regression",
         createStyle(textDecoration="bold", halign="center"),
         rows=1, cols=1:4, gridExpand=TRUE)

saveWorkbook(wb, xlsx_path, overwrite=TRUE)

## ===== 6) LMM overview 표 (Δ0–6m + LMM 지표) =====
suppressPackageStartupMessages({ library(emmeans); library(openxlsx); library(dplyr) })

label_for <- function(dv){
  switch(dv,
         "CAL"   = "CAL (mm)",
         "PPD"   = "PPD (mm)",
         "BDAC"  = "LDF (BD-AC) (mm)",
         "MOB"   = "Mobility",
         "PI"    = "PI",
         "GI"    = "GI",
         "ANGLE" = "Defect angle (°)",
         dv)
}
fmt_ci <- function(est, lcl, ucl){
  if (any(is.na(c(est,lcl,ucl)))) return("")
  sprintf("%.2f [%.2f, %.2f]", est, lcl, ucl)
}
get_r2 <- function(dv, type=c("Marginal R2","Conditional R2")){
  out <- rep(NA_real_, length(type))
  for(i in seq_along(type)){
    v <- R2_all %>%
      dplyr::filter(DV == dv, Metric == type[i]) %>%
      dplyr::pull(Value)
    out[i] <- ifelse(length(v), v[1], NA_real_)
  }
  out
}
get_icc <- function(dv){
  row <- ICC_all %>% dplyr::filter(DV == dv)
  if (!nrow(row)) return(c(NA_real_, NA_real_))
  c(row$ICC_patient[1], row$ICC_ID[1])
}
get_int_p <- function(res){
  an <- res$anova
  eff_name <- if ("Group:Time" %in% an$Effect) "Group:Time" else "Time:Group"
  p <- an$`Pr(>F)`[an$Effect == eff_name][1]
  p
}

build_lmm_block <- function(dv,
                            res_map = res_list,
                            dat     = raw_b6){

  res <- res_map[[dv]]
  if (is.null(res)) return(NULL)

  p_int <- get_int_p(res)

  icc  <- get_icc(dv)
  r2   <- get_r2(dv, c("Marginal R2","Conditional R2"))
  icc_pat <- icc[1]; icc_id <- icc[2]
  r2_m <- r2[1];  r2_c <- r2[2]
  aic_val <- res$aic
  bic_val <- res$bic
  ll_val  <- res$logLik

  m <- res$model
  emm_t_g <- emmeans::emmeans(m, ~ Time | Group)
  ct_g <- as.data.frame(
    summary(
      emmeans::contrast(emm_t_g,
                        method = list("0m-6m" = c(1,0,-1)),
                        adjust = "bonferroni"),
      infer = c(TRUE, TRUE)
    )
  )

  row_header <- tibble::tibble(
    `Clinical measurements`   = label_for(dv),
    Group                     = "",
    `Δ0–6m EMM [95% CI]`      = "",
    `p (Δ0–6m)`               = "",
    ICC_patient               = ifelse(is.na(icc_pat), "", sprintf("%.3f", icc_pat)),
    ICC_ID                    = ifelse(is.na(icc_id),  "", sprintf("%.3f", icc_id)),
    R2_marginal               = ifelse(is.na(r2_m),    "", sprintf("%.3f", r2_m)),
    R2_conditional            = ifelse(is.na(r2_c),    "", sprintf("%.3f", r2_c)),
    Sep                       = "",
    AIC                       = ifelse(is.na(aic_val), "", sprintf("%.1f", aic_val)),
    BIC                       = ifelse(is.na(bic_val), "", sprintf("%.1f", bic_val)),
    logLik                    = ifelse(is.na(ll_val),  "", sprintf("%.1f", ll_val))
  )

  grps <- c("BT","CO","BO")
  rows_grp <- lapply(seq_along(grps), function(i){
    g <- grps[i]
    r <- ct_g[ct_g$Group == g, ]
    if (!nrow(r)) {
      return(tibble::tibble(
        `Clinical measurements`   = if (i==1) "" else "",
        Group                     = g,
        `Δ0–6m EMM [95% CI]`      = "",
        `p (Δ0–6m)`               = "",
        ICC_patient               = "",
        ICC_ID                    = "",
        R2_marginal               = "",
        R2_conditional            = "",
        Sep                       = "",
        AIC                       = "",
        BIC                       = "",
        logLik                    = ""
      ))
    }
    tibble::tibble(
      `Clinical measurements`   = if (i==1) "" else "",
      Group                     = g,
      `Δ0–6m EMM [95% CI]`      =
        fmt_ci(r$estimate[1], r$lower.CL[1], r$upper.CL[1]),
      `p (Δ0–6m)`               = fmt_p(r$p.value[1]),
      ICC_patient               = "",
      ICC_ID                    = "",
      R2_marginal               = "",
      R2_conditional            = "",
      Sep                       = "",
      AIC                       = "",
      BIC                       = "",
      logLik                    = ""
    )
  }) %>% dplyr::bind_rows()

  row_inter <- tibble::tibble(
    `Clinical measurements`   = "p Time×Group interaction",
    Group                     = "",
    `Δ0–6m EMM [95% CI]`      = fmt_p(p_int),
    `p (Δ0–6m)`               = "",
    ICC_patient               = "",
    ICC_ID                    = "",
    R2_marginal               = "",
    R2_conditional            = "",
    Sep                       = "",
    AIC                       = "",
    BIC                       = "",
    logLik                    = ""
  )

  dplyr::bind_rows(row_header, rows_grp, row_inter)
}

dv_for_table <- c("CAL","PPD","MOB","PI","GI","BDAC","ANGLE")
dv_for_table <- dv_for_table[dv_for_table %in% names(res_list)]

blocks <- lapply(dv_for_table, build_lmm_block)
blocks <- Filter(Negate(is.null), blocks)

if (length(blocks)) {
  spacer_lmm <- tibble::tibble(
    `Clinical measurements`   = "",
    Group                     = "",
    `Δ0–6m EMM [95% CI]`      = "",
    `p (Δ0–6m)`               = "",
    ICC_patient               = "",
    ICC_ID                    = "",
    R2_marginal               = "",
    R2_conditional            = "",
    Sep                       = "",
    AIC                       = "",
    BIC                       = "",
    logLik                    = ""
  )

  LMM_overview <- purrr::imap_dfr(
    blocks,
    function(x, i) dplyr::bind_rows(x, if (i < length(blocks)) spacer_lmm else NULL)
  )

  wb2 <- if (file.exists(xlsx_path)) openxlsx::loadWorkbook(xlsx_path) else openxlsx::createWorkbook()
  if ("LMM_Time×Group" %in% openxlsx::sheets(wb2))
    openxlsx::removeWorksheet(wb2, "LMM_Time×Group")

  openxlsx::addWorksheet(wb2, "LMM_Time×Group")
  openxlsx::writeData(wb2, "LMM_Time×Group", LMM_overview,
                      withFilter = FALSE, na.string = "")

  openxlsx::writeData(wb2, "LMM_Time×Group", "",
                      startRow = 1, startCol = 9)

  openxlsx::setColWidths(
    wb2, "LMM_Time×Group",
    cols   = 1:12,
    widths = c(30, 8, 26, 10, 12, 12, 14, 14, 4, 6, 6, 7)
  )

  openxlsx::addStyle(
    wb2, "LMM_Time×Group",
    openxlsx::createStyle(textDecoration="bold", halign="center"),
    rows = 1, cols = 1:12, gridExpand = TRUE
  )

  lab_rows2 <- which(LMM_overview$`Clinical measurements` != "" &
                       LMM_overview$`Clinical measurements` != "p TimexGroup interaction") + 1L
  if (length(lab_rows2)){
    openxlsx::addStyle(
      wb2, "LMM_Time×Group",
      openxlsx::createStyle(textDecoration="bold"),
      rows = lab_rows2, cols = 1, gridExpand = TRUE
    )
  }

  openxlsx::saveWorkbook(wb2, xlsx_path, overwrite = TRUE)
}

## ===== 7) Pocket closure & COM rates (PC_COM_summary) =====
fmt_npct <- function(x){
  n1 <- sum(x==1L, na.rm=TRUE); d <- sum(!is.na(x))
  if (d==0) "" else sprintf("%d(%.0f%%)", n1, 100*n1/d)
}

build_rate_block <- function(dat, var, header_label){
  sub <- dat %>% filter(Time=="6m") %>% mutate(b = bin_gt0(.data[[var]]))
  total <- fmt_npct(sub$b)
  grps <- c("BT","CO","BO")
  g_rows <- lapply(grps, function(g){
    tibble(A = "", B = g,
           C = fmt_npct(sub %>% filter(Group==g) %>% pull(b)))
  }) %>% bind_rows()
  tab <- table(sub$Group, sub$b)
  pval <- if (nrow(tab)>=2 && ncol(tab)>=2) fisher.test(tab)$p.value else NA_real_
  p_str <- ifelse(is.na(pval),"", ifelse(pval<0.001,"<0.001", sprintf("%.2f", pval)))
  bind_rows(
    tibble(A = header_label, B = "",       C = ""),
    tibble(A = "",           B = "Total",  C = total),
    g_rows,
    tibble(A = "",           B = "p inter-group", C = p_str)
  )
}

TABLE_PC_COM <- bind_rows(
  build_rate_block(raw_b6, "PC",  "Pocket closure (PPD ≤ 3 or 4 mm with no BOP)"),
  build_rate_block(raw_b6, "COM", "COM (PPD ≤ 4 mm and CAL gain ≥ 3 mm)")
)

wb_pc <- if (file.exists(xlsx_path)) openxlsx::loadWorkbook(xlsx_path) else openxlsx::createWorkbook()
if ("PC_COM_summary" %in% openxlsx::sheets(wb_pc)) openxlsx::removeWorksheet(wb_pc, "PC_COM_summary")
openxlsx::addWorksheet(wb_pc, "PC_COM_summary")
openxlsx::writeData(wb_pc, "PC_COM_summary", TABLE_PC_COM,
                    startRow=1, startCol=1, withFilter=FALSE, na.string="")
openxlsx::setColWidths(wb_pc, "PC_COM_summary", cols=1:3, widths=c(48,18,14))
hdr_rows_pc <- which(TABLE_PC_COM$A != "") + 1L
if (length(hdr_rows_pc))
  openxlsx::addStyle(wb_pc, "PC_COM_summary",
                     createStyle(textDecoration="bold"),
                     rows=hdr_rows_pc, cols=1, gridExpand=TRUE)
openxlsx::addStyle(wb_pc, "PC_COM_summary",
                   createStyle(halign="center"),
                   rows=2:(nrow(TABLE_PC_COM)+1), cols=3, gridExpand=TRUE)
openxlsx::saveWorkbook(wb_pc, xlsx_path, overwrite=TRUE)

###########################################################
# 이하 Normality / Permutation ANCOVA / Dunn / Graph 부분
# (기존 코드에서 Group 레벨만 BT–CO–BO로 맞춤)
###########################################################

# ==== 정규성 & 등분산성 평가 ====
suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(car); library(rstatix); library(openxlsx)
})

vars_check <- c("BDAC","PPD","CAL","MOB","ANGLE","PI","GI")
vars_check <- intersect(vars_check, names(raw_b6))

norm_results <- list()
var_results  <- list()

for(v in vars_check){
  dat <- raw_b6 %>%
    filter(Time %in% c("0m","6m")) %>%
    mutate(Group=factor(Group, levels=c("BT","CO","BO")),
           Time = factor(Time, levels=c("0m","6m")),
           val  = num(.data[[v]]))

  for(t in c("0m","6m")){
    sub <- dat %>% filter(Time==t & is.finite(val))
    sw <- sub %>%
      group_by(Group) %>%
      summarise(
        n = n(),
        p_normality = ifelse(n >= 3, shapiro.test(val)$p.value, NA_real_)
      ) %>%
      mutate(Time=t, Variable=v)
    norm_results[[paste(v,t,sep="_")]] <- sw

    if(dplyr::n_distinct(sub$Group) > 1 && nrow(sub) > 4){
      lev <- car::leveneTest(val ~ Group, data=sub)
      var_results[[paste(v,t,sep="_")]] <- tibble(
        Variable=v, Time=t,
        F = lev[1,"F value"], p_var_equal = lev[1,"Pr(>F)"]
      )
    }
  }
}

normality_tbl <- bind_rows(norm_results) %>%
  mutate(signif = case_when(
    p_normality < 0.001 ~ "***",
    p_normality < 0.01  ~ "**",
    p_normality < 0.05  ~ "*",
    TRUE ~ ""
  ))

var_tbl <- bind_rows(var_results) %>%
  mutate(signif = case_when(
    p_var_equal < 0.001 ~ "***",
    p_var_equal < 0.01  ~ "**",
    p_var_equal < 0.05  ~ "*",
    TRUE ~ ""
  ))

xlsx_results <- xlsx_path_results %||% "results_ppd_cal_bdac.xlsx"
wb_norm <- if (file.exists(xlsx_results)) loadWorkbook(xlsx_results) else createWorkbook()

if ("Normality_Homoscedasticity" %in% sheets(wb_norm))
  removeWorksheet(wb_norm, "Normality_Homoscedasticity")
addWorksheet(wb_norm, "Normality_Homoscedasticity")

writeData(wb_norm, "Normality_Homoscedasticity", normality_tbl,
          startRow=1, startCol=1, withFilter=TRUE)
writeData(wb_norm, "Normality_Homoscedasticity", var_tbl,
          startRow=nrow(normality_tbl)+3, startCol=1, withFilter=TRUE)

setColWidths(wb_norm, "Normality_Homoscedasticity",
             cols=1:6, widths=c(12,8,6,12,12,6))
addStyle(wb_norm, "Normality_Homoscedasticity",
         createStyle(textDecoration="bold", halign="center"),
         rows=c(1, nrow(normality_tbl)+3), cols=1:6, gridExpand=TRUE)

saveWorkbook(wb_norm, xlsx_results, overwrite=TRUE)

# ==== Permutation ANCOVA at 6m (nonparametric) ====
suppressPackageStartupMessages({
  library(coin); library(dplyr); library(tidyr); library(purrr); library(openxlsx)
})

make_dat6 <- function(var){
  b <- raw_b6 %>% filter(Time=="0m") %>%
    transmute(ID, base = num(.data[[var]]))
  raw_b6 %>%
    filter(Time=="6m") %>%
    transmute(ID,
              Group=factor(Group, levels=c("BT","CO","BO")),
              val = num(.data[[var]])) %>%
    left_join(b, by="ID") %>%
    filter(is.finite(val), is.finite(base))
}

vars_np <- c("BDAC","PPD","CAL","ANGLE")
omni_list <- list(); pair_list <- list()

for(v in vars_np){
  d6 <- make_dat6(v)
  if(nrow(d6) < 6 || dplyr::n_distinct(d6$Group) < 2) next

  it <- coin::independence_test(val ~ Group + base, data=d6,
                                distribution = coin::approximate(B=9999))
  p_all <- coin::pvalue(it)
  stat  <- coin::statistic(it, type="standardized")

  omni_list[[v]] <- tibble(
    Variable = v, Test = "Permutation ANCOVA (coin)",
    N = nrow(d6), Groups = n_distinct(d6$Group),
    Statistic = as.numeric(stat), p = as.numeric(p_all)
  )

  gr <- levels(d6$Group)
  combs <- t(combn(gr, 2)); if(nrow(combs)==0) next
  pw <- apply(combs, 1, function(ab){
    sub <- d6 %>% filter(Group %in% ab) %>% droplevels()
    it2 <- coin::independence_test(val ~ Group + base, data=sub,
                                   distribution = coin::approximate(B=9999))
    tibble(group1=ab[1], group2=ab[2], p_raw = as.numeric(coin::pvalue(it2)))
  }) %>% bind_rows()
  if(nrow(pw)){
    pw$p_adj <- p.adjust(pw$p_raw, method="holm")
    pw$Variable <- v
    pair_list[[v]] <- pw %>% select(Variable, group1, group2, p_raw, p_adj)
  }
}

ANCOVA_perm_omni <- bind_rows(omni_list) %>%
  mutate(signif = dplyr::case_when(
    p < 0.001 ~ "***", p < 0.01 ~ "**", p < 0.05 ~ "*", TRUE ~ ""
  ))

ANCOVA_perm_pairs <- bind_rows(pair_list) %>%
  mutate(signif = dplyr::case_when(
    p_adj < 0.001 ~ "***", p_adj < 0.01 ~ "**", p_adj < 0.05 ~ "*", TRUE ~ ""
  ))

wb_anc <- if (file.exists(xlsx_results)) openxlsx::loadWorkbook(xlsx_results) else openxlsx::createWorkbook()

if ("ANCOVA_perm_6m" %in% openxlsx::sheets(wb_anc))
  openxlsx::removeWorksheet(wb_anc, "ANCOVA_perm_6m")
openxlsx::addWorksheet(wb_anc, "ANCOVA_perm_6m")
openxlsx::writeData(wb_anc, "ANCOVA_perm_6m", ANCOVA_perm_omni,
                    startRow=1, startCol=1, withFilter=TRUE)
openxlsx::writeData(wb_anc, "ANCOVA_perm_6m", ANCOVA_perm_pairs,
                    startRow=nrow(ANCOVA_perm_omni)+3, startCol=1, withFilter=TRUE)
openxlsx::setColWidths(wb_anc, "ANCOVA_perm_6m",
                       cols=1:7, widths=c(12,24,6,7,10,8,6))
openxlsx::addStyle(wb_anc, "ANCOVA_perm_6m",
                   openxlsx::createStyle(textDecoration="bold", halign="center"),
                   rows=c(1, nrow(ANCOVA_perm_omni)+3), cols=1:7, gridExpand=TRUE)
openxlsx::saveWorkbook(wb_anc, xlsx_results, overwrite=TRUE)

## ===== Dunn post hoc for baseline_6m (Kruskal–Wallis) =====
to_install <- c("FSA","dunn.test","rstatix")
ip <- to_install[!to_install %in% installed.packages()[,"Package"]]
if (length(ip)) install.packages(ip, dependencies = TRUE)
suppressPackageStartupMessages({ library(rstatix) })

dunn_time <- function(dat, var, time, p.adjust.method="bonferroni"){
  sub <- dat %>%
    filter(Time==time) %>%
    mutate(v = num(.data[[var]])) %>%
    select(Group, v) %>% filter(!is.na(v))
  if (nrow(sub) < 3 || dplyr::n_distinct(sub$Group) < 2)
    return(tibble(group1=character(), group2=character(), p.adj=numeric()))
  rstatix::dunn_test(v ~ Group, data=sub, p.adjust.method=p.adjust.method) %>%
    dplyr::transmute(Time=time, group1, group2,
                     p = p, p.adj = p.adj,
                     signif = case_when(
                       p.adj < 0.001 ~ "***",
                       p.adj < 0.01  ~ "**",
                       p.adj < 0.05  ~ "*",
                       TRUE ~ ""
                     ))
}

dunn_delta <- function(dat, var, p.adjust.method="bonferroni"){
  sub <- dat %>%
    mutate(v=num(.data[[var]])) %>%
    select(ID, Group, Time, v) %>%
    tidyr::pivot_wider(names_from=Time, values_from=v)
  if (!all(c("0m","6m") %in% names(sub)))
    return(tibble(group1=character(), group2=character(), p.adj=numeric()))
  sub <- sub %>% mutate(delta = `0m` - `6m`) %>%
    select(Group, delta) %>% filter(is.finite(delta))
  if (nrow(sub) < 3 || dplyr::n_distinct(sub$Group) < 2)
    return(tibble(group1=character(), group2=character(), p.adj=numeric()))
  rstatix::dunn_test(delta ~ Group, data=sub, p.adjust.method=p.adjust.method) %>%
    dplyr::transmute(Time="Δ(0m−6m)", group1, group2,
                     p = p, p.adj = p.adj,
                     signif = case_when(
                       p.adj < 0.001 ~ "***",
                       p.adj < 0.01  ~ "**",
                       p.adj < 0.05  ~ "*",
                       TRUE ~ ""
                     ))
}

vars_for_dunn <- c("BDAC","PPD","CAL","MOB","ANGLE")
vars_for_dunn <- intersect(vars_for_dunn, names(raw_b6))

DUNN_all <- purrr::map_dfr(vars_for_dunn, function(v){
  bind_rows(
    dunn_time(raw_b6, v, "0m"),
    dunn_time(raw_b6, v, "6m"),
    dunn_delta(raw_b6, v)
  ) %>% mutate(Variable = v, .before=1)
})

wb_dunn <- if (file.exists(xlsx_path)) openxlsx::loadWorkbook(xlsx_path) else openxlsx::createWorkbook()
if ("PostHoc_Dunn" %in% openxlsx::sheets(wb_dunn)) openxlsx::removeWorksheet(wb_dunn, "PostHoc_Dunn")
openxlsx::addWorksheet(wb_dunn, "PostHoc_Dunn")
openxlsx::writeData(wb_dunn, "PostHoc_Dunn", DUNN_all, withFilter=TRUE)
openxlsx::setColWidths(wb_dunn, "PostHoc_Dunn",
                       cols=1:6, widths=c(14,12,12,10,10,8))
openxlsx::addStyle(wb_dunn, "PostHoc_Dunn",
                   openxlsx::createStyle(textDecoration="bold", halign="center"),
                   rows=1, cols=1:6, gridExpand=TRUE)
openxlsx::saveWorkbook(wb_dunn, xlsx_path, overwrite=TRUE)

# 이후 그래프 부분은 기존 코드 그대로 사용 (Group factor는 이미 BT–CO–BO로 맞춰짐)
