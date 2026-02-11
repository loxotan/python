# =========================================================
# RM_ready.csv → 분석 + 표 생성 (개선판)
# - LMM 결과(results_ppd_cal_bdac.xlsx)  [기능 유지]
# - 0m vs 6m 통합표(table_baseline_6m.xlsx)  [CAL,PPD,BD-AC,MOB,ANGLE,PI,GI]
# - Multiple_regression (ΔCAL 예측 회귀)      [별도 시트]
# - LMM Time×Group 시트                      [기능 유지]
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
make_sheet <- function(x){ if (is.null(x)) return(data.frame(note="(no data)"))
  if (is.data.frame(x)) { if (ncol(x)==0) return(data.frame(note="(no data)")); return(x) }
  as.data.frame(x) }
.safe <- function(expr, fallback) tryCatch(expr, error=function(e){ message("WARN: ", conditionMessage(e)); fallback })

## ===== 1) 파일 읽기 & 정리 =====
stopifnot(file.exists("RM_ready.csv"))
raw0 <- read.csv("RM_ready.csv", na.strings=c("","NA"), check.names = FALSE, fileEncoding="UTF-8")
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
raw$Group <- factor(raw$Group %||% sub("^([A-Za-z]{2}).*$","\\1", as.character(raw$ID)))
raw$Time  <- enc2utf8(as.character(raw$Time))
raw$Time  <- gsub("\\s+","", raw$Time) |> tolower() |>
  gsub("^0m$|^0month(s)?$","0m", x=_) |>
  gsub("^3m$|^3month(s)?$","3m", x=_) |>
  gsub("^6m$|^6month(s)?$","6m", x=_)
raw$Time  <- factor(raw$Time, levels=c("0m","3m","6m"))

## ===== 2) LMM 준비/적합 (기능 유지) =====
cont_cols <- intersect(c("PPD","CAL","BDAC","MOB","ANGLE","PI","GI"), names(raw))
df_keep <- raw %>% pivot_longer(all_of(cont_cols), names_to="DV", values_to="Value") %>%
  mutate(Value = num(Value)) %>% filter(!is.na(Value))

fit_lmm <- function(dv_name, dat=df_keep, tol_r2=1e-8){
  dat_dv <- dplyr::filter(dat, DV==dv_name)
  if (n_distinct(dat_dv$Time) < 2 || nrow(dat_dv) < 5) return(NULL)
  .std_names <- function(d){ nm <- names(d); nm <- gsub("[[:space:]]+","_",nm)
    nm <- gsub("\\.","_",nm); nm <- gsub("\\(|\\)|>|<|=","",nm); names(d) <- nm; d }
  .pick <- function(nm, p){ hit <- grep(p, nm, value=TRUE, ignore.case=TRUE); if (length(hit)) hit[1] else NA }

  m <- .safe(lmer(Value ~ Group * Time + (1|ID), data=dat_dv, REML=TRUE,
                  control=lme4::lmerControl(check.conv.singular="message")), NULL)
  if (is.null(m)) return(NULL)

  an_raw <- .safe(anova(m, type=3) |> as.data.frame() |> tibble::rownames_to_column("Effect"), NULL)
  if (is.null(an_raw)) return(NULL)
  an <- .std_names(an_raw); nm <- names(an)
  cs <- .pick(nm,"^Sum_?Sq$"); cm <- .pick(nm,"^Mean_?Sq$"); cN <- .pick(nm,"^NumDF$")
  cD <- .pick(nm,"^DenDF$");   cF <- .pick(nm,"^F_?value$|^F$"); cp <- .pick(nm,"^Pr|^p_?value$")
  if (any(is.na(c(cs,cm,cN,cD,cF,cp)))) return(NULL)

  anova_tbl <- transmute(an, Effect,
                         `Sum Sq`=.data[[cs]], `Mean Sq`=.data[[cm]],
                         NumDF=.data[[cN]], DenDF=.data[[cD]],
                         `F value`=.data[[cF]], `Pr(>F)`=.data[[cp]])
  effsize <- transmute(anova_tbl, Effect,
                       Eta2_partial = ifelse(is.finite(`F value`),
                                             (`F value`*NumDF)/((`F value`*NumDF)+DenDF), NA_real_),
                       CI_low=NA_real_, CI_high=NA_real_)
  r2 <- tryCatch(performance::r2_nakagawa(m, tolerance=tol_r2), error=function(e) NULL)
  r2_tbl <- tibble(Metric=c("Conditional R2","Marginal R2"),
                   Value=c(r2$R2_conditional %||% NA_real_, r2$R2_marginal %||% NA_real_))

  emm_g_t_obj <- emmeans(m, ~ Group | Time)
  emm_t_g_obj <- emmeans(m, ~ Time  | Group)

  list(model=m, singular=lme4::isSingular(m, tol=1e-5),
       anova=anova_tbl, effsize=effsize, r2=r2_tbl,
       emm_g_t=as.data.frame(emm_g_t_obj),
       emm_t_g=as.data.frame(emm_t_g_obj),
       post_g_t=as.data.frame(pairs(emm_g_t_obj, adjust="bonferroni")),
       post_t_g=as.data.frame(contrast(emm_t_g_obj, method=list("0m-6m"=c(1,0,-1)), adjust="bonferroni")))
}

dv_list <- intersect(c("PPD","CAL","BDAC","MOB","ANGLE","PI","GI"), unique(df_keep$DV))
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
PostHoc_Group_at_Time <- bind_rows(lapply(names(res_list), \(dv) res_list[[dv]]$post_g_t %>% mutate(DV=dv, .before=1)))
PostHoc_Time_in_Group <- bind_rows(lapply(names(res_list), \(dv) res_list[[dv]]$post_t_g %>% mutate(DV=dv, .before=1)))
EMM_Group_at_Time <- bind_rows(lapply(names(res_list), \(dv) res_list[[dv]]$emm_g_t %>% mutate(DV=dv, .before=1)))
EMM_Time_in_Group  <- bind_rows(lapply(names(res_list), \(dv) res_list[[dv]]$emm_t_g %>% mutate(DV=dv, .before=1)))
Singularity_flag   <- tibble(DV=names(res_list), Singular=vapply(res_list, function(x) x$singular, logical(1)))

xlsx_path_results <- "results_ppd_cal_bdac.xlsx"
csv_path_master   <- "MASTER_summary_table.csv"
sheets <- list(
  ANOVA_TypeIII        = make_sheet(ANOVA_all),
  EffectSize_eta2p     = make_sheet(EffectSize_all),
  R2_Nakagawa          = make_sheet(R2_all),
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
  mutate(Group = factor(Group, levels = c("BO","BT","CO")))

fmt_p   <- function(p) ifelse(is.na(p),"", ifelse(p<0.001,"<0.001", sprintf("%.3f", p)))
fmt_ms  <- function(x) if (length(x)==0 || all(is.na(x))) "" else
  sprintf("%.1f \u00B1 %.1f", mean(x,na.rm=TRUE), sd(x,na.rm=TRUE))
fmt_pct <- function(x) if (length(x)==0 || all(is.na(x))) "" else
  sprintf("%.1f", 100*mean(x>0, na.rm=TRUE))

num <- function(x) suppressWarnings(as.numeric(gsub("[^0-9.+-]", "", as.character(x))))

## ----- 군간 p (연속) -----
p_kw <- function(dat, var, time){
  sub <- dat %>% filter(Time==time) %>%
    mutate(v = num(.data[[var]])) %>% select(Group, v) %>% filter(!is.na(v))
  if (nrow(sub)<2 || dplyr::n_distinct(sub$Group)<2) return(NA_real_)
  kruskal.test(v ~ Group, data=sub)$p.value
}

## ----- 군내 p (연속; paired Wilcoxon) -----
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

## ----- Δ(0m−6m) (연속) -----
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

## ----- Δ inter-group p (연속) -----
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

## ----- 이분형 변환 (>0) -----
bin_gt0 <- function(x){
  v <- suppressWarnings(as.numeric(gsub("[^0-9.+-]", "", as.character(x))))
  ifelse(is.na(v), NA_integer_, as.integer(v > 0))
}

## ----- 군내 p (이분형; McNemar) -----
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

## ----- 군간 p (이분형; 시점별) -----
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

## ----- 블록 빌더 (연속형) -----
build_block_cont2 <- function(dat, var, label){
  grps <- levels(dat$Group)
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

## ----- 블록 빌더 (이분형 % 표기) -----
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

## ----- 임상 블록 (PPD, CAL, MOB, PI, GI) -----
blocks_clin <- list(
  build_block_cont2(raw_b6, "PPD", "PPD (mm; mean ± SD)"),
  build_block_cont2(raw_b6, "CAL", "CAL (mm; mean ± SD)"),
  if ("MOB" %in% names(raw_b6))
    build_block_cont2(raw_b6, "MOB", "Mobility (mean ± SD)"),
  if ("PI" %in% names(raw_b6))
    build_block_pct2(raw_b6, "PI", "PI (%)"),
  if ("GI" %in% names(raw_b6))
    build_block_pct2(raw_b6, "GI", "GI (%)")
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

## ----- 방사선 블록 (BDAC, ANGLE) -----
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
    header_rad,        # 스샷에 보이는 Radiographical measurements 행
    TABLE_rad_core
  )
} else {
  TABLE_base_6m <- TABLE_clin
}
## ===== 4) ΔCAL 회귀 (별도 시트) =====
# ID별 WALLS 고정값
walls_by_id <- raw %>% mutate(WALLS=num(WALLS)) %>% group_by(ID) %>%
  summarise(WALLS = na.omit(WALLS)[1], .groups="drop")
# wide 만들고 Δ 생성
wide0 <- raw %>%
  select(ID, Time, PPD, CAL, MOB, ANGLE) %>%
  mutate(across(c(PPD,CAL,MOB,ANGLE), num)) %>%
  pivot_wider(names_from=Time, values_from=c(PPD,CAL,MOB,ANGLE),
              values_fn=list(PPD=~mean(.,na.rm=TRUE), CAL=~mean(.,na.rm=TRUE),
                             MOB=~mean(.,na.rm=TRUE), ANGLE=~mean(.,na.rm=TRUE))) %>%
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

## ===== 5) 엑셀 작성 (빈칸 공백으로) =====
xlsx_path <- "table_baseline_6m.xlsx"
wb <- openxlsx::createWorkbook()
addWorksheet(wb, "TABLE_base_6m")
writeData(wb, "TABLE_base_6m", TABLE_base_6m, startRow=1, startCol=1, withFilter=FALSE, na.string="")
setColWidths(wb, "TABLE_base_6m", cols=1:6, widths=c(30,8,18,18,22,26))
sty_hdr <- createStyle(textDecoration="bold", halign="center", valign="center")
addStyle(wb, "TABLE_base_6m", sty_hdr, rows=1, cols=1:6, gridExpand=TRUE)
lab_rows <- which(TABLE_base_6m$`Clinical measurements`!="" & TABLE_base_6m$`Clinical measurements`!="p inter-group")+1
if (length(lab_rows)) addStyle(wb, "TABLE_base_6m", createStyle(textDecoration="bold"), rows=lab_rows, cols=1, gridExpand=TRUE)
conditionalFormatting(wb, "TABLE_base_6m", cols=6, rows=2:(nrow(TABLE_base_6m)+1),
  type="expression", rule='OR(LEFT($F2,1)="<", VALUE($F2)<0.05)', style=createStyle(bgFill="#C6EFCE"))
rows_pin <- which(TABLE_base_6m$`Clinical measurements`=="p inter-group")+1
if (length(rows_pin)){
  conditionalFormatting(wb, "TABLE_base_6m", cols=3:5, rows=rows_pin,
    type="expression",
    rule='OR(LEFT(CELL("contents",INDIRECT(ADDRESS(ROW(),COLUMN()))),1)="<", VALUE(INDIRECT(ADDRESS(ROW(),COLUMN())))<0.05)',
    style=createStyle(bgFill="#C6EFCE"))
}

# 회귀 시트
addWorksheet(wb, "Multiple_regression")
writeData(wb, "Multiple_regression", REG_table, withFilter=FALSE, na.string="")
setColWidths(wb, "Multiple_regression", cols=1:4, widths=c(36,10,22,10))
addStyle(wb, "Multiple_regression", createStyle(textDecoration="bold", halign="center"), rows=1, cols=1:4, gridExpand=TRUE)

saveWorkbook(wb, xlsx_path, overwrite=TRUE)

## ===== 6) LMM Time×Group 표 (기능 유지) =====
label_for <- function(dv) switch(dv,
  "CAL"="CAL (mm)", "PPD"="PPD (mm)", "BDAC"="Linear defect fill (BD-AC) (mm)",
  "ANGLE"="Defect angle modification (°)", "MOB"="Mobility", dv)
fmt_ci  <- function(est,lcl,ucl,p) sprintf("%.2f [%.2f, %.2f] (p=%s)", est,lcl,ucl,fmt_p(p))

build_timeXgroup_block <- function(dv, raw_dat=raw_b6, res_map=res_list){
  res <- res_map[[dv]]; if (is.null(res)) return(NULL)
  emm_t_g <- emmeans::emmeans(res$model, ~ Time | Group)
  ct <- as.data.frame(summary(emmeans::contrast(emm_t_g, method=list("0m-6m"=c(1,0,-1)), adjust="bonferroni"), infer=c(TRUE,TRUE)))
  an  <- res$anova; eff <- res$effsize
  eff_name <- if ("Group:Time" %in% an$Effect) "Group:Time" else "Time:Group"
  F_int <- an$`F value`[an$Effect==eff_name][1]; p_int <- an$`Pr(>F)`[an$Effect==eff_name][1]
  eta_p <- eff$Eta2_partial[eff$Effect==eff_name][1]

  grps <- levels(raw_dat$Group)
  rows <- lapply(seq_along(grps), function(i){
    g <- grps[i]; r <- ct[ct$Group==g, ]
    cell <- if (nrow(r)) fmt_ci(r$estimate, r$lower.CL, r$upper.CL, r$p.value) else ""
    tibble(`Clinical measurements`=if (i==1) label_for(dv) else "", Group=g,
           `Δ0–6m`=cell,
           `F (Time×Group)`= if (i==1) sprintf("%.2f",F_int) else "",
           `p (Time×Group)`= if (i==1) fmt_p(p_int) else "",
           `η²p (Time×Group)`= if (i==1) ifelse(is.na(eta_p),"",sprintf("%.2f",eta_p)) else "")
  }) %>% bind_rows()
  rows
}
dv_for_table <- c("PPD","CAL","BDAC","MOB","ANGLE"); dv_for_table <- dv_for_table[dv_for_table %in% names(res_list)]
blocks <- lapply(dv_for_table, build_timeXgroup_block)
blocks <- Filter(Negate(is.null), blocks)
if (length(blocks)){
  spacer2 <- tibble(`Clinical measurements`="", Group="", `Δ0–6m`="", `F (Time×Group)`="", `p (Time×Group)`="", `η²p (Time×Group)`="")
  TABLE_LMM_TimeXGroup <- purrr::imap_dfr(blocks, \(x,i) bind_rows(x, if (i<length(blocks)) spacer2 else NULL))
  wb2 <- if (file.exists(xlsx_path)) loadWorkbook(xlsx_path) else createWorkbook()
  if ("LMM_Time×Group" %in% sheets(wb2)) removeWorksheet(wb2, "LMM_Time×Group")
  addWorksheet(wb2, "LMM_Time×Group")
  writeData(wb2, "LMM_Time×Group", TABLE_LMM_TimeXGroup, withFilter=FALSE, na.string="")
  setColWidths(wb2, "LMM_Time×Group", cols=1:6, widths=c(30,8,32,16,16,16))
  addStyle(wb2, "LMM_Time×Group", createStyle(textDecoration="bold", halign="center"), rows=1, cols=1:6, gridExpand=TRUE)
  lab_rows2 <- which(TABLE_LMM_TimeXGroup$`Clinical measurements`!="")+1
  if (length(lab_rows2)) addStyle(wb2, "LMM_Time×Group", createStyle(textDecoration="bold"), rows=lab_rows2, cols=1, gridExpand=TRUE)
  conditionalFormatting(wb2, "LMM_Time×Group", cols=5, rows=2:(nrow(TABLE_LMM_TimeXGroup)+1),
                        type="expression", rule='OR(LEFT($E2,1)="<", VALUE($E2)<0.05)', style=createStyle(bgFill="#C6EFCE"))
  saveWorkbook(wb2, xlsx_path, overwrite=TRUE)
}
## ===== 7) Pocket closure & COM rates (add new sheet) =====
# helper: N(%) string
fmt_npct <- function(x){
  n1 <- sum(x==1L, na.rm=TRUE); d <- sum(!is.na(x))
  if (d==0) "" else sprintf("%d(%.0f%%)", n1, 100*n1/d)
}
# helper: build one block like the screenshot
build_rate_block <- function(dat, var, header_label){
  sub <- dat %>% filter(Time=="6m") %>% mutate(b = bin_gt0(.data[[var]]))
  # Total
  total <- fmt_npct(sub$b)
  # Group-wise
  grps <- c("BO","BT","CO")
  g_rows <- lapply(grps, function(g){
    tibble(A = "", B = g,
           C = fmt_npct(sub %>% filter(Group==g) %>% pull(b)))
  }) %>% bind_rows()
  # Fisher exact p (inter-group)
  tab <- table(sub$Group, sub$b)
  pval <- if (nrow(tab)>=2 && ncol(tab)>=2) fisher.test(tab)$p.value else NA_real_
  p_str <- ifelse(is.na(pval),"", ifelse(pval<0.001,"<0.001", sprintf("%.2f", pval)))
  # Assemble
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

# write as the last sheet of table_baseline_6m.xlsx
wb_pc <- if (file.exists(xlsx_path)) openxlsx::loadWorkbook(xlsx_path) else openxlsx::createWorkbook()
if ("PC_COM_summary" %in% openxlsx::sheets(wb_pc)) openxlsx::removeWorksheet(wb_pc, "PC_COM_summary")
openxlsx::addWorksheet(wb_pc, "PC_COM_summary")
openxlsx::writeData(wb_pc, "PC_COM_summary", TABLE_PC_COM, startRow=1, startCol=1, withFilter=FALSE, na.string="")
openxlsx::setColWidths(wb_pc, "PC_COM_summary", cols=1:3, widths=c(48,18,14))
# styles: bold headers in col A; center % column; bold section labels
hdr_rows <- which(TABLE_PC_COM$A != "") + 1L
if (length(hdr_rows))
  openxlsx::addStyle(wb_pc, "PC_COM_summary", createStyle(textDecoration="bold"), rows=hdr_rows, cols=1, gridExpand=TRUE)
openxlsx::addStyle(wb_pc, "PC_COM_summary", createStyle(halign="center"), rows=2:(nrow(TABLE_PC_COM)+1), cols=3, gridExpand=TRUE)

openxlsx::saveWorkbook(wb_pc, xlsx_path, overwrite=TRUE)



############### Post-hoc ###############





# ==== 정규성 & 등분산성 평가 ====
suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(car); library(rstatix); library(openxlsx)
})

num <- function(x) suppressWarnings(as.numeric(gsub("[^0-9.+-]", "", as.character(x))))

vars_check <- c("BDAC","PPD","CAL","MOB","ANGLE","PI","GI")
vars_check <- intersect(vars_check, names(raw_b6))

norm_results <- list()
var_results  <- list()

for(v in vars_check){
  dat <- raw_b6 %>%
    filter(Time %in% c("0m","6m")) %>%
    mutate(Group=factor(Group, levels=c("BO","BT","CO")),
           Time = factor(Time, levels=c("0m","6m")),
           val  = num(.data[[v]]))
  
  for(t in c("0m","6m")){
    sub <- dat %>% filter(Time==t & is.finite(val))
    # ---- Shapiro–Wilk by group ----
    sw <- sub %>%
      group_by(Group) %>%
      summarise(
        n = n(),
        p_normality = ifelse(n >= 3, shapiro.test(val)$p.value, NA_real_)
      ) %>%
      mutate(Time=t, Variable=v)
    norm_results[[paste(v,t,sep="_")]] <- sw
    
    # ---- Levene’s test (equal variance) ----
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

# ---- 엑셀 저장 ----
xlsx_results <- xlsx_path_results %||% "results_ppd_cal_bdac.xlsx"
wb <- if (file.exists(xlsx_results)) loadWorkbook(xlsx_results) else createWorkbook()

if ("Normality_Homoscedasticity" %in% sheets(wb)) removeWorksheet(wb, "Normality_Homoscedasticity")
addWorksheet(wb, "Normality_Homoscedasticity")

writeData(wb, "Normality_Homoscedasticity", normality_tbl, startRow=1, startCol=1, withFilter=TRUE)
writeData(wb, "Normality_Homoscedasticity", var_tbl, startRow=nrow(normality_tbl)+3, startCol=1, withFilter=TRUE)

setColWidths(wb, "Normality_Homoscedasticity", cols=1:6, widths=c(12,8,6,12,12,6))
addStyle(wb, "Normality_Homoscedasticity", createStyle(textDecoration="bold", halign="center"),
          rows=c(1, nrow(normality_tbl)+3), cols=1:6, gridExpand=TRUE)

saveWorkbook(wb, xlsx_results, overwrite=TRUE)






# ==== Permutation ANCOVA at 6m (nonparametric) ====
suppressPackageStartupMessages({
  library(coin); library(dplyr); library(tidyr); library(purrr); library(openxlsx)
})

# 공용 헬퍼: 6m 데이터 + baseline merge
make_dat6 <- function(var){
  b <- raw_b6 %>% filter(Time=="0m") %>% transmute(ID, base = num(.data[[var]]))
  raw_b6 %>%
    filter(Time=="6m") %>%
    transmute(ID, Group=factor(Group, levels=c("BO","BT","CO")), val = num(.data[[var]])) %>%
    left_join(b, by="ID") %>%
    filter(is.finite(val), is.finite(base))
}

vars_np <- c("BDAC","PPD","CAL","ANGLE")
omni_list <- list(); pair_list <- list()

for(v in vars_np){
  d6 <- make_dat6(v)
  if(nrow(d6) < 6 || dplyr::n_distinct(d6$Group) < 2){ next }

  # Omnibus permutation ANCOVA: val ~ Group + base
  it <- coin::independence_test(val ~ Group + base, data=d6,
                                distribution = coin::approximate(B=9999))
  p_all <- coin::pvalue(it)
  stat  <- coin::statistic(it, type="standardized")

  omni_list[[v]] <- tibble(
    Variable = v, Test = "Permutation ANCOVA (coin)",
    N = nrow(d6), Groups = n_distinct(d6$Group),
    Statistic = as.numeric(stat), p = as.numeric(p_all)
  )

  # Pairwise permutation ANCOVA (Group 쌍별, baseline 공변량 유지) + Holm 보정
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

# ---- 엑셀 저장 (results 파일에 추가/갱신) ----
xlsx_results <- xlsx_path_results %||% "results_ppd_cal_bdac.xlsx"
wb <- if (file.exists(xlsx_results)) openxlsx::loadWorkbook(xlsx_results) else openxlsx::createWorkbook()

if ("ANCOVA_perm_6m" %in% openxlsx::sheets(wb)) openxlsx::removeWorksheet(wb, "ANCOVA_perm_6m")
openxlsx::addWorksheet(wb, "ANCOVA_perm_6m")
openxlsx::writeData(wb, "ANCOVA_perm_6m", ANCOVA_perm_omni, startRow=1, startCol=1, withFilter=TRUE)
openxlsx::writeData(wb, "ANCOVA_perm_6m", ANCOVA_perm_pairs, startRow=nrow(ANCOVA_perm_omni)+3, startCol=1, withFilter=TRUE)
openxlsx::setColWidths(wb, "ANCOVA_perm_6m", cols=1:7, widths=c(12,24,6,7,10,8,6))
openxlsx::addStyle(wb, "ANCOVA_perm_6m",
                   openxlsx::createStyle(textDecoration="bold", halign="center"),
                   rows=c(1, nrow(ANCOVA_perm_omni)+3), cols=1:7, gridExpand=TRUE)

openxlsx::saveWorkbook(wb, xlsx_results, overwrite=TRUE)







## ===== Dunn post hoc for baseline_6m (Kruskal–Wallis) =====
to_install <- c("FSA","dunn.test","rstatix")
ip <- to_install[!to_install %in% installed.packages()[,"Package"]]
if (length(ip)) install.packages(ip, dependencies = TRUE)
suppressPackageStartupMessages({ library(rstatix) })  # 깔끔한 출력

# 공용: 시점별 Dunn, Δ-Dunn
dunn_time <- function(dat, var, time, p.adjust.method="bonferroni"){
  sub <- dat %>% filter(Time==time) %>% 
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
  sub <- dat %>% mutate(v=num(.data[[var]])) %>%
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

# 대상 변수 선택(연속형)
vars_for_dunn <- c("BDAC","PPD","CAL","MOB","ANGLE")
vars_for_dunn <- intersect(vars_for_dunn, names(raw_b6))

# 변수별 Dunn 결과 결합
DUNN_all <- purrr::map_dfr(vars_for_dunn, function(v){
  bind_rows(
    dunn_time(raw_b6, v, "0m"),
    dunn_time(raw_b6, v, "6m"),
    dunn_delta(raw_b6, v)
  ) %>% mutate(Variable = v, .before=1)
})

# 엑셀에 쓰기(기존 파일 이어쓰기)
wb_dunn <- if (file.exists(xlsx_path)) openxlsx::loadWorkbook(xlsx_path) else openxlsx::createWorkbook()
if ("PostHoc_Dunn" %in% openxlsx::sheets(wb_dunn)) openxlsx::removeWorksheet(wb_dunn, "PostHoc_Dunn")
openxlsx::addWorksheet(wb_dunn, "PostHoc_Dunn")
openxlsx::writeData(wb_dunn, "PostHoc_Dunn", DUNN_all, withFilter=TRUE)
openxlsx::setColWidths(wb_dunn, "PostHoc_Dunn", cols=1:6, widths=c(14,12,12,10,10,8))
openxlsx::addStyle(wb_dunn, "PostHoc_Dunn",
                   openxlsx::createStyle(textDecoration="bold", halign="center"),
                   rows=1, cols=1:6, gridExpand=TRUE)
openxlsx::saveWorkbook(wb_dunn, xlsx_path, overwrite=TRUE)






# ---------------------- GRAPH ----------------------







# =======================
# 0선 기준 상·하 그래프 자동 생성 (CAL, PPD, BDAC)
# =======================

library(dplyr)
library(tidyr)
library(ggplot2)
library(grid)

if (!exists("raw_b6")) stop("raw_b6 데이터 필요")
num <- function(x) suppressWarnings(as.numeric(gsub("[^0-9.+-]", "", as.character(x))))

fig_dir <- "fig_out"
if (!dir.exists(fig_dir)) dir.create(fig_dir)

p2star <- function(p){
  if (is.na(p)) return("ns")
  if (p < 0.001) return("***")
  if (p < 0.01)  return("**")
  if (p < 0.05)  return("*")
  "ns"
}

add_top_bracket <- function(p, x0, x1, y, label, col="black"){
  off <- 0.02 * (abs(y) + 1)
  p +
    annotate("segment", x=x0, xend=x0, y=y,     yend=y-off, colour=col, linewidth=0.8) +
    annotate("segment", x=x0, xend=x1, y=y-off, yend=y-off, colour=col, linewidth=0.8) +
    annotate("segment", x=x1, xend=x1, y=y-off, yend=y,     colour=col, linewidth=0.8) +
    annotate("label",   x=(x0+x1)/2, y=y, label=label,
             fill="white", colour=col, label.size=0.4,
             vjust=-0.6, label.padding=grid::unit(0.1,"lines"))
}


# =======================
# 밝은·어두운 테마로 각각 저장
# =======================

# 라이트/다크 두 버전 동시 저장

savefig <- function(p, name, w=8, h=5, dpi=300){
  fig_dir <- "fig_out"
  if (!dir.exists(fig_dir)) dir.create(fig_dir)

  # =======================
  # 밝은 테마 (Light)
  # =======================
  p_light <- p +
    theme_classic(base_size=12) +
    theme(
      plot.background  = element_rect(fill="white", color="black", linewidth=0.6),
      panel.background = element_rect(fill="white", color=NA),
      text        = element_text(color="black"),
      axis.text   = element_text(color="black"),
      axis.title  = element_text(color="black"),
      plot.title  = element_text(color="black"),
      legend.text = element_text(color="black"),
      legend.background = element_rect(fill="white", color="black"),
      axis.line   = element_line(color="black", linewidth=0.6),
      axis.ticks  = element_line(color="black", linewidth=0.6)
    )

  ggsave(file.path(fig_dir, paste0(name, "_light.png")), p_light, width=w, height=h, dpi=dpi)
  ggsave(file.path(fig_dir, paste0(name, "_light.tiff")), p_light, width=w, height=h, dpi=dpi, compression="lzw")

  # =======================
  # 어두운 테마 (Dark)
  # =======================
  p_dark <- p +
    theme_classic(base_size=12) +
    theme(
      plot.background  = element_rect(fill="black", color=NA), 
      panel.background = element_rect(fill="black", color=NA),
      text        = element_text(color="white"),
      axis.text   = element_text(color="white"),
      axis.title  = element_text(color="white"),
      plot.title  = element_text(color="white"),
      legend.text = element_text(color="white"),
      legend.background = element_rect(fill="black", color="white", linewidth=0.4),
      axis.line   = element_line(color="white", linewidth=0.6),
      axis.ticks  = element_line(color="white", linewidth=0.6),
      panel.grid  = element_blank()
    )

  # 모든 layer 색상 반전 (막대 윤곽, 에러바 등)
  p_dark$layers <- lapply(p_dark$layers, function(layer) {
    if (inherits(layer$geom, "GeomCol") || inherits(layer$geom, "GeomBar"))
      layer$aes_params$colour <- "white"
    if (inherits(layer$geom, "GeomErrorbar"))
      layer$aes_params$colour <- "white"
    if (inherits(layer$geom, "GeomHline"))
      layer$aes_params$colour <- "white"
    layer
  })

  ggsave(file.path(fig_dir, paste0(name, "_dark.png")), p_dark, width=w, height=h, dpi=dpi)
  ggsave(file.path(fig_dir, paste0(name, "_dark.tiff")), p_dark, width=w, height=h, dpi=dpi, compression="lzw")

  message("Saved both themes with white outlines and zero line: ", name)
}

# ==== 흑백 박스 테마 ====
theme_box_bw <- function(base_size = 12){
  theme_bw(base_size = base_size) +
    theme(
      plot.background  = element_rect(fill = "white", color = "black", linewidth = 0.8),
      panel.border     = element_rect(fill = NA, color = "black", linewidth = 0.8),
      panel.grid.major = element_blank(),   # ← 삭제
      panel.grid.minor = element_blank(),   # ← 삭제
      axis.line        = element_blank(),
      axis.ticks       = element_line(color = "black", linewidth = 0.6),
      axis.text        = element_text(color = "black"),
      axis.title       = element_text(color = "black"),
      legend.background= element_rect(fill = "white", color = "black", linewidth = 0.6),
      legend.key       = element_rect(fill = "white", color = "black", linewidth = 0.4),
      legend.title     = element_text(color = "black"),
      plot.title       = element_text(face="bold")
    )
}

# ==== 흑백 파일 저장(한 버전만 저장) ====
savefig_bw <- function(p, name, w=8, h=5, dpi=300){
  fig_dir <- "fig_out"
  if (!dir.exists(fig_dir)) dir.create(fig_dir)
  ggsave(file.path(fig_dir, paste0(name, "_bw.png")),  p, width=w, height=h, dpi=dpi)
  ggsave(file.path(fig_dir, paste0(name, "_bw.tiff")), p, width=w, height=h, dpi=dpi, compression="lzw")
}


# ===== 함수 정의 =====
plot_delta_only <- function(var, label){
  dat <- raw_b6 %>% mutate(v = num(.data[[var]]))

  delta <- dat %>%
    select(ID, Group, Time, v) %>%
    tidyr::pivot_wider(names_from=Time, values_from=v) %>%
    mutate(delta = `0m` - `6m`) %>%
    group_by(Group) %>%
    summarise(n = sum(!is.na(delta)),
              mean = mean(delta, na.rm=TRUE),
              sd   = sd(delta,   na.rm=TRUE),
              se   = sd/sqrt(pmax(n,1)),
              l = mean - se, u = mean + se, .groups="drop") %>%
    mutate(Group = factor(Group, levels=c("BO","BT","CO")))

  yr   <- range(c(delta$l, delta$u, 0), na.rm=TRUE)
  pad  <- diff(yr) * 0.25
  ytop <- yr[2] + pad

  p <- ggplot(delta, aes(x=Group, y=mean, fill=Group)) +
    geom_hline(yintercept=0, linewidth=0.6) +
    geom_col(width=0.55, color="black") +
    geom_errorbar(aes(ymin=l, ymax=u), width=.15, color="black") +
    scale_fill_grey(start = 0.7, end = 0.3, guide = "none") +
    labs(x=NULL, y=label, title=paste0(label, " Δ(0m−6m)")) +
    theme_box_bw() +
    expand_limits(y = ytop)

  savefig_bw(p, paste0("fig_", var, "_delta_only"))
}

# ===== 실행 =====

# ---- Δ만 ----
plot_delta_only("CAL",  "CAL (mm)")
plot_delta_only("PPD",  "PPD (mm)")
plot_delta_only("BDAC", "Linear defect fill (mm)")

# =======================
# 0m·6m 양수 막대 (MOB, ANGLE, PI, GI)
# =======================

# 원하는 군만 브래킷을 그리도록: show_groups 인자 추가
plot_two_time_bar <- function(var, label, force_percent = NA, show_groups = c("BO","BT","CO")){
  dat <- raw_b6 %>% mutate(v = num(.data[[var]]))

  if (is.na(force_percent)) {
    rng <- range(dat$v, na.rm = TRUE)
    is_pct <- is.finite(rng[1]) && is.finite(rng[2]) && rng[2] <= 1.0001
  } else is_pct <- isTRUE(force_percent)

  scale_val <- if (is_pct) 100 else 1
  ylab <- if (is_pct) paste0(label, " (%)") else label

  sum_abs <- dat %>%
    group_by(Group, Time) %>%
    summarise(n = sum(!is.na(v)),
              mean = mean(v, na.rm=TRUE) * scale_val,
              sd   = sd(v,   na.rm=TRUE) * scale_val,
              se   = (sd/scale_val)/sqrt(pmax(n,1)) * scale_val,
              l = mean - se, u = mean + se, .groups="drop") %>%
    mutate(Panel = factor(Time, levels=c("0m","6m")),
           Group = factor(Group, levels=c("BO","BT","CO")))

  yr   <- range(c(sum_abs$l, sum_abs$u, 0), na.rm=TRUE)
  pad  <- diff(yr) * 0.25
  ytop <- yr[2] + pad

  p <- ggplot(sum_abs, aes(x=Group, y=mean, fill=Panel)) +
    geom_hline(yintercept = 0, linewidth = 0.6) +
    geom_col(position=position_dodge(width=0.7), width=0.55, color="black") +
    geom_errorbar(aes(ymin=l, ymax=u),
                  position=position_dodge(width=0.7), width=.15, color="black") +
    scale_fill_grey(name = "Time point", start = 0.85, end = 0.25) +
    labs(x=NULL, y=ylab, title=label) +
    theme_box_bw() +
    expand_limits(y = ytop)

  savefig_bw(p, paste0("fig_", var, "_twolevel_pos"))
}

# ===== 실행 =====
# ---- 절대값(0m·6m) ----
plot_two_time_bar("CAL",  "CAL (mm)")
plot_two_time_bar("PPD",  "PPD (mm)")
plot_two_time_bar("BDAC", "Linear defect fill (mm)")

plot_two_time_bar("MOB",   "Mobility")
plot_two_time_bar("ANGLE", "Defect angle (°)")
plot_two_time_bar("PI",    "PI",  force_percent = NA)  # 자동 % 판별
plot_two_time_bar("GI",    "GI",  force_percent = NA)  # 자동 % 판별




############### LMM 궤적 그래프 (CAL, PPD) ###############

pkgs <- c("ggplot2","patchwork")
to_install <- pkgs[!pkgs %in% installed.packages()[,"Package"]]
if (length(to_install)) install.packages(to_install, dependencies = TRUE)
suppressPackageStartupMessages({
  library(ggplot2); library(patchwork)
})

plot_lmm_trajectory <- function(dv_sel,
                                dat   = EMM_Time_in_Group,
                                ylab  = NULL,
                                y_min = NULL,
                                y_max = NULL){

  d <- dat %>%
    dplyr::filter(DV == dv_sel) %>%
    mutate(
      emmean   = as.numeric(emmean),
      lower.CL = as.numeric(lower.CL),
      upper.CL = as.numeric(upper.CL)
    )

  if (nrow(d) == 0) {
    warning("선택한 DV에 해당하는 EMM이 없습니다: ", dv_sel)
    return(invisible(NULL))
  }

  if (is.null(ylab)) {
    ylab <- switch(dv_sel,
                   "CAL" = "CAL (mm)",
                   "PPD" = "PPD (mm)",
                   dv_sel)
  }

  # 공통 y축 범위 자동/고정
  yr <- range(c(d$lower.CL, d$upper.CL), na.rm = TRUE)
  if (is.null(y_min)) y_min <- floor(yr[1])
  if (is.null(y_max)) y_max <- ceiling(yr[2])

  ggplot(d, aes(x = Time, y = emmean, group = Group)) +
    geom_line(linewidth = 0.8, colour = "black") +
    geom_point(aes(shape = Group),
               size   = 3,
               stroke = 1.0,
               colour = "white",
               fill   = "black") +
    scale_shape_manual(values = c(21, 22, 24)) +
    scale_y_continuous(
      limits = c(y_min, y_max),
      breaks = seq(y_min, y_max, by = 1),
      expand = c(0.02, 0)
    ) +
    scale_x_discrete(expand = c(0.2, 0)) +
    labs(x = "Time", y = ylab) +
    theme_bw(base_size = 11) +
    theme(
      plot.title        = element_blank(),
      plot.tag          = element_blank(),
      plot.margin       = margin(t = 6, r = 10, b = 6, l = 10),
      panel.grid.minor  = element_blank(),
      panel.grid.major.x= element_blank(),
      axis.title.x      = element_text(margin = margin(t = 6)),
      axis.title.y      = element_text(margin = margin(r = 6)),
      legend.position   = "right",
      legend.title      = element_blank(),
      legend.key.width  = unit(1.4, "lines")
    )
}

# 개별 그림
p_cal <- plot_lmm_trajectory("CAL")
p_ppd <- plot_lmm_trajectory("PPD")

ggsave("CAL_LMM_trajectory.png", p_cal, width = 4, height = 7, dpi = 600)
ggsave("PPD_LMM_trajectory.png", p_ppd, width = 4, height = 7, dpi = 600)

# 두 개를 한 Figure로
p_multi <- p_ppd + p_cal +
  patchwork::plot_layout(guides = "collect") &
  theme(legend.position = "right")

ggsave("CAL_PPD_LMM_multiview.png", p_multi, width = 8, height = 7, dpi = 600)



