# =========================================================
# RM_ready.csv 로부터 분석 + 표 생성 (모두 한 스크립트)
# - LMM 결과 엑셀(results_ppd_cal_bdac.xlsx)
# - 0m vs 6m 요약표(table_baseline_6m.xlsx)
# - Radiographic outcome 시트 추가
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
BOM  <- "\uFEFF"                # UTF-8 BOM
NBSP <- "\u00A0"                # non-breaking space

fix_names <- function(nm){
  nm <- enc2utf8(nm)
  nm <- gsub(BOM,  "", nm, fixed = TRUE)
  nm <- gsub("\\x{00A0}", "", nm, perl = TRUE)  # NBSP 제거
  trimws(nm)
}

to_bin01 <- function(x){
  s <- enc2utf8(as.character(x))
  s <- gsub(BOM,  "", s, fixed = TRUE)
  s <- gsub("\\x{00A0}", "", s, perl = TRUE)    # NBSP 제거
  s <- gsub("[[:space:]\r\n\t]", "", s, perl = TRUE)
  s[s == "+"] <- "1"; s[s == "-"] <- "0"
  s <- gsub("[^0-9]", "", s, perl = TRUE)
  s[s == ""] <- NA
  v <- suppressWarnings(as.integer(s))
  v[!(v %in% c(0L,1L))] <- NA_integer_
  v
}

make_sheet <- function(x) {
  if (is.null(x)) return(data.frame(note="(no data)"))
  if (is.data.frame(x)) { if (ncol(x)==0) return(data.frame(note="(no data)")); return(x) }
  as.data.frame(x)
}
.safe <- function(expr, fallback) tryCatch(expr, error=function(e){ message("WARN: ", conditionMessage(e)); fallback })

## ===== 1) 파일 읽기 & 형식 감지 =====
stopifnot(file.exists("RM_ready.csv"))
raw0 <- read.csv("RM_ready.csv", na.strings=c("","NA"), check.names = FALSE,
                 fileEncoding = "UTF-8")
names(raw0) <- fix_names(names(raw0))

message("WD: ", getwd())
message("Columns: ", paste(names(raw0), collapse=", "))

has_wide <- all(c("ID","Group","Time","PPD","CAL","BDAC") %in% names(raw0))
has_long <- all(c("ID","DV","Time","Value") %in% names(raw0))

if (!has_wide && !has_long) {
  stop("RM_ready.csv 형식을 인식할 수 없습니다. (wide: ID,Group,Time,PPD... / long: ID,DV,Time,Value)")
}

if (has_wide) {
  raw <- raw0
} else {
  # LONG → WIDE 복원
  raw <- raw0 %>%
    mutate(Group = factor(str_extract(ID, "^[A-Za-z]{2}"))) %>%
    tidyr::pivot_wider(names_from = DV, values_from = Value)
}

# 새 변수(없으면 생성)
for (nm in c("PPD","CAL","BDAC","MOB","ANGLE","COM","PC","PI","GI","WALLS"))
  if (!nm %in% names(raw)) raw[[nm]] <- NA

# 기본 열 정리
raw$ID    <- factor(raw$ID)
raw$Group <- factor(raw$Group %||% sub("^([A-Za-z]{2}).*$","\\1", as.character(raw$ID)))
raw$Time  <- enc2utf8(as.character(raw$Time))
raw$Time  <- gsub("\\s+", "", raw$Time)
raw$Time  <- tolower(raw$Time)
raw$Time  <- gsub("^0m$|^0month$|^0months$","0m", raw$Time)
raw$Time  <- gsub("^3m$|^3month$|^3months$","3m", raw$Time)
raw$Time  <- gsub("^6m$|^6month$|^6months$","6m", raw$Time)
raw$Time  <- factor(raw$Time, levels=c("0m","3m","6m"))

## ===== 2) LMM용 LONG 구성 =====
cont_cols <- intersect(c("PPD","CAL","BDAC","MOB","ANGLE","PI","GI"), names(raw))
df_keep <- raw %>%
  pivot_longer(all_of(cont_cols), names_to="DV", values_to="Value") %>%
  filter(!is.na(Value))

# ---- LMM 적합 & 요약 ----
fit_lmm <- function(dv_name, dat=df_keep, tol_r2=1e-8){
  dat_dv <- dplyr::filter(dat, DV==dv_name)
  if (n_distinct(dat_dv$Time) < 2 || nrow(dat_dv) < 5) return(NULL)
  .std_names <- function(d){
    nm <- names(d); nm <- gsub("[[:space:]]+","_",nm); nm <- gsub("\\.","_",nm)
    nm <- gsub("\\(|\\)|>|<|=","",nm); names(d) <- nm; d
  }
  .pick_col <- function(nm, pattern){ hit <- grep(pattern, nm, value=TRUE, ignore.case=TRUE); if (length(hit)) hit[1] else NA }
  m <- .safe(lmer(Value ~ Group * Time + (1|ID), data=dat_dv, REML=TRUE,
                  control=lme4::lmerControl(check.conv.singular="message")), NULL)
  if (is.null(m)) return(NULL)

  an_raw <- .safe(anova(m, type=3) |> as.data.frame() |> tibble::rownames_to_column("Effect"), NULL)
  if (is.null(an_raw)) return(NULL)
  an_cln <- .std_names(an_raw); nm <- names(an_cln)
  cs <- .pick_col(nm,"^Sum_?Sq$"); cm <- .pick_col(nm,"^Mean_?Sq$")
  cN <- .pick_col(nm,"^NumDF$");   cD <- .pick_col(nm,"^DenDF$")
  cF <- .pick_col(nm,"^F_?value$|^F$"); cp <- .pick_col(nm,"^Pr|^P_?value$|^p_?value$")
  if (any(is.na(c(cs,cm,cN,cD,cF,cp)))) return(NULL)

  anova_tbl <- transmute(an_cln, Effect,
                         `Sum Sq`=.data[[cs]], `Mean Sq`=.data[[cm]],
                         NumDF=.data[[cN]], DenDF=.data[[cD]],
                         `F value`=.data[[cF]], `Pr(>F)`=.data[[cp]])
  effsize <- transmute(anova_tbl, Effect,
                       Eta2_partial = ifelse(is.finite(`F value`),
                         (`F value`*NumDF)/((`F value`*NumDF)+DenDF), NA_real_),
                       CI_low=NA_real_, CI_high=NA_real_)
  r2 <- tryCatch(performance::r2_nakagawa(m, tolerance=tol_r2), error=function(e) NULL)
  r2_tbl <- tibble(Metric=c("Conditional R2","Marginal R2"),
                   Value=c(r2$R2_conditional %||% NA_real_,
                           r2$R2_marginal    %||% NA_real_))

  emm_g_t_obj <- emmeans(m, ~ Group | Time)
  emm_t_g_obj <- emmeans(m, ~ Time  | Group)

  emm_g_t <- as.data.frame(emm_g_t_obj)
  emm_t_g <- as.data.frame(emm_t_g_obj)

  post_g_t <- as.data.frame(pairs(emm_g_t_obj, adjust="bonferroni"))
  post_t_g <- as.data.frame(
    contrast(emm_t_g_obj, method = list("0m-6m" = c(1, 0, -1)), adjust = "bonferroni")
  )

  list(model=m, singular=lme4::isSingular(m, tol=1e-5),
       anova=anova_tbl, effsize=effsize, r2=r2_tbl,
       emm_g_t=emm_g_t, emm_t_g=emm_t_g,
       post_g_t=post_g_t, post_t_g=post_t_g)
}

dv_list <- intersect(c("PPD","CAL","BDAC","MOB","ANGLE","PI","GI"), unique(df_keep$DV))
res_list <- setNames(lapply(dv_list, fit_lmm), dv_list)
res_list <- res_list[!sapply(res_list, is.null)]  # 실패한 DV 제거

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
EMM_Time_in_Group <- bind_rows(lapply(names(res_list), \(dv) res_list[[dv]]$emm_t_g %>% mutate(DV=dv, .before=1)))
Singularity_flag <- tibble(DV=names(res_list),
                           Singular=vapply(res_list, function(x) x$singular, logical(1)))

ci_str <- function(est,lcl,ucl,digits=2) paste0(formatC(est,format="f",digits=digits)," [",
                                                formatC(lcl,format="f",digits=digits),", ",
                                                formatC(ucl,format="f",digits=digits),"]")
p_str <- function(p,digits=3) ifelse(is.na(p), NA, ifelse(p<0.001,"<0.001",formatC(p,format="f",digits=digits)))
effect_cell <- function(est,lcl,ucl,p) paste0(ci_str(est,lcl,ucl)," (p=",p_str(p),")")

make_master_for_dv <- function(res, dv_label){
  m <- res$model
  emm_t_g <- emmeans(m, ~ Time | Group)
  contr <- contrast(emm_t_g, method = list("0m-6m" = c(1, 0, -1)), adjust = "bonferroni")
  summ_contr <- as.data.frame(summary(contr, infer=c(TRUE,TRUE))) %>%
    mutate(DV=dv_label) %>% select(DV, Group, contrast, estimate, lower.CL, upper.CL, p.value)
  emm_g_t <- emmeans(m, ~ Group | Time)
  pairs_g_t <- pairs(emm_g_t, adjust="bonferroni")
  summ_pairs <- as.data.frame(pairs_g_t) %>% group_by(Time) %>%
    summarise(p_min_across_groups=min(p.value, na.rm=TRUE), .groups="drop") %>%
    mutate(DV=dv_label) %>% select(DV, Time, p_min_across_groups)
  an  <- res$anova; eff <- res$effsize
  time_row <- an  %>% filter(Effect=="Time")
  eta_row  <- eff %>% filter(Effect=="Time")
  time_effect <- tibble(DV=dv_label,
                        Time_F=time_row$`F value` %||% NA_real_,
                        Time_p=time_row$`Pr(>F)`  %||% NA_real_,
                        Partial_eta2_Time=eta_row$Eta2_partial %||% NA_real_)
  delta_wide <- summ_contr %>%
    mutate(cell = effect_cell(estimate, lower.CL, upper.CL, p.value)) %>%
    select(DV, Group, contrast, cell) %>%
    pivot_wider(names_from = contrast, values_from = cell) %>%
    rename(`Δ0–6m` = `0m-6m`)
  pmin_wide <- summ_pairs %>% mutate(pmin_cell=p_str(p_min_across_groups)) %>%
    select(DV, Time, pmin_cell) %>%
    pivot_wider(names_from=Time, values_from=pmin_cell, names_prefix="p(min)_: ")
  delta_wide %>% left_join(pmin_wide, by="DV") %>% left_join(time_effect, by="DV") %>%
    mutate(Time_F=round(Time_F,2), Time_p=p_str(Time_p), Partial_eta2_Time=round(Partial_eta2_Time,2)) %>%
    relocate(DV, Group)
}
master_tbl <- if (length(res_list)) purrr::map2_dfr(res_list, names(res_list), make_master_for_dv) else tibble()

## ===== 3) COM / PC 요약(6m만) =====
find_col <- function(cands, nm=names(raw)) {
  hit <- grep(paste0("(", paste(cands, collapse="|"), ")"), nm, ignore.case=TRUE, value=TRUE)
  if (length(hit)) hit[1] else NA_character_
}
COM_col <- find_col(c("^COM$","composite","COM_?6","COM\\s*\\(6m\\)"))
PC_col  <- find_col(c("^PC$","pocket.?closure","PC_?6","PC\\s*\\(6m\\)"))

if (!is.na(COM_col)) raw[[COM_col]] <- to_bin01(raw[[COM_col]])
if (!is.na(PC_col))  raw[[PC_col]]  <- to_bin01(raw[[PC_col]])

at6_only <- raw %>% filter(Time=="6m")
if (!is.na(COM_col)) at6_only[[COM_col]] <- ifelse(at6_only$Time=="6m", at6_only[[COM_col]], NA)
if (!is.na(PC_col))  at6_only[[PC_col]]  <- ifelse(at6_only$Time=="6m", at6_only[[PC_col]],  NA)

bin_summary <- function(x){
  x <- x[!is.na(x)]
  tibble(n=length(x), pos=sum(x==1), pct=ifelse(length(x)>0, round(100*sum(x==1)/length(x),1), NA))
}
make_group_summary_col <- function(dat, col){
  overall <- tibble(Group="ALL",
                    n=sum(!is.na(dat[[col]])),
                    pos=sum(dat[[col]]==1, na.rm=TRUE),
                    pct=ifelse(n>0, round(100*pos/n,1), NA_real_))
  bygrp <- dat %>% group_by(Group) %>%
    summarise(n=sum(!is.na(.data[[col]])),
              pos=sum(.data[[col]]==1, na.rm=TRUE),
              pct=ifelse(n>0, round(100*pos/n,1), NA_real_), .groups="drop")
  bind_rows(overall, bygrp)
}
COM_summary <- if (!is.na(COM_col)) make_group_summary_col(at6_only, COM_col) else tibble(note="COM column not found")
PC_summary  <- if (!is.na(PC_col))  make_group_summary_col(at6_only, PC_col)  else tibble(note="PC column not found")

chisq_or_fisher <- function(tab) {
  if (length(tab)==0 || sum(tab, na.rm=TRUE)==0 || nrow(tab)<2 || ncol(tab)<2)
    return(list(test="NA", stat=NA_real_, df=NA_real_, p=NA_real_))
  ok <- tryCatch(all(suppressWarnings(chisq.test(tab))$expected >= 5), error=function(e) FALSE)
  if (ok) { z <- suppressWarnings(chisq.test(tab))
    list(test="chisq", stat=unname(z$statistic), df=unname(z$parameter), p=z$p.value)
  } else { z <- fisher.test(tab)
    list(test="fisher", stat=NA_real_, df=NA_real_, p=z$p.value)
  }
}
COM_tab <- if (!is.na(COM_col)) table(at6_only$Group, at6_only[[COM_col]], useNA="no") else NULL
PC_tab  <- if (!is.na(PC_col))  table(at6_only$Group, at6_only[[PC_col]],  useNA="no") else NULL
COM_test <- if (!is.null(COM_tab)) chisq_or_fisher(COM_tab) else list(test="NA", stat=NA_real_, df=NA_real_, p=NA_real_)
PC_test  <- if (!is.null(PC_tab))  chisq_or_fisher(PC_tab)  else list(test="NA", stat=NA_real_, df=NA_real_, p=NA_real_)
COM_test_tbl <- tibble(Endpoint="COM", Test=COM_test$test, Stat=COM_test$stat, df=COM_test$df, p.value=signif(COM_test$p,3))
PC_test_tbl  <- tibble(Endpoint="PC",  Test=PC_test$test,  Stat=PC_test$stat,  df=PC_test$df,  p.value=signif(PC_test$p,3))

# ===== PI / GI : 0/1로 해석한 비율(%) by Group, by Time =====
as01 <- function(v) suppressWarnings(as.integer(
  gsub("[^0-1]", "", gsub("[[:space:]]+", "", as.character(v)))
))
pi_bin <- as01(raw$PI); gi_bin <- as01(raw$GI)
raw$PI_bin <- ifelse(pi_bin %in% c(0L,1L), pi_bin, NA)
raw$GI_bin <- ifelse(gi_bin %in% c(0L,1L), gi_bin, NA)

bin_summary_by_gt <- function(dat, col){
  dat %>% group_by(Group, Time) %>%
    summarise(n=sum(!is.na(.data[[col]])),
              pos=sum(.data[[col]]==1, na.rm=TRUE),
              pct=ifelse(n>0, round(100*pos/n,1), NA_real_), .groups="drop") %>%
    pivot_wider(id_cols=Group, names_from=Time, values_from=pct, names_prefix="pct_")
}
PI_pct_by_gt <- if ("PI_bin" %in% names(raw)) bin_summary_by_gt(raw, "PI_bin") else tibble(note="no PI")
GI_pct_by_gt <- if ("GI_bin" %in% names(raw)) bin_summary_by_gt(raw, "GI_bin") else tibble(note="no GI")

chisq_or_fisher2 <- function(dat, col){
  times <- na.omit(unique(dat$Time))
  out <- lapply(times, function(tt){
    sub <- dat %>% filter(Time==tt)
    tab <- table(sub$Group, sub[[col]], useNA="no")
    res <- chisq_or_fisher(tab)
    tibble(Time=tt, Test=res$test, Stat=res$stat, df=res$df, p.value=signif(res$p,3))
  })
  bind_rows(out)
}
PI_tests_by_time <- if ("PI_bin" %in% names(raw)) chisq_or_fisher2(raw, "PI_bin") else tibble(note="no PI")
GI_tests_by_time <- if ("GI_bin" %in% names(raw)) chisq_or_fisher2(raw, "GI_bin") else tibble(note="no GI")

## ===== 4) LMM·요약 결과 엑셀 저장 =====
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
  Singularity          = make_sheet(Singularity_flag),
  MASTER_summary_tbl   = make_sheet(master_tbl),
  COM_summary_byGroup  = make_sheet(COM_summary),
  PC_summary_byGroup   = make_sheet(PC_summary),
  COM_PC_tests         = make_sheet(dplyr::bind_rows(COM_test_tbl, PC_test_tbl)),
  PI_pct_byGroup_Time  = make_sheet(PI_pct_by_gt),
  GI_pct_byGroup_Time  = make_sheet(GI_pct_by_gt),
  PI_tests_byTime      = make_sheet(PI_tests_by_time),
  GI_tests_byTime      = make_sheet(GI_tests_by_time)
)
.safe(openxlsx::write.xlsx(x=sheets, file=xlsx_path_results, asTable=TRUE, overwrite=TRUE),
      message("엑셀 저장 실패: ", xlsx_path_results))
.safe(write.csv(master_tbl, csv_path_master, row.names=FALSE),
      message("CSV 저장 실패: ", csv_path_master))
message("DONE --> ", xlsx_path_results, " / ", csv_path_master)

## ===== 5) 0m vs 6m 표: 임상 + 방사선 항목을 한 시트로 통합 =====
t0 <- "0m"; t2 <- "6m"
raw_b6 <- raw %>% filter(Time %in% c(t0, t2)) %>% mutate(Group=factor(Group, levels=c("BO","BT","CO")))

fmt_p   <- function(p) ifelse(is.na(p),"", ifelse(p<0.001,"<0.001", sprintf("%.3f", p)))
fmt_ms  <- function(x) if (length(x)==0 || all(is.na(x))) "" else sprintf("%.1f \u00B1 %.1f", mean(x,na.rm=TRUE), sd(x,na.rm=TRUE))
fmt_pct <- function(x) if (length(x)==0 || all(is.na(x))) "" else sprintf("%.1f", 100*mean(x>0, na.rm=TRUE))

# 군간 p (연속)
p_kw <- function(dat, var, time){
  sub <- dat %>% filter(Time==time) %>%
    mutate(v = suppressWarnings(as.numeric(!!sym(var)))) %>% select(Group, v) %>% filter(!is.na(v))
  if (nrow(sub)<2 || dplyr::n_distinct(sub$Group)<2) return(NA_real_)
  kruskal.test(v ~ Group, data=sub)$p.value
}
# 군내 p (연속; 짝 윌콕슨)
p_wil <- function(dat, var, grp){
  wide <- dat %>% filter(Group==grp) %>%
    mutate(v = suppressWarnings(as.numeric(!!sym(var)))) %>%
    select(ID, Time, v) %>% pivot_wider(names_from=Time, values_from=v)
  if (!all(c(t0,t2) %in% names(wide))) return(NA_real_)
  a <- wide[[t0]]; b <- wide[[t2]]; ok <- !is.na(a) & !is.na(b)
  if (!any(ok)) return(NA_real_)
  suppressWarnings(wilcox.test(a[ok], b[ok], paired=TRUE)$p.value)
}
# Δ(연속) = mean±SD of (0m - 6m)
delta_ms_str <- function(dat, var, grp){
  wide <- dat %>%
    filter(Group==grp) %>% mutate(v = suppressWarnings(as.numeric(!!sym(var)))) %>%
    select(ID, Time, v) %>% pivot_wider(names_from=Time, values_from=v)
  if (!all(c(t0,t2) %in% names(wide))) return("")
  a <- wide[[t0]]; b <- wide[[t2]]; ok <- !is.na(a) & !is.na(b)
  if (!any(ok)) return("")
  d <- a[ok] - b[ok]
  sprintf("%.1f \u00B1 %.1f", mean(d,na.rm=TRUE), sd(d,na.rm=TRUE))
}
# 군내 p (0/비0; McNemar)
p_mcn <- function(dat, var, grp){
  wide <- dat %>% filter(Group==grp) %>%
    mutate(v = !!sym(var)) %>% select(ID, Time, v) %>% pivot_wider(names_from=Time, values_from=v)
  if (!all(c(t0,t2) %in% names(wide))) return(NA_real_)
  a <- as.integer(wide[[t0]]>0); b <- as.integer(wide[[t2]]>0)
  ok <- !is.na(a) & !is.na(b); a <- a[ok]; b <- b[ok]
  if (length(a)==0) return(NA_real_)
  tab <- table(a, b)
  if (nrow(tab)<2 || ncol(tab)<2 || sum(tab)==0 ||
      (if(all(dim(tab)==2)) (tab[1,2]+tab[2,1]) else 0) == 0) return(NA_real_)
  mcnemar.test(tab)$p.value
}
# 군간 p (0/비0; 0m/6m 각각)
p_inter_bin <- function(dat, var, time){
  sub <- dat %>% filter(Time==time) %>% mutate(b = as.integer(!!sym(var) > 0)) %>% select(Group, b) %>% filter(!is.na(b))
  if (nrow(sub)<2 || dplyr::n_distinct(sub$Group)<2 || length(unique(sub$b))<2) return(NA_real_)
  tab <- table(sub$Group, sub$b)
  ok <- tryCatch(all(suppressWarnings(chisq.test(tab))$expected >= 5), error=function(e) FALSE)
  if (ok) suppressWarnings(chisq.test(tab))$p.value else fisher.test(tab)$p.value
}

num <- function(x) suppressWarnings(as.numeric(gsub("[^0-9.+-]", "", as.character(x))))


# Δ(연속) = mean±SD of (0m - 6m)
delta_ms <- function(dat, var, grp){
  wide <- dat %>%
    filter(Group == grp) %>%
    mutate(v = num(.data[[var]])) %>%
    select(ID, Time, v) %>%
    tidyr::pivot_wider(
      names_from = Time, values_from = v,
      values_fn  = list(v = ~mean(., na.rm = TRUE))
    )
  if (!all(c(t0, t2) %in% names(wide))) return(NULL)
  ok <- !is.na(wide[[t0]]) & !is.na(wide[[t2]])
  if (!any(ok)) return(NULL)
  d <- wide[[t0]][ok] - wide[[t2]][ok]
  list(mean = mean(d), sd = sd(d))
}


p_kw_delta <- function(dat, var){
  sub <- dat %>%
    mutate(v = num(.data[[var]])) %>%
    select(ID, Group, Time, v) %>%
    tidyr::pivot_wider(
      names_from = Time, values_from = v,
      values_fn  = list(v = ~mean(., na.rm = TRUE))  # 중복 시간값 평균
    )
  if (!all(c(t0, t2) %in% names(sub))) return(NA_real_)
  ok <- !is.na(sub[[t0]]) & !is.na(sub[[t2]])
  if (!any(ok)) return(NA_real_)
  df <- data.frame(
    d = sub[[t0]][ok] - sub[[t2]][ok],
    g = droplevels(sub$Group[ok])
  )
  df <- df[is.finite(df$d) & !is.na(df$g), ]
  if (nrow(df) < 3 || dplyr::n_distinct(df$g) < 2) return(NA_real_)
  suppressWarnings(kruskal.test(d ~ g, data = df)$p.value)
}


# 표 만들기에서 수정
build_block_cont2 <- function(dat, var, label){
  grps <- levels(dat$Group)
  rows <- lapply(seq_along(grps), function(i){
    g <- grps[i]
    d_res <- delta_ms(dat, var, g)
    tibble(
      `Clinical measurements` = if (i==1) label else "",
      Group = g,
      Baseline  = fmt_ms(dat %>% filter(Group==g, Time==t0) %>% pull(!!sym(var))),
      `6 months`= fmt_ms(dat %>% filter(Group==g, Time==t2) %>% pull(!!sym(var))),
      `Δ baseline to 6 months` =
        if (!is.null(d_res)) sprintf("%.1f ± %.1f", d_res$mean, d_res$sd) else "",
      `p (baseline to 6 months) intra-group` = fmt_p(p_wil(dat, var, g))
    )
  }) %>% bind_rows()

  Δp <- p_kw_delta(dat, var)
  Δp_str <- if (is.na(Δp)) "" else fmt_p(Δp)

  tibble(
    `Clinical measurements` = "p inter-group", Group = "",
    Baseline  = fmt_p(p_kw(dat, var, t0)),
    `6 months`= fmt_p(p_kw(dat, var, t2)),
    `Δ baseline to 6 months` = Δp_str,
    `p (baseline to 6 months) intra-group` = ""
  ) %>% bind_rows(rows, .)
}

build_block_pct2 <- function(dat, var, label){
  grps <- levels(dat$Group)
  rows <- lapply(seq_along(grps), function(i){
    g <- grps[i]
    tibble(
      `Clinical measurements` = if (i==1) label else "",
      Group   = g,
      Baseline = fmt_pct(dat %>% filter(Group==g, Time==t0) %>% pull(!!sym(var))),
      `6 months` = fmt_pct(dat %>% filter(Group==g, Time==t2) %>% pull(!!sym(var))),
      `Δ baseline to 6 months` = {
        wide <- dat %>% filter(Group==g) %>% select(ID, Time, v=.data[[var]]) %>% pivot_wider(names_from=Time, values_from=v)
        if (!all(c(t0,t2) %in% names(wide))) "" else {
          a <- wide[[t0]]; b <- wide[[t2]]; ok <- !is.na(a)&!is.na(b)
          if (!any(ok)) "" else sprintf("%.1f", 100*(mean(a[ok]>0) - mean(b[ok]>0)))
        }
      },
      `p (baseline to 6 months) intra-group` = fmt_p(p_mcn(dat, var, g))
    )
  }) %>% bind_rows()

  tibble(
    `Clinical measurements` = "p inter-group", Group = "",
    Baseline  = fmt_p(p_inter_bin(dat, var, t0)),
    `6 months`= fmt_p(p_inter_bin(dat, var, t2)),
    `Δ baseline to 6 months` = "",
    `p (baseline to 6 months) intra-group` = ""
  ) %>% bind_rows(rows, .)
}

openxlsx::writeData(
  wb, "TABLE_base_6m", TABLE_base_6m,
  startRow = 1, startCol = 1, withFilter = FALSE, na.string = ""
)

openxlsx::writeData(
  wb, "Multiple_regression", reg_sheet,
  withFilter = FALSE, na.string = ""
)

openxlsx::writeData(
  wb, "LMM_Time×Group", TABLE_LMM_TimeXGroup,
  withFilter = FALSE, na.string = ""
)

Δp <- p_kw_delta(dat, var)
Δp_str <- if (is.na(Δp)) "" else fmt_p(Δp)


# ---- 통합 표: 지정 순서(CAL, PPD, BD-AC, MOB, ANGLE, PI, GI) ----
blocks <- list(
  build_block_cont2(raw_b6, "CAL",   "CAL (mm; mean ± SD)"),
  build_block_cont2(raw_b6, "PPD",   "PPD (mm; mean ± SD)"),
  build_block_cont2(raw_b6, "BDAC",  "Linear defect fill (BD-AC) (mm; mean ± SD)"),
  if ("MOB"   %in% names(raw_b6)) build_block_cont2(raw_b6, "MOB",   "Mobility (mean ± SD)"),
  if ("ANGLE" %in% names(raw_b6)) build_block_cont2(raw_b6, "ANGLE", "Defect angle modification (°; mean ± SD)"),
  if ("PI"    %in% names(raw_b6)) build_block_pct2(raw_b6, "PI",  "PI (%)"),
  if ("GI"    %in% names(raw_b6)) build_block_pct2(raw_b6, "GI",  "GI (%)")
)
blocks <- Filter(Negate(is.null), blocks)

spacer <- tibble(`Clinical measurements`="", Group="", Baseline="", `6 months`="", `Δ baseline to 6 months`="", `p (baseline to 6 months) intra-group`="")
TABLE_base_6m <- purrr::imap_dfr(blocks, \(x,i) bind_rows(x, if (i<length(blocks)) spacer else NULL))

# ---- 회귀: CAL gain ~ baseline PPD + MOB + ANGLE + WALLS(3 vs 2) ----
# 0m/6m wide로 전개 후, ΔCAL 계산
wide_b6 <- raw_b6 %>%
  select(ID, Group, Time, CAL, PPD, MOB, ANGLE, WALLS) %>%
  pivot_wider(names_from = Time, values_from = c(CAL, PPD, MOB, ANGLE))
# 벽수 3 vs 2 인코딩
wide_b6 <- wide_b6 %>% mutate(WALLS3 = ifelse(WALLS >= 3, 1L, 0L))
# 회귀 데이터
regdat <- wide_b6 %>%
  transmute(
    CAL_gain = CAL_0m - CAL_6m,
    PPD0  = suppressWarnings(as.numeric(PPD_0m)),
    MOB0  = suppressWarnings(as.numeric(MOB_0m)),
    ANG0  = suppressWarnings(as.numeric(ANGLE_0m)),
    WALLS3 = as.integer(WALLS3)
  ) %>% filter(!is.na(CAL_gain) & !is.na(PPD0) & !is.na(MOB0) & !is.na(ANG0) & !is.na(WALLS3))

if (nrow(regdat) >= 5) {
  fit_lm <- lm(CAL_gain ~ PPD0 + MOB0 + ANG0 + WALLS3, data = regdat)
  ci95   <- confint(fit_lm)
  smm    <- summary(fit_lm)$coefficients
  get_row <- function(term_label, coef_name){
    b   <- smm[coef_name,"Estimate"]
    p   <- smm[coef_name,"Pr(>|t|)"]
    lcl <- ci95[coef_name, 1]; ucl <- ci95[coef_name, 2]
    tibble(
      Variable = term_label,
      `β`      = sprintf("%.2f", b),
      `95% CI` = sprintf("%.2f to %.2f", lcl, ucl),
      `p`      = ifelse(p < 1e-4, "<0.0001", sprintf("%.4f", p))
    )
  }
  REG_table <- bind_rows(
    get_row("PPD baseline (mm)", "PPD0"),
    get_row("Mobility",          "MOB0"),
    get_row("Defect angle (°)",  "ANG0"),
    get_row("Defect morphology (3 vs 2 walls)", "WALLS3")
  )
} else {
  REG_table <- tibble(
    Variable = c("PPD baseline (mm)", "Mobility", "Defect angle (°)", "Defect morphology (3 vs 2 walls)"),
    `β` = "", `95% CI` = "", `p` = ""
  )
}

# ---- 엑셀: 0m-6m 통합표만 기록 (회귀는 별도 시트에서 다룸) ----
xlsx_path <- "table_baseline_6m.xlsx"
wb <- openxlsx::createWorkbook()
openxlsx::addWorksheet(wb, "TABLE_base_6m")

openxlsx::writeData(wb, "TABLE_base_6m", TABLE_base_6m, startRow = 1, startCol = 1, withFilter = FALSE)
openxlsx::setColWidths(wb, "TABLE_base_6m", cols = 1:6, widths = c(30, 8, 18, 18, 22, 26))
sty_hdr <- openxlsx::createStyle(textDecoration="bold", halign="center", valign="center")
openxlsx::addStyle(wb, "TABLE_base_6m", sty_hdr, rows=1, cols=1:6, gridExpand=TRUE)
lab_rows <- which(TABLE_base_6m$`Clinical measurements` != "" & TABLE_base_6m$`Clinical measurements` != "p inter-group") + 1
if (length(lab_rows)) openxlsx::addStyle(wb, "TABLE_base_6m", openxlsx::createStyle(textDecoration="bold"), rows = lab_rows, cols = 1, gridExpand = TRUE)
# p 하이라이트
openxlsx::conditionalFormatting(wb, "TABLE_base_6m", cols=6, rows=2:(nrow(TABLE_base_6m)+1),
  type="expression", rule='OR(LEFT($F2,1)="<", VALUE($F2)<0.05)', style=openxlsx::createStyle(bgFill="#C6EFCE"))
rows_pin <- which(TABLE_base_6m$`Clinical measurements`=="p inter-group") + 1
if (length(rows_pin)){
  openxlsx::conditionalFormatting(
    wb, "TABLE_base_6m", cols=3:4, rows=rows_pin,
    type="expression",
    rule='OR(LEFT(CELL("contents",INDIRECT(ADDRESS(ROW(),COLUMN()))),1)="<", VALUE(INDIRECT(ADDRESS(ROW(),COLUMN())))<0.05)',
    style=openxlsx::createStyle(bgFill = "#C6EFCE")
  )
}

## ---- ΔCAL 회귀: WALLS를 ID별로 고정 추출 후 join ----
num <- function(x) suppressWarnings(as.numeric(as.character(x)))

# 1) ID별 WALLS 한 값만 안전하게 추출
walls_by_id <- raw %>% mutate(WALLS = num(WALLS)) %>%
  group_by(ID) %>% summarise(WALLS = na.omit(WALLS)[1], .groups="drop")

wide0 <- raw %>%
  select(ID, Time, PPD, CAL, MOB, ANGLE) %>%
  mutate(across(c(PPD, CAL, MOB, ANGLE), num)) %>%
  tidyr::pivot_wider(
    names_from = Time, values_from = c(PPD, CAL, MOB, ANGLE),
    values_fn = list(PPD = ~mean(., na.rm=TRUE),
                     CAL = ~mean(., na.rm=TRUE),
                     MOB = ~mean(., na.rm=TRUE),
                     ANGLE= ~mean(., na.rm=TRUE))
  ) %>%
  left_join(walls_by_id, by="ID") %>%
  mutate(
    dCAL   = CAL_0m - CAL_6m,
    PPD0   = PPD_0m, MOB0 = MOB_0m, ANGLE0 = ANGLE_0m,
    WALLS_3vs2 = ifelse(WALLS == 3, 1L, ifelse(WALLS == 2, 0L, NA_integer_))
  ) %>%
  filter(complete.cases(dCAL, PPD0, MOB0, ANGLE0, WALLS_3vs2))


# 3) 회귀 완전케이스 만들기 + 남은 n 확인
dat_reg <- wide0 %>%
  select(dCAL, PPD0, MOB0, ANGLE0, WALLS_3vs2) %>%
  filter(complete.cases(.))

message(sprintf("Regression complete cases: n = %d", nrow(dat_reg)))

# 4) 회귀 적합 및 표 생성
if (nrow(dat_reg) >= 8) {  # 표본이 작으면 8로 완화(필요시 10으로 유지)
  fit <- lm(dCAL ~ PPD0 + MOB0 + ANGLE0 + WALLS_3vs2, data = dat_reg)
  co <- summary(fit)$coefficients
  ci <- confint(fit)

  reg_sheet <- tibble::tibble(
    Variable = c("PPD baseline (mm)", "Mobility", "Defect angle (°)", "Defect morphology (3 vs 2 walls)"),
    beta     = c(co["PPD0","Estimate"], co["MOB0","Estimate"], co["ANGLE0","Estimate"], co["WALLS_3vs2","Estimate"]),
    `95% CI` = sprintf("%.2f to %.2f",
                       c(ci["PPD0",1],   ci["MOB0",1],   ci["ANGLE0",1],   ci["WALLS_3vs2",1]),
                       c(ci["PPD0",2],   ci["MOB0",2],   ci["ANGLE0",2],   ci["WALLS_3vs2",2])),
    p        = sapply(c(co["PPD0","Pr(>|t|)"], co["MOB0","Pr(>|t|)"], co["ANGLE0","Pr(>|t|)"], co["WALLS_3vs2","Pr(>|t|)"]),
                      function(x) ifelse(is.na(x), "", ifelse(x < 0.001, "<0.001", sprintf("%.4f", x))))
  )
} else {
  reg_sheet <- tibble::tibble(Variable = "Insufficient complete cases for regression",
                              `β` = NA, `95% CI` = NA, p = NA)
}

# 5) 별도 시트에 쓰기 (파일 핸들 겹치지 않게 기존 wb 재사용)
wb <- if (file.exists("table_baseline_6m.xlsx")) openxlsx::loadWorkbook("table_baseline_6m.xlsx")
      else openxlsx::createWorkbook()
if ("Multiple_regression" %in% openxlsx::sheets(wb)) openxlsx::removeWorksheet(wb, "Multiple_regression")
openxlsx::addWorksheet(wb, "Multiple_regression")
openxlsx::writeData(wb, "Multiple_regression", reg_sheet, withFilter = FALSE)
openxlsx::setColWidths(wb, "Multiple_regression", cols = 1:4, widths = c(36, 10, 22, 10))
openxlsx::addStyle(wb, "Multiple_regression",
                   openxlsx::createStyle(textDecoration="bold", halign="center"),
                   rows = 1, cols = 1:4, gridExpand = TRUE)
openxlsx::saveWorkbook(wb, "table_baseline_6m.xlsx", overwrite = TRUE)

# ==== LMM Time×Group 표 (각 군 Δ0→6m + 상호작용 p, η²p) ====

# 1) 라벨
label_for <- function(dv){
  switch(dv,
         "CAL"   = "CAL (mm)",
         "PPD"   = "PPD (mm)",
         "BDAC"  = "Linear defect fill (BD-AC) (mm)",
         "ANGLE" = "Defect angle modification (°)",
         "MOB"   = "Mobility",
         dv)
}

fmt_p   <- function(p) ifelse(is.na(p),"", ifelse(p<0.001,"<0.001", sprintf("%.3f",p)))
fmt_ci  <- function(est, lcl, ucl, p) sprintf("%.2f [%.2f, %.2f] (p=%s)", est, lcl, ucl, fmt_p(p))

# 2) DV 하나 블록 만들기
build_timeXgroup_block <- function(dv, raw_dat = raw_b6, res_map = res_list){
  res <- res_map[[dv]]
  if (is.null(res)) return(NULL)

  # (a) within-group Δ(0m-6m) (EMM, Bonferroni)
  emm_t_g <- emmeans::emmeans(res$model, ~ Time | Group)
  ct <- as.data.frame(
    summary(emmeans::contrast(emm_t_g, method = list("0m-6m" = c(1,0,-1)), adjust = "bonferroni"),
            infer = c(TRUE, TRUE))
  ) # Group, estimate, lower.CL, upper.CL, p.value

  # (b) 상호작용 F/p/η²p
  an  <- res$anova
  eff <- res$effsize
  eff_name <- if ("Group:Time" %in% an$Effect) "Group:Time" else "Time:Group"
  F_int <- an$`F value` [an$Effect==eff_name][1]
  p_int <- an$`Pr(>F)`  [an$Effect==eff_name][1]
  eta_p <- eff$Eta2_partial[eff$Effect==eff_name][1]

  grps <- levels(raw_dat$Group)
  rows <- lapply(seq_along(grps), function(i){
    g <- grps[i]
    r <- ct[ct$Group==g, ]
    cell <- if (nrow(r)) fmt_ci(r$estimate, r$lower.CL, r$upper.CL, r$p.value) else ""
    tibble::tibble(
      `Clinical measurements` = if (i==1) label_for(dv) else "",
      Group = g,
      `Δ0–6m` = cell,
      `F (Time×Group)`    = if (i==1) sprintf("%.2f", F_int) else "",
      `p (Time×Group)`     = if (i==1) fmt_p(p_int) else "",
      `η²p (Time×Group)`   = if (i==1) ifelse(is.na(eta_p), "", sprintf("%.2f", eta_p)) else ""
    )
  }) |> dplyr::bind_rows()

  rows
}

# 3) 어떤 DV를 넣을지
dv_for_table <- c("CAL","PPD","BDAC","ANGLE","MOB")
dv_for_table <- dv_for_table[dv_for_table %in% names(res_list)]

blocks <- lapply(dv_for_table, build_timeXgroup_block)
blocks <- Filter(Negate(is.null), blocks)

if (length(blocks)) {
  spacer <- tibble::tibble(`Clinical measurements`="", Group="", `Δ0–6m`="",
                           `F (Time×Group)`="", `p (Time×Group)`="", `η²p (Time×Group)`="")
  TABLE_LMM_TimeXGroup <- purrr::imap_dfr(blocks, \(x,i) dplyr::bind_rows(x, if (i<length(blocks)) spacer else NULL))

  # 4) 엑셀 시트로 쓰기 + p 하이라이트
  xlsx_path <- "table_baseline_6m.xlsx"
  wb <- if (file.exists(xlsx_path)) openxlsx::loadWorkbook(xlsx_path) else openxlsx::createWorkbook()
  if ("LMM_Time×Group" %in% openxlsx::sheets(wb)) openxlsx::removeWorksheet(wb, "LMM_Time×Group")
  openxlsx::addWorksheet(wb, "LMM_Time×Group")
  openxlsx::writeData(wb, "LMM_Time×Group", TABLE_LMM_TimeXGroup, withFilter = FALSE)
  openxlsx::setColWidths(wb, "LMM_Time×Group", cols = 1:6, widths = c(30, 8, 32, 16, 16, 16))

  hdr <- openxlsx::createStyle(textDecoration="bold", halign="center")
  openxlsx::addStyle(wb, "LMM_Time×Group", hdr, rows=1, cols=1:6, gridExpand=TRUE)

  # 블록 라벨(첫 행들) 볼드
  lab_rows <- which(TABLE_LMM_TimeXGroup$`Clinical measurements` != "") + 1
  if (length(lab_rows)) openxlsx::addStyle(wb, "LMM_Time×Group",
                                           openxlsx::createStyle(textDecoration="bold"),
                                           rows=lab_rows, cols=1, gridExpand=TRUE)

  # p(Time×Group) < 0.05 강조
  openxlsx::conditionalFormatting(
    wb, "LMM_Time×Group", cols=5, rows=2:(nrow(TABLE_LMM_TimeXGroup)+1),
    type="expression", rule='OR(LEFT($E2,1)="<", VALUE($E2)<0.05)',
    style=openxlsx::createStyle(bgFill="#C6EFCE")
  )

  openxlsx::saveWorkbook(wb, xlsx_path, overwrite=TRUE)
  message("DONE → LMM_Time×Group sheet appended to ", xlsx_path)
} else {
  message("LMM_Time×Group: 사용할 DV가 없습니다.")
}



################### 그래프 ###################





################# GRAPH ##############

pkgs <- c("dplyr","tibble","purrr","tidyr",
          "lme4","lmerTest","emmeans","performance",
          "openxlsx","stringr","ggplot2","patchwork")
to_install <- pkgs[!pkgs %in% installed.packages()[,"Package"]]
if (length(to_install)) install.packages(to_install, dependencies = TRUE)
invisible(lapply(pkgs, library, character.only = TRUE))


## ===== LMM 궤적 그래프 (EMM 기반, 흑백 저널용) =====
plot_lmm_trajectory <- function(dv_sel,
                                dat   = EMM_Time_in_Group,
                                ylab  = NULL,
                                y_min = 2,
                                y_max = 10){

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

  if (is.null(ylab)) ylab <- paste0(dv_sel, " (mm)")

  ggplot(d, aes(x = Time, y = emmean, group = Group)) +
    ## 선: 모두 검은 실선
    geom_line(linewidth = 0.8, colour = "black") +
    ## 점: 내부 검은색, 테두리 흰색 (●, ■, ▲)
    geom_point(aes(shape = Group),
               size   = 3,
               stroke = 1.0,
               colour = "white",   # border
               fill   = "black") + # inside
    scale_shape_manual(values = c(21, 22, 24)) +
    ## y축: 두 그래프 공통 스케일만 고정 (라벨은 기본 숫자)
    scale_y_continuous(
      limits = c(y_min, y_max),
      breaks = seq(y_min, y_max, by = 2),
      expand = c(0.02, 0)
    ) +
    ## x축 여백 조금
    scale_x_discrete(expand = c(0.2, 0)) +
    labs(
      x = "Time",
      y = ylab
    ) +
    theme_bw(base_size = 11) +
    theme(
      plot.title        = element_blank(),          # 상단 타이틀 제거
      plot.tag          = element_blank(),          # 혹시 모를 tag도 제거
      plot.margin       = margin(t = 6, r = 10, b = 6, l = 10),
      panel.grid.minor  = element_blank(),
      panel.grid.major.x = element_blank(),
      axis.title.x      = element_text(margin = margin(t = 6)),
      axis.title.y      = element_text(margin = margin(r = 6)),
      legend.position   = "right",
      legend.title      = element_blank(),
      legend.key.width  = unit(1.4, "lines")
    )
}

## 개별 그래프
p_cal <- plot_lmm_trajectory("CAL", ylab = "CAL (mm)")
p_ppd <- plot_lmm_trajectory("PPD", ylab = "PPD (mm)")

## 단일 그림 저장 (세로 길게)
ggsave("CAL_LMM_trajectory.png",
       p_cal, width = 4, height = 7, dpi = 600)
ggsave("PPD_LMM_trajectory.png",
       p_ppd, width = 4, height = 7, dpi = 600)

## ===== CAL + PPD 멀티뷰 (범례 하나만 오른쪽) =====
p_multi <- p_ppd + p_cal +
  plot_layout(guides = "collect") & theme(legend.position = "right")

ggsave("CAL_PPD_LMM_multiview.png",
       p_multi, width = 8, height = 7, dpi = 600)

