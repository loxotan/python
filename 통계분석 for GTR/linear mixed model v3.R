## ===== 0) 필수 패키지 =====
pkgs <- c(
  "dplyr","tibble","purrr","tidyr",
  "lme4","lmerTest","emmeans","performance",
  "openxlsx","stringr"
)
to_install <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
if (length(to_install)) install.packages(to_install, dependencies = TRUE)

suppressPackageStartupMessages({
  library(dplyr); library(tibble); library(purrr); library(tidyr)
  library(lme4);  library(lmerTest); library(emmeans); library(performance)
  library(openxlsx); library(stringr)
})

## 안전 연산자
`%||%` <- function(x, y) if (is.null(x) || length(x)==0) y else x

## ===== 1) 데이터 준비 =====
# df_keep 가 없으면 RM_ready.csv 에서 생성
if (!exists("df_keep")) {
  stopifnot(file.exists("RM_ready.csv"))
  df_raw <- read.csv("RM_ready.csv", na.strings = c("", "NA"))
  # 기대 열: ID, Group, Time, PPD, CAL, BDAC, mob, angle
  need_cols <- c("ID","Group","Time","PPD","CAL","BDAC")
  miss <- setdiff(need_cols, names(df_raw))
  if (length(miss)) stop("RM_ready.csv 누락 열: ", paste(miss, collapse=", "))

  df_keep <- df_raw %>%
    mutate(
      ID    = as.factor(ID),
      Group = as.factor(Group),
      Time  = factor(Time, levels = c("0m","3m","6m"))
    ) %>%
    pivot_longer(c(PPD,CAL,BDAC), names_to = "DV", values_to = "Value") %>%
    filter(!is.na(Value))
}

## ===== 2) 보조 함수 =====
.std_names <- function(d) {
  nm <- names(d)
  nm <- gsub("[[:space:]]+", "_", nm)
  nm <- gsub("\\.", "_", nm)
  nm <- gsub("\\(|\\)|>|<|=", "", nm)
  names(d) <- nm
  d
}
.pick_col <- function(nm, pattern, default = NULL) {
  hit <- grep(pattern, nm, value = TRUE, ignore.case = TRUE)
  if (length(hit)) hit[1] else default
}

fit_lmm <- function(dv_name, dat = df_keep, tol_r2 = 1e-8) {
  dat_dv <- dplyr::filter(dat, DV == dv_name)

  m <- lmer(
    Value ~ Group * Time + (1 | ID),
    data = dat_dv, REML = TRUE,
    control = lme4::lmerControl(check.conv.singular = "message")
  )

  an_raw <- anova(m, type = 3) |> as.data.frame() |> tibble::rownames_to_column("Effect")
  an_cln <- .std_names(an_raw)
  nm <- names(an_cln)

  col_sum  <- .pick_col(nm, "^Sum_?Sq$")
  col_mean <- .pick_col(nm, "^Mean_?Sq$")
  col_num  <- .pick_col(nm, "^NumDF$")
  col_den  <- .pick_col(nm, "^DenDF$")
  col_F    <- .pick_col(nm, "^F_?value$|^F$")
  col_p    <- .pick_col(nm, "^Pr|^P_?value$|^p_?value$")

  if (any(sapply(list(col_sum,col_mean,col_num,col_den,col_F,col_p), is.null))) {
    stop("ANOVA 열 탐지 실패: ", paste(nm, collapse=", "))
  }

  anova_tbl <- dplyr::transmute(
    an_cln,
    Effect,
    `Sum Sq`  = .data[[col_sum]],
    `Mean Sq` = .data[[col_mean]],
    NumDF     = .data[[col_num]],
    DenDF     = .data[[col_den]],
    `F value` = .data[[col_F]],
    `Pr(>F)`  = .data[[col_p]]
  )

  effsize <- dplyr::transmute(
    anova_tbl,
    Effect,
    Eta2_partial = ifelse(is.finite(`F value`),
                          (`F value` * NumDF) / ((`F value` * NumDF) + DenDF),
                          NA_real_),
    CI_low  = NA_real_,  CI_high = NA_real_
  )

  r2 <- tryCatch(performance::r2_nakagawa(m, tolerance = tol_r2), error = function(e) NULL)
  r2_tbl <- tibble::tibble(
    Metric = c("Conditional R2","Marginal R2"),
    Value  = c(if (!is.null(r2)) r2$R2_conditional else NA_real_,
               if (!is.null(r2)) r2$R2_marginal    else NA_real_)
  )

  emm_g_t <- emmeans(m, ~ Group | Time)
  emm_t_g <- emmeans(m, ~ Time  | Group)

  post_g_t <- as.data.frame(pairs(emm_g_t, adjust = "bonferroni")) |>
    dplyr::rename(Contrast = contrast, Estimate = estimate, SE = SE, df = df,
                  t.ratio = t.ratio, p.value = p.value)

  post_t_g <- as.data.frame(pairs(emm_t_g, adjust = "bonferroni")) |>
    dplyr::rename(Contrast = contrast, Estimate = estimate, SE = SE, df = df,
                  t.ratio = t.ratio, p.value = p.value)

  list(model=m, singular=lme4::isSingular(m, tol=1e-5),
       anova=anova_tbl, effsize=effsize, r2=r2_tbl,
       emm_g_t=as.data.frame(emm_g_t), emm_t_g=as.data.frame(emm_t_g),
       post_g_t=post_g_t, post_t_g=post_t_g)
}

## ===== 3) 모델 적합 =====
dv_list <- intersect(c("PPD","CAL","BDAC"), unique(df_keep$DV))
res_list <- purrr::set_names(lapply(dv_list, fit_lmm), dv_list)

## ===== 4) 결과 테이블 병합 =====
ANOVA_all <- dplyr::bind_rows(lapply(names(res_list), \(dv)
  res_list[[dv]]$anova |> dplyr::mutate(DV = dv, .before = 1)))

EffectSize_all <- dplyr::bind_rows(lapply(names(res_list), \(dv)
  res_list[[dv]]$effsize |> dplyr::mutate(DV = dv, .before = 1)))

R2_all <- dplyr::bind_rows(lapply(names(res_list), \(dv)
  res_list[[dv]]$r2 |> dplyr::mutate(DV = dv, .before = 1)))

PostHoc_Group_at_Time <- dplyr::bind_rows(lapply(names(res_list), \(dv)
  res_list[[dv]]$post_g_t |> dplyr::mutate(DV = dv, .before = 1)))

PostHoc_Time_in_Group <- dplyr::bind_rows(lapply(names(res_list), \(dv)
  res_list[[dv]]$post_t_g |> dplyr::mutate(DV = dv, .before = 1)))

EMM_Group_at_Time <- dplyr::bind_rows(lapply(names(res_list), \(dv)
  res_list[[dv]]$emm_g_t |> dplyr::mutate(DV = dv, .before = 1)))

EMM_Time_in_Group <- dplyr::bind_rows(lapply(names(res_list), \(dv)
  res_list[[dv]]$emm_t_g |> dplyr::mutate(DV = dv, .before = 1)))

Singularity_flag <- tibble::tibble(
  DV = names(res_list),
  Singular = vapply(res_list, function(x) x$singular, logical(1))
)

## 보기 좋게 반올림
fmt_num <- function(x, d = 4) if (is.numeric(x)) round(x, d) else x
ANOVA_all  <- ANOVA_all  |> dplyr::mutate(across(everything(), fmt_num))
EffectSize_all <- EffectSize_all |> dplyr::mutate(across(where(is.numeric), ~round(.x, 4)))
R2_all     <- R2_all     |> dplyr::mutate(across(where(is.numeric), ~round(.x, 4)))

## ===== 5) Master summary table 생성 =====
ci_str <- function(est, lcl, ucl, digits = 2) {
  paste0(formatC(est, format="f", digits=digits), " [",
         formatC(lcl, format="f", digits=digits), ", ",
         formatC(ucl, format="f", digits=digits), "]")
}
p_str <- function(p, digits = 3) {
  ifelse(is.na(p), NA,
         ifelse(p < 0.001, "<0.001", formatC(p, format="f", digits=digits)))
}
effect_cell <- function(est, lcl, ucl, p) paste0(ci_str(est, lcl, ucl), " (p=", p_str(p), ")")

make_master_for_dv <- function(res, dv_label) {
  m <- res$model

  emm_t_g <- emmeans(m, ~ Time | Group)
  contr <- contrast(emm_t_g,
                    method = list("0m-3m" = c(1,-1,0),
                                  "0m-6m" = c(1,0,-1),
                                  "3m-6m" = c(0,1,-1)),
                    adjust = "bonferroni")
  summ_contr <- as.data.frame(summary(contr, infer = c(TRUE, TRUE))) %>%
    mutate(DV = dv_label) %>%
    select(DV, Group, contrast, estimate, lower.CL, upper.CL, p.value)

  emm_g_t <- emmeans(m, ~ Group | Time)
  pairs_g_t <- pairs(emm_g_t, adjust = "bonferroni")
  summ_pairs <- as.data.frame(pairs_g_t) %>%
    group_by(Time) %>%
    summarise(p_min_across_groups = min(p.value, na.rm = TRUE), .groups = "drop") %>%
    mutate(DV = dv_label) %>%
    select(DV, Time, p_min_across_groups)

  an  <- res$anova
  eff <- res$effsize
  time_row <- an  %>% filter(Effect == "Time")
  eta_row  <- eff %>% filter(Effect == "Time")
  time_effect <- tibble(
    DV = dv_label,
    Time_F = time_row$`F value` %||% NA_real_,
    Time_p = time_row$`Pr(>F)`  %||% NA_real_,
    Partial_eta2_Time = eta_row$Eta2_partial %||% NA_real_
  )

  delta_wide <- summ_contr %>%
    mutate(cell = effect_cell(estimate, lower.CL, upper.CL, p.value)) %>%
    select(DV, Group, contrast, cell) %>%
    pivot_wider(names_from = contrast, values_from = cell) %>%
    rename(`Δ0–3m` = `0m-3m`, `Δ0–6m` = `0m-6m`, `Δ3–6m` = `3m-6m`)

  pmin_wide <- summ_pairs %>%
    mutate(pmin_cell = p_str(p_min_across_groups)) %>%
    select(DV, Time, pmin_cell) %>%
    pivot_wider(names_from = Time, values_from = pmin_cell, names_prefix = "p(min)_: ")

  out <- delta_wide %>%
    left_join(pmin_wide, by = "DV") %>%
    left_join(time_effect, by = "DV") %>%
    mutate(Time_F = round(Time_F, 2),
           Time_p = p_str(Time_p),
           Partial_eta2_Time = round(Partial_eta2_Time, 2)) %>%
    relocate(DV, Group)
  out
}

stopifnot(length(res_list) > 0)
master_tbl <- purrr::map2_dfr(res_list, names(res_list), ~ make_master_for_dv(.x, .y))
lv <- c("PPD","CAL","BDAC")
if (all(lv %in% master_tbl$DV)) {
  master_tbl <- master_tbl %>% mutate(DV = factor(DV, levels = lv)) %>% arrange(DV, Group)
}

## ===== 6) 저장 =====
openxlsx::write.xlsx(
  x = list(
    ANOVA_TypeIII      = ANOVA_all,
    EffectSize_eta2p   = EffectSize_all,
    R2_Nakagawa        = R2_all,
    PostHoc_G_at_T     = PostHoc_Group_at_Time,
    PostHoc_T_in_G     = PostHoc_Time_in_Group,
    EMM_G_at_T         = EMM_Group_at_Time,
    EMM_T_in_G         = EMM_Time_in_Group,
    Singularity        = Singularity_flag,
    MASTER_summary_tbl = master_tbl
  ),
  file = "results_ppd_cal_bdac.xlsx",
  asTable = TRUE
)
write.csv(master_tbl, "MASTER_summary_table.csv", row.names = FALSE)

message("완료: results_ppd_cal_bdac.xlsx / MASTER_summary_table.csv")
