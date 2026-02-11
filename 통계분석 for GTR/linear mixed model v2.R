.std_names <- function(d) {
  nm <- names(d)
  nm <- gsub("[[:space:]]+", "_", nm)   # 공백 -> _
  nm <- gsub("\\.", "_", nm)            # . -> _
  nm <- gsub("\\(|\\)|>|<|=", "", nm)   # (), <, >, = 제거
  names(d) <- nm
  d
}
.pick_col <- function(nm, pattern, default = NULL) {
  hit <- grep(pattern, nm, value = TRUE, ignore.case = TRUE)
  if (length(hit)) hit[1] else default
}

fit_lmm <- function(dv_name, dat = df_keep, tol_r2 = 1e-8) {
  dat_dv <- dplyr::filter(dat, DV == dv_name)

  m <- lmer(Value ~ Group * Time + (1 | ID),
            data = dat_dv, REML = TRUE,
            control = lmerControl(check.conv.singular = .makeCC("message", tol = 1e-4)))

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
dv_list <- c("PPD","CAL","BDAC")
dv_list <- intersect(dv_list, levels(df_keep$DV))

res_list <- set_names(lapply(dv_list, fit_lmm), dv_list)

ANOVA_all <- bind_rows(lapply(names(res_list), \(dv) res_list[[dv]]$anova |> mutate(DV = dv, .before = 1)))
EffectSize_all <- bind_rows(lapply(names(res_list), \(dv) res_list[[dv]]$effsize |> mutate(DV = dv, .before = 1)))
R2_all <- bind_rows(lapply(names(res_list), \(dv) res_list[[dv]]$r2 |> mutate(DV = dv, .before = 1)))
PostHoc_Group_at_Time <- bind_rows(lapply(names(res_list), \(dv) res_list[[dv]]$post_g_t |> mutate(DV = dv, .before = 1)))
PostHoc_Time_in_Group <- bind_rows(lapply(names(res_list), \(dv) res_list[[dv]]$post_t_g |> mutate(DV = dv, .before = 1)))
EMM_Group_at_Time <- bind_rows(lapply(names(res_list), \(dv) res_list[[dv]]$emm_g_t |> mutate(DV = dv, .before = 1)))
EMM_Time_in_Group <- bind_rows(lapply(names(res_list), \(dv) res_list[[dv]]$emm_t_g |> mutate(DV = dv, .before = 1)))
Singularity_flag <- tibble(
  DV = names(res_list),
  Singular = vapply(res_list, function(x) x$singular, logical(1))
)

fmt_num <- function(x, d = 4) if (is.numeric(x)) round(x, d) else x
ANOVA_all  <- ANOVA_all  |> mutate(across(everything(), fmt_num))
EffectSize_all <- EffectSize_all |> mutate(across(where(is.numeric), ~round(.x, 4)))
R2_all     <- R2_all     |> mutate(across(where(is.numeric), ~round(.x, 4)))
PostHoc_Group_at_Time |> mutate(across(where(is.numeric), ~round(.x, 4))) -> PostHoc_Group_at_Time
PostHoc_Time_in_Group |> mutate(across(where(is.numeric), ~round(.x, 4))) -> PostHoc_Time_in_Group
EMM_Group_at_Time     |> mutate(across(where(is.numeric), ~round(.x, 4))) -> EMM_Group_at_Time
EMM_Time_in_Group     |> mutate(across(where(is.numeric), ~round(.x, 4))) -> EMM_Time_in_Group

wb <- openxlsx::createWorkbook()
for (sn in c("ANOVA_TypeIII","EffectSize_eta2p","R2_Nakagawa",
             "PostHoc_G_at_T","PostHoc_T_in_G","EMM_G_at_T","EMM_T_in_G","Singularity")) {
  openxlsx::addWorksheet(wb, sn)
}
openxlsx::writeData(wb, "ANOVA_TypeIII", ANOVA_all)
openxlsx::writeData(wb, "EffectSize_eta2p", EffectSize_all)
openxlsx::writeData(wb, "R2_Nakagawa", R2_all)
openxlsx::writeData(wb, "PostHoc_G_at_T", PostHoc_Group_at_Time)
openxlsx::writeData(wb, "PostHoc_T_in_G", PostHoc_Time_in_Group)
openxlsx::writeData(wb, "EMM_G_at_T", EMM_Group_at_Time)
openxlsx::writeData(wb, "EMM_T_in_G", EMM_Time_in_Group)
openxlsx::writeData(wb, "Singularity", Singularity_flag)
openxlsx::saveWorkbook(wb, "results_ppd_cal_bdac.xlsx", overwrite = TRUE)

## Master Summary Table (one-glance table)
## 요구 패키지
suppressPackageStartupMessages({
  library(tidyverse)
  library(emmeans)
  library(openxlsx)
})

# ---------- 포맷 유틸 ----------
ci_str <- function(est, lcl, ucl, digits = 2) {
  paste0(
    formatC(est, format="f", digits=digits), " [",
    formatC(lcl, format="f", digits=digits), ", ",
    formatC(ucl, format="f", digits=digits), "]"
  )
}
p_str <- function(p, digits = 3) {
  ifelse(is.na(p), NA,
         ifelse(p < 0.001, "<0.001", formatC(p, format="f", digits=digits)))
}
## "(estimate [CI]) (p=...)"로 묶기
effect_cell <- function(est, lcl, ucl, p) {
  paste0(ci_str(est, lcl, ucl), " (p=", p_str(p), ")")
}

# ---------- 핵심 생성 함수 ----------
make_master_for_dv <- function(res, dv_label) {
  # 1) Δ(0–3, 0–6, 3–6): Group별 시간 대비(EMM 기반)
  #    emmeans는 모델 객체에서 다시 만들면 가장 안전
  m <- res$model

  # by=Group으로 시간 대비 계산 (Bonferroni 보정)
  emm_t_g <- emmeans(m, ~ Time | Group)
  contr <- contrast(emm_t_g,
                    method = list(
                      "0m-3m" = c(1,-1, 0),   # (0m - 3m)
                      "0m-6m" = c(1, 0,-1),   # (0m - 6m)
                      "3m-6m" = c(0, 1,-1)    # (3m - 6m)
                    ),
                    adjust = "bonferroni")
  summ_contr <- as.data.frame(summary(contr, infer = c(TRUE, TRUE))) %>%
    # columns: Group, contrast, estimate, SE, df, lower.CL, upper.CL, t.ratio, p.value
    mutate(DV = dv_label) %>%
    select(DV, Group, contrast, estimate, lower.CL, upper.CL, p.value)

  # 2) 시점별 군간 비교의 최소 p (Bonferroni 보정)
  emm_g_t <- emmeans(m, ~ Group | Time)
  pairs_g_t <- pairs(emm_g_t, adjust = "bonferroni")
  summ_pairs <- as.data.frame(pairs_g_t) %>%  # columns: Time, contrast, estimate, SE, df, t.ratio, p.value
    group_by(Time) %>%
    summarise(p_min_across_groups = min(p.value, na.rm = TRUE), .groups = "drop") %>%
    mutate(DV = dv_label) %>%
    select(DV, Time, p_min_across_groups)

  # 3) Time 주효과 (F, p, partial eta2)
  an <- res$anova  # columns: Effect, Sum Sq, Mean Sq, NumDF, DenDF, F value, Pr(>F)
  eff <- res$effsize # columns: Effect, Eta2_partial, CI_low, CI_high
  time_row <- an %>% filter(Effect == "Time")
  eta_row  <- eff %>% filter(Effect == "Time")
  time_effect <- tibble(
    DV = dv_label,
    Time_F = time_row$`F value` %||% NA_real_,
    Time_p = time_row$`Pr(>F)`  %||% NA_real_,
    Partial_eta2_Time = eta_row$Eta2_partial %||% NA_real_
  )

  # 4) 폭넓은 형태로 묶기: Group 행마다 Δ3개 + 시점별 최소 p 3개 + Time 주효과
  #   a) Δ wide
  delta_wide <- summ_contr %>%
    mutate(cell = effect_cell(estimate, lower.CL, upper.CL, p.value)) %>%
    select(DV, Group, contrast, cell) %>%
    pivot_wider(names_from = contrast, values_from = cell) %>%
    # 열 이름 보기좋게
    rename(`Δ0–3m` = `0m-3m`, `Δ0–6m` = `0m-6m`, `Δ3–6m` = `3m-6m`)

  #   b) 시점별 최소 p wide
  pmin_wide <- summ_pairs %>%
    mutate(pmin_cell = p_str(p_min_across_groups)) %>%
    select(DV, Time, pmin_cell) %>%
    pivot_wider(names_from = Time, values_from = pmin_cell,
                names_prefix = "p(min)_") %>%
    # Time factor가 "0m","3m","6m"일 때 순서를 유지하도록 보장
    relocate(starts_with("p(min)_0m"), .after = DV) %>%
    relocate(starts_with("p(min)_3m"), .after = starts_with("p(min)_0m")) %>%
    relocate(starts_with("p(min)_6m"), .after = starts_with("p(min)_3m"))

  #   c) 모든 조합 병합 (Group 단위로 time_effect 붙일 때는 중복되므로 마지막에 unique로)
  out <- delta_wide %>%
    left_join(pmin_wide, by = "DV") %>%
    left_join(time_effect, by = "DV") %>%
    # display 자리수 정리
    mutate(
      Time_F = round(Time_F, 2),
      Time_p = p_str(Time_p),
      Partial_eta2_Time = round(Partial_eta2_Time, 2)
    ) %>%
    relocate(DV, Group,
             `Δ0–3m`, `Δ0–6m`, `Δ3–6m`,
             starts_with("p(min)_"),
             Time_F, Time_p, Partial_eta2_Time)

  out
}

# ---------- 전체 DV에 대해 마스터 표 생성 ----------
# res_list 객체가 있어야 함: res_list[["PPD"]]$model 등
stopifnot(exists("res_list"))

master_tbl <- purrr::map2_dfr(res_list, names(res_list), ~ make_master_for_dv(.x, .y))

# 보기 좋은 정렬
lv <- c("PPD","CAL","BDAC")
if (all(lv %in% master_tbl$DV)) {
  master_tbl <- master_tbl %>% mutate(DV = factor(DV, levels = lv)) %>% arrange(DV, Group)
}

# 콘솔 확인
print(master_tbl, n = 100)

# ---------- 저장 ----------
# 엑셀 + CSV
openxlsx::write.xlsx(master_tbl, file = "MASTER_summary_table.xlsx", asTable = TRUE)
write.csv(master_tbl, file = "MASTER_summary_table.csv", row.names = FALSE)
