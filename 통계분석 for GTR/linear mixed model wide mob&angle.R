options(stringsAsFactors = FALSE, warn = 1)

## ===== 0) 패키지 =====
pkgs <- c("dplyr","tibble","purrr","tidyr",
          "lme4","lmerTest","emmeans","performance",
          "openxlsx","stringr","forcats","rlang")
to_install <- pkgs[!pkgs %in% installed.packages()[,"Package"]]
if (length(to_install)) install.packages(to_install, dependencies = TRUE)

suppressPackageStartupMessages({
  library(dplyr); library(tibble); library(purrr); library(tidyr)
  library(lme4);  library(lmerTest); library(emmeans); library(performance)
  library(openxlsx); library(stringr); library(forcats); library(rlang)
})

`%||%` <- function(x, y) if (is.null(x) || length(x)==0) y else x
BOM  <- "\uFEFF"; NBSP <- "\u00A0"

fix_names <- function(nm){
  nm <- enc2utf8(nm)
  nm <- gsub(BOM,  "", nm, fixed = TRUE)
  nm <- gsub("\\x{00A0}", "", nm, perl = TRUE)
  trimws(nm)
}
to_bin01 <- function(x){
  s <- enc2utf8(as.character(x))
  s <- gsub(BOM,  "", s, fixed = TRUE)
  s <- gsub("\\x{00A0}", "", s, perl = TRUE)
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
ci_str <- function(est,lcl,ucl,digits=2) paste0(formatC(est,format="f",digits=digits)," [",
                                                formatC(lcl,format="f",digits=digits),", ",
                                                formatC(ucl,format="f",digits=digits),"]")
p_str <- function(p,digits=3) ifelse(is.na(p), NA, ifelse(p<0.001,"<0.001",formatC(p,format="f",digits=digits)))
effect_cell <- function(est,lcl,ucl,p) paste0(ci_str(est,lcl,ucl)," (p=",p_str(p),")")

## ===== 1) 파일 =====
stopifnot(file.exists("RM_ready.csv"))
raw0 <- read.csv("RM_ready.csv", na.strings=c("","NA"), check.names = FALSE,
                 fileEncoding = "UTF-8")
names(raw0) <- fix_names(names(raw0))
message("WD: ", getwd())
message("Columns: ", paste(names(raw0), collapse=", "))

has_wide <- all(c("ID","Group","Time","PPD","CAL","BDAC") %in% names(raw0))
has_long <- all(c("ID","DV","Time","Value") %in% names(raw0))
if (!has_wide && !has_long) stop("RM_ready.csv 형식을 인식할 수 없습니다.")

raw <- if (has_wide) raw0 else {
  raw0 %>% mutate(Group = factor(str_extract(ID, "^[A-Za-z]{2}"))) %>%
    pivot_wider(names_from = DV, values_from = Value)
}
for (nm in c("PPD","CAL","BDAC","MOB","ANGLE","COM","PC")) if (!nm %in% names(raw)) raw[[nm]] <- NA

raw$ID    <- factor(raw$ID)
raw$Group <- factor(raw$Group %||% sub("^([A-Za-z]{2}).*$","\\1", as.character(raw$ID)))
raw$Time  <- enc2utf8(as.character(raw$Time)) |> gsub("\\s+","",x=_) |> tolower()
raw$Time  <- gsub("^0m$|^0month$|^0months$","0m", raw$Time)
raw$Time  <- gsub("^3m$|^3month$|^3months$","3m", raw$Time)
raw$Time  <- gsub("^6m$|^6month$|^6months$","6m", raw$Time)
raw$Time  <- factor(raw$Time, levels=c("0m","3m","6m"))

## ===== 1-B) 서브그룹 변수 생성(베이스라인에서) =====
base <- raw %>% filter(Time=="0m") %>%
  transmute(ID,
            MOB_BASE   = suppressWarnings(as.numeric(MOB)),
            ANGLE_BASE = suppressWarnings(as.numeric(ANGLE)))
base <- base %>%
  mutate(
    MOB_BIN   = case_when(!is.na(MOB_BASE)   & MOB_BASE >= 2  ~ "2-3",
                          !is.na(MOB_BASE)                   ~ "0-1",
                          TRUE ~ NA_character_),
    ANGLE_BIN = case_when(!is.na(ANGLE_BASE) & ANGLE_BASE >= 30 ~ "\u226530",
                          !is.na(ANGLE_BASE)                    ~ "<30",
                          TRUE ~ NA_character_)
  )

subg_map <- base %>% select(ID, MOB_BIN, ANGLE_BIN)
raw <- raw %>% left_join(subg_map, by="ID") %>%
  mutate(MOB_BIN=factor(MOB_BIN, levels=c("0-1","2-3")),
         ANGLE_BIN=factor(ANGLE_BIN, levels=c("<30","\u226530")))

## ===== 2) LONG =====
cont_cols <- intersect(c("PPD","CAL","BDAC","MOB","ANGLE"), names(raw))

## raw에는 이미 subg_map가 조인되어 MOB_BIN/ANGLE_BIN이 존재함
df_keep <- raw %>%
  pivot_longer(all_of(cont_cols), names_to = "DV", values_to = "Value") %>%
  filter(!is.na(Value))

## ===== 2-B) LONG + 서브그룹(전역 한 번만)
df_keep_sub <- df_keep %>%
  mutate(
    MOB_BIN   = factor(MOB_BIN,   levels = c("0-1","2-3")),
    ANGLE_BIN = factor(ANGLE_BIN, levels = c("<30","\u226530"))
  )

## ===== anova() 컬럼 안전 추출기 =====
.pick <- function(df){
  nm <- names(df)
  list(
    f   = intersect(nm, c("F value","F.value","F"))[1],
    p   = intersect(nm, c("Pr(>F)","Pr..F.","Pr..F.."))[1],
    ndf = intersect(nm, c("NumDF","num.Df","numDf"))[1],
    ddf = intersect(nm, c("DenDF","den.Df","denDf"))[1]
  )
}

## ===== 공통 LMM =====
fit_lmm <- function(dv_name, dat=df_keep, tol_r2=1e-8){
  dat_dv <- dplyr::filter(dat, DV==dv_name)
  if (n_distinct(dat_dv$Time) < 2 || nrow(dat_dv) < 5) return(NULL)
  m <- .safe(lmer(Value ~ Group * Time + (1|ID), data=dat_dv, REML=TRUE,
                  control=lme4::lmerControl(check.conv.singular="message")), NULL)
  if (is.null(m)) return(NULL)

  an_raw <- .safe(anova(m, type=3) |> as.data.frame() |> tibble::rownames_to_column("Effect"), NULL)
  if (is.null(an_raw)) return(NULL)
  keys <- .pick(an_raw); if (any(is.na(unlist(keys)))) return(NULL)

  anova_tbl <- an_raw %>%
    transmute(Effect,
              `F value` = .data[[keys$f]],
              `Pr(>F)`  = .data[[keys$p]],
              NumDF     = .data[[keys$ndf]],
              DenDF     = .data[[keys$ddf]])

  effsize <- transmute(anova_tbl, Effect,
                       Eta2_partial = ifelse(is.finite(`F value`),
                         (`F value`*NumDF)/((`F value`*NumDF)+DenDF), NA_real_),
                       CI_low=NA_real_, CI_high=NA_real_)
  r2 <- suppressWarnings(performance::r2_nakagawa(m, tolerance=tol_r2))
  r2_tbl <- tibble(Metric=c("Conditional R2","Marginal R2"),
                   Value=c(r2$R2_conditional %||% NA_real_, r2$R2_marginal %||% NA_real_))

  emm_g_t_obj <- emmeans(m, ~ Group | Time)
  emm_t_g_obj <- emmeans(m, ~ Time  | Group)
  emm_g_t <- as.data.frame(emm_g_t_obj)
  emm_t_g <- as.data.frame(emm_t_g_obj)
  post_g_t <- as.data.frame(pairs(emm_g_t_obj, adjust="bonferroni"))
  post_t_g <- as.data.frame(contrast(emm_t_g_obj, method=list("0m-6m"=c(1,0,-1)), adjust="bonferroni"))

  list(model=m, singular=lme4::isSingular(m, tol=1e-5),
       anova=anova_tbl, effsize=effsize, r2=r2_tbl,
       emm_g_t=emm_g_t, emm_t_g=emm_t_g, post_g_t=post_g_t, post_t_g=post_t_g)
}

dv_list <- intersect(c("PPD","CAL","BDAC","MOB","ANGLE"), unique(df_keep$DV))
res_list <- setNames(lapply(dv_list, fit_lmm), dv_list)
res_list <- res_list[!sapply(res_list, is.null)]

ANOVA_all <- bind_rows(lapply(names(res_list), \(dv) res_list[[dv]]$anova %>% mutate(DV=dv, .before=1)))
EffectSize_all <- bind_rows(lapply(names(res_list), \(dv) res_list[[dv]]$effsize %>% mutate(DV=dv, .before=1)))
R2_all <- bind_rows(lapply(names(res_list), \(dv) res_list[[dv]]$r2 %>% mutate(DV=dv, .before=1)))
PostHoc_Group_at_Time <- bind_rows(lapply(names(res_list), \(dv) res_list[[dv]]$post_g_t %>% mutate(DV=dv, .before=1)))
PostHoc_Time_in_Group <- bind_rows(lapply(names(res_list), \(dv) res_list[[dv]]$post_t_g %>% mutate(DV=dv, .before=1)))
EMM_Group_at_Time <- bind_rows(lapply(names(res_list), \(dv) res_list[[dv]]$emm_g_t %>% mutate(DV=dv, .before=1)))
EMM_Time_in_Group <- bind_rows(lapply(names(res_list), \(dv) res_list[[dv]]$emm_t_g %>% mutate(DV=dv, .before=1)))
Singularity_flag <- tibble(DV=names(res_list), Singular=vapply(res_list, function(x) x$singular, logical(1)))

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
    select(DV, Group, contrast, cell) %>% pivot_wider(names_from = contrast, values_from = cell) %>%
    rename(`Δ0–6m` = `0m-6m`)
  pmin_wide <- summ_pairs %>% mutate(pmin_cell=p_str(p_min_across_groups)) %>%
    select(DV, Time, pmin_cell) %>% pivot_wider(names_from=Time, values_from=pmin_cell, names_prefix="p(min)_: ")
  delta_wide %>% left_join(pmin_wide, by="DV") %>% left_join(time_effect, by="DV")
}
master_tbl <- if (length(res_list)) purrr::map2_dfr(res_list, names(res_list), make_master_for_dv) else tibble()

## ===== 3) COM / PC (6m) =====
find_col <- function(cands, nm=names(raw)) {
  hit <- grep(paste0("(", paste(cands, collapse="|"), ")"), nm, ignore.case=TRUE, value=TRUE)
  if (length(hit)) hit[1] else NA_character_
}
COM_col <- find_col(c("^COM$","composite","COM_?6","COM\\s*\\(6m\\)"))
PC_col  <- find_col(c("^PC$","pocket.?closure","PC_?6","PC\\s*\\(6m\\)"))

if (!is.na(COM_col)) raw[[COM_col]] <- to_bin01(raw[[COM_col]])
if (!is.na(PC_col))  raw[[PC_col]]  <- to_bin01(raw[[PC_col]])

at6 <- raw %>% filter(Time=="6m")
if (!is.na(COM_col)) at6[[COM_col]] <- ifelse(at6$Time=="6m", at6[[COM_col]], NA)
if (!is.na(PC_col))  at6[[PC_col]]  <- ifelse(at6$Time=="6m", at6[[PC_col]],  NA)

bin_summary <- function(dat, col){
  dat <- dat %>% filter(!is.na(.data[[col]]))
  tibble(n=nrow(dat), pos=sum(dat[[col]]==1), pct=ifelse(nrow(dat)>0, round(100*sum(dat[[col]]==1)/nrow(dat),1), NA))
}
make_group_summary_col <- function(dat, col){
  overall <- tibble(Group="ALL",
                    n   = sum(!is.na(dat[[col]])),
                    pos = sum(dat[[col]] == 1, na.rm = TRUE),
                    pct = ifelse(n > 0, round(100*pos/n, 1), NA_real_))
  bygrp <- dat %>% group_by(Group) %>% summarise(
    n   = sum(!is.na(.data[[col]])),
    pos = sum(.data[[col]] == 1, na.rm = TRUE),
    pct = ifelse(n > 0, round(100*pos/n, 1), NA_real_), .groups="drop")
  bind_rows(overall, bygrp)
}
COM_summary <- if (!is.na(COM_col)) make_group_summary_col(at6, COM_col) else tibble(note="COM column not found")
PC_summary  <- if (!is.na(PC_col))  make_group_summary_col(at6, PC_col)  else tibble(note="PC column not found")

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
COM_tab <- if (!is.na(COM_col)) table(at6$Group, at6[[COM_col]], useNA="no") else NULL
PC_tab  <- if (!is.na(PC_col))  table(at6$Group, at6[[PC_col]],  useNA="no") else NULL
COM_test <- if (!is.null(COM_tab)) chisq_or_fisher(COM_tab) else list(test="NA", stat=NA_real_, df=NA_real_, p=NA_real_)
PC_test  <- if (!is.null(PC_tab))  chisq_or_fisher(PC_tab)  else list(test="NA", stat=NA_real_, df=NA_real_, p=NA_real_)
COM_test_tbl <- tibble(Endpoint="COM", Test=COM_test$test, Stat=COM_test$stat, df=COM_test$df, p.value=signif(COM_test$p,3))
PC_test_tbl  <- tibble(Endpoint="PC",  Test=PC_test$test,  Stat=PC_test$stat,  df=PC_test$df,  p.value=signif(PC_test$p,3))

logit_safely <- function(dat, resp, preds, inter=FALSE){
  ok <- all(c(resp, preds) %in% names(dat))
  if (!ok) return(tibble(note="no data"))
  d <- dat %>% select(all_of(c("Group", preds, resp))) %>% filter(!is.na(.data[[resp]]))
  if (nrow(d)==0 || length(unique(d[[resp]]))<2) return(tibble(note="insufficient variation"))
  f <- if (inter) reformulate(c("Group", preds, paste0("Group:", preds)), response = resp)
       else       reformulate(c("Group", preds), response = resp)
  m <- suppressWarnings(glm(f, data=d, family=binomial()))
  s <- summary(m)$coefficients
  tibble(Contrast=rownames(s), Estimate=s[,1], SE=s[,2], z=s[,3], p.value=s[,4],
         OR=exp(s[,1]), `OR 95% L`=exp(s[,1]-1.96*s[,2]), `OR 95% U`=exp(s[,1]+1.96*s[,2]))
}

## ===== 4) 서브그룹 LMM (최종) — .before 제거 버전
fit_lmm_subgroup <- function(dv, fact_col, dat = df_keep_sub){
  dat <- dat %>% filter(DV == dv)
  if (!fact_col %in% names(dat)) { warning(paste("서브그룹 컬럼이 없습니다:", fact_col)); return(NULL) }
  dat <- dat %>% filter(!is.na(.data[[fact_col]]))
  if (nrow(dat) < 8 || n_distinct(dat[[fact_col]]) < 2) return(NULL)

  fml <- as.formula(paste0("Value ~ Group * Time * ", fact_col, " + (1|ID)"))
  m <- .safe(lmer(fml, data = dat, REML = TRUE,
                  control = lme4::lmerControl(check.conv.singular = "message")), NULL)
  if (is.null(m)) return(NULL)

  an <- as.data.frame(anova(m, type = 3)) %>% tibble::rownames_to_column("Effect")
  keys <- .pick(an); if (any(is.na(unlist(keys)))) return(NULL)

  an <- an %>%
    transmute(
      DV = dv, Factor = fact_col, Effect,
      `F value` = .data[[keys$f]], `Pr(>F)` = .data[[keys$p]],
      NumDF     = .data[[keys$ndf]], DenDF = .data[[keys$ddf]]
    ) %>% relocate(DV, Factor, Effect)

  # ← 여기만 문자열로 안전하게
  emm_t_gf <- emmeans(m, as.formula(paste("~ Time | Group *", fact_col)))
  contr    <- contrast(emm_t_gf, method = list("0m-6m" = c(1,0,-1)), adjust = "bonferroni")

  # ← 여기서도 문자열 기반 rename
  delta <- as.data.frame(summary(contr, infer = c(TRUE,TRUE))) %>%
    rename(!!"FactorLevel" := !!fact_col) %>%
    mutate(DV = dv, Factor = fact_col,
           Cell = effect_cell(estimate, lower.CL, upper.CL, p.value)) %>%
    select(DV, Factor, Group, FactorLevel, contrast, Cell) %>%
    tidyr::pivot_wider(names_from = contrast, values_from = Cell, names_prefix = "Δ") %>%
    relocate(DV, Factor, Group, FactorLevel)

  list(anova = an, delta = delta)
}


dv_for_mob   <- c("CAL","PPD","BDAC")
dv_for_angle <- c("CAL","PPD","BDAC","ANGLE")

mob_res   <- setNames(lapply(dv_for_mob,   \(dv) fit_lmm_subgroup(dv, "MOB_BIN")),   dv_for_mob)
angle_res <- setNames(lapply(dv_for_angle, \(dv) fit_lmm_subgroup(dv, "ANGLE_BIN")), dv_for_angle)

# NULL 제거
mob_res   <- Filter(Negate(is.null), mob_res)
angle_res <- Filter(Negate(is.null), angle_res)

SUBG_MOB_LMM <- if (length(mob_res)) bind_rows(lapply(mob_res,   \(x) x$anova)) else tibble()
SUBG_ANG_LMM <- if (length(angle_res)) bind_rows(lapply(angle_res, \(x) x$anova)) else tibble()
SUBG_MOB_DLT <- if (length(mob_res)) bind_rows(lapply(mob_res,   \(x) x$delta)) else tibble()
SUBG_ANG_DLT <- if (length(angle_res)) bind_rows(lapply(angle_res, \(x) x$delta)) else tibble()

names(df_keep_sub)              # MOB_BIN, ANGLE_BIN 존재해야 함
df_keep_sub %>% count(DV, MOB_BIN)
df_keep_sub %>% count(DV, ANGLE_BIN)


pick_p <- function(tbl, patt){
  if (is.null(tbl) || nrow(tbl)==0) return(tibble())
  tbl %>% filter(grepl(patt, Effect)) %>%
    transmute(DV, Factor, Effect, `F value`, p.value=`Pr(>F)`) %>%
    mutate(p.value=signif(p.value,3))
}
SUBG_MOB_INTER <- bind_rows(
  pick_p(SUBG_MOB_LMM, "Time:MOB_BIN$"),
  pick_p(SUBG_MOB_LMM, "Group:Time:MOB_BIN")
)
SUBG_ANG_INTER <- bind_rows(
  pick_p(SUBG_ANG_LMM, "Time:ANGLE_BIN$"),
  pick_p(SUBG_ANG_LMM, "Group:Time:ANGLE_BIN")
)

## ===== 5) COM/PC 로지스틱(6m) =====
COM_col <- find_col(c("^COM$","composite","COM_?6","COM\\s*\\(6m\\)"))
PC_col  <- find_col(c("^PC$","pocket.?closure","PC_?6","PC\\s*\\(6m\\)"))
SUBG_LOGIT_MOB_COM  <- if (!is.na(COM_col)) logit_safely(at6, COM_col,  "MOB_BIN",   inter=TRUE) else tibble(note="COM NA")
SUBG_LOGIT_MOB_PC   <- if (!is.na(PC_col))  logit_safely(at6, PC_col,   "MOB_BIN",   inter=TRUE) else tibble(note="PC NA")
SUBG_LOGIT_ANG_COM  <- if (!is.na(COM_col)) logit_safely(at6, COM_col,  "ANGLE_BIN", inter=TRUE) else tibble(note="COM NA")
SUBG_LOGIT_ANG_PC   <- if (!is.na(PC_col))  logit_safely(at6, PC_col,   "ANGLE_BIN", inter=TRUE) else tibble(note="PC NA")

mob_tab_COM <- if (!is.na(COM_col)) table(at6$MOB_BIN,   at6[[COM_col]],  useNA="no") else NULL
mob_tab_PC  <- if (!is.na(PC_col))  table(at6$MOB_BIN,   at6[[PC_col]],   useNA="no") else NULL
ang_tab_COM <- if (!is.na(COM_col)) table(at6$ANGLE_BIN, at6[[COM_col]],  useNA="no") else NULL
ang_tab_PC  <- if (!is.na(PC_col))  table(at6$ANGLE_BIN, at6[[PC_col]],   useNA="no") else NULL

tab_test <- function(tab, name){
  if (is.null(tab)) return(tibble(Endpoint=name, Test="NA", Stat=NA, df=NA, p=NA))
  z <- chisq_or_fisher(tab)
  tibble(Endpoint=name, Test=z$test, Stat=z$stat, df=z$df, p=signif(z$p,3))
}
SUBG_COM_PC_tests <- bind_rows(
  tab_test(mob_tab_COM, "COM_by_MOB"), tab_test(mob_tab_PC, "PC_by_MOB"),
  tab_test(ang_tab_COM, "COM_by_ANGLE"), tab_test(ang_tab_PC, "PC_by_ANGLE")
)

## ===== 6) 저장 =====
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
  COM_PC_tests         = make_sheet(bind_rows(COM_test_tbl, PC_test_tbl)),

  SUBGROUP_MOB_LMM         = make_sheet(SUBG_MOB_LMM),
  SUBGROUP_ANGLE_LMM       = make_sheet(SUBG_ANG_LMM),
  SUBGROUP_MOB_Delta_0m_6m = make_sheet(SUBG_MOB_DLT),
  SUBGROUP_ANGLE_Delta_0m_6m = make_sheet(SUBG_ANG_DLT),
  SUBGROUP_MOB_Inter_ax_p  = make_sheet(SUBG_MOB_INTER),
  SUBGROUP_ANGLE_Inter_ax_p = make_sheet(SUBG_ANG_INTER),

  SUBGROUP_COM_PC_MOB_logit = make_sheet(bind_rows(
    tibble(Endpoint="COM", SUBGROUP="MOB_BIN") %>% bind_cols(SUBG_LOGIT_MOB_COM),
    tibble(Endpoint="PC",  SUBGROUP="MOB_BIN") %>% bind_cols(SUBG_LOGIT_MOB_PC)
  )),
  SUBGROUP_COM_PC_ANGLE_logit = make_sheet(bind_rows(
    tibble(Endpoint="COM", SUBGROUP="ANGLE_BIN") %>% bind_cols(SUBG_LOGIT_ANG_COM),
    tibble(Endpoint="PC",  SUBGROUP="ANGLE_BIN") %>% bind_cols(SUBG_LOGIT_ANG_PC)
  )),
  SUBGROUP_COM_PC_tests   = make_sheet(SUBG_COM_PC_tests)
)

xlsx_path <- "results_ppd_cal_bdac.xlsx"
csv_path  <- "MASTER_summary_table.csv"

.safe(openxlsx::write.xlsx(x=sheets, file=xlsx_path, asTable=TRUE, overwrite=TRUE),
      message("엑셀 저장 실패: ", xlsx_path))
.safe(write.csv(master_tbl, csv_path, row.names=FALSE),
      message("CSV 저장 실패: ", csv_path))
message("DONE --> ", xlsx_path, " / ", csv_path)
