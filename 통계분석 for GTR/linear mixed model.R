## ── 패키지
req <- c("tidyverse","lme4","lmerTest","emmeans","effectsize","performance")
to_install <- setdiff(req, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install)
invisible(lapply(req, library, character.only = TRUE))

## ── 데이터 로드 (엑셀에서 만든 RM_ready.csv: ID,DV,Time,Value[,Group])
df <- read.csv("RM_ready.csv", stringsAsFactors = FALSE)
if (!"Group" %in% names(df)) df$Group <- sub("-.*$", "", df$ID)

## ── factor 및 순서
df <- df %>%
  mutate(
    ID    = factor(ID),
    Group = factor(Group),
    DV    = factor(DV),
    Time  = factor(Time, levels = c("0m","3m","6m"), ordered = TRUE)
  )

## ── semi-join 규칙: 6m 없는 ID는 제외(6m만 있어도 포함). 이후 0·3·6m 모두 사용.
ids_with_6m <- df %>%
  filter(Time == "6m", !is.na(Value)) %>%
  distinct(DV, ID)

df_keep <- df %>%
  semi_join(ids_with_6m, by = c("DV","ID")) %>%
  droplevels()

## ── df 진단용: 원자료 평균과 필터 후 평균 비교(선택)
cell_means <- function(dat, tag) {
  dat %>% group_by(DV, Group, Time) %>%
    summarise(n = sum(!is.na(Value)),
              mean = mean(Value, na.rm = TRUE), .groups = "drop") %>%
    mutate(source = tag)
}
print(bind_rows(cell_means(df, "raw_all"),
                cell_means(df_keep, "keep_6m_only")) %>%
        arrange(DV, Group, Time, source), n = 100)

## ── KR or Satterthwaite df
if (requireNamespace("pbkrtest", quietly = TRUE)) {
  emm_options(lmer.df = "kenward-roger")
} else {
  emm_options(lmer.df = "satterthwaite")
}

## ── LMM 적합(특이점 표기 포함)
fit_lmm <- function(dv_name, dat = df_keep, tol_r2 = 1e-8) {
  dat_dv <- dat %>% filter(DV == dv_name)
  m <- lmer(
    Value ~ Group * Time + (1 | ID),
    data = dat_dv, REML = TRUE,
    control = lmerControl(check.conv.singular = .makeCC("message", tol = 1e-4))
  )
  list(
    model    = m,
    singular = lme4::isSingular(m, tol = 1e-5),
    anova    = anova(m, type = 3),
    r2       = tryCatch(performance::r2_nakagawa(m, tolerance = tol_r2), error = function(e) NULL),
    emm_g_t  = emmeans(m, ~ Group | Time),
    emm_t_g  = emmeans(m, ~ Time  | Group)
  )
}

res_bdac <- fit_lmm("BDAC", df_keep)
res_cal  <- fit_lmm("CAL",  df_keep)
res_ppd  <- fit_lmm("PPD",  df_keep)

# 예시 확인
res_ppd$anova; res_ppd$singular; res_ppd$r2

## ── EMM 테이블(모델기반 추정평균)
emm_all <- bind_rows(
  as.data.frame(res_bdac$emm_g_t) %>% mutate(DV = "BDAC"),
  as.data.frame(res_cal$emm_g_t)  %>% mutate(DV = "CAL"),
  as.data.frame(res_ppd$emm_g_t)  %>% mutate(DV = "PPD")
) %>%
  mutate(
    DV    = factor(DV, levels = c("BDAC","CAL","PPD")),
    Time  = factor(Time, levels = c("0m","3m","6m"), ordered = TRUE),
    Group = factor(Group)
  )

## ── 원자료 점 준비: 3m&6m 모두 가진 ID 표시(개별 relapse 시각화)
ids_36 <- df_keep %>%
  filter(Time %in% c("3m","6m")) %>%
  distinct(DV, ID, Time) %>%
  tidyr::pivot_wider(names_from = Time, values_from = Time) %>%
  filter(!is.na(`3m`), !is.na(`6m`)) %>%
  select(DV, ID)

raw_points <- df_keep %>%
  mutate(kind = ifelse(paste(DV, ID) %in% paste(ids_36$DV, ids_36$ID),
                       "paired_3m6m", "all_kept"))

## ── 그래프: EMM 라인 + 95% CI + 원자료 점(3m 포함)
library(ggplot2)
pd  <- position_dodge(width = 0.15)
pjd <- position_jitterdodge(jitter.width = 0.06, jitter.height = 0, dodge.width = 0.15)

ggplot(emm_all, aes(Time, emmean, group = Group, color = Group, fill = Group)) +
  geom_ribbon(aes(ymin = lower.CL, ymax = upper.CL),
              alpha = 0.18, linewidth = 0, position = pd) +
  geom_line(linewidth = 1.1, position = pd) +
  geom_point(size = 2.6, shape = 21, stroke = 0.8, colour = "black", position = pd) +
  # 원자료 점(에러 원인 수정: x=Time, group=Group 지정, jitterdodge 사용)
  #geom_point(data = raw_points,
             #aes(x = Time, y = Value, group = Group, shape = kind),
             #position = pjd, color = "grey30", alpha = 0.35,
             #inherit.aes = FALSE, size = 1.8) +
  scale_shape_manual(values = c(all_kept = 16, paired_3m6m = 17),
                     breaks = c("paired_3m6m","all_kept"),
                     labels = c("paired 3m&6m","all kept")) +
  facet_wrap(~ DV, scales = "free_y", nrow = 1) +
  labs(x = "Time", y = "Estimated mean (EMM)", shape = NULL) +
  theme_classic(base_size = 13) +
  theme(legend.position = "top",
        legend.title = element_blank(),
        strip.background = element_rect(fill = "grey95", colour = NA),
        strip.text = element_text(face = "bold"))
