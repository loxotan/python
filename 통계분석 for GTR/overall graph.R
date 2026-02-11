pkgs <- c("dplyr","tidyr","ggplot2")
to_install <- pkgs[!pkgs %in% installed.packages()[,"Package"]]
if(length(to_install)) install.packages(to_install, dependencies=TRUE)
suppressPackageStartupMessages({library(dplyr); library(tidyr); library(ggplot2)})

num <- function(x) suppressWarnings(as.numeric(gsub("[^0-9.+-]", "", as.character(x))))

raw <- read.csv("RM_ready.csv", check.names=FALSE, fileEncoding="UTF-8")
raw$Time <- tolower(gsub("\\s+","", as.character(raw$Time)))
raw$Time <- ifelse(raw$Time %in% c("0m","0month","0months"), "0m",
                   ifelse(raw$Time %in% c("6m","6month","6months"), "6m", raw$Time))
raw$Time <- factor(raw$Time, levels=c("0m","6m"))

vars <- c("PPD","CAL","MOB","PI","GI","BDAC","ANGLE")
vars <- intersect(vars, names(raw))

dat <- raw %>%
  filter(Time %in% c("0m","6m")) %>%
  select(ID, Time, all_of(vars)) %>%
  pivot_longer(all_of(vars), names_to="DV", values_to="Value") %>%
  mutate(Value = num(Value)) %>%
  filter(is.finite(Value)) %>%
  group_by(ID, DV, Time) %>%
  summarise(Value = mean(Value, na.rm=TRUE), .groups="drop")

lab <- c(PPD="PPD (mm)", CAL="CAL (mm)", MOB="Mobility",
         PI="PI", GI="GI", BDAC="LDF (BD-AC) (mm)", ANGLE="Defect angle (°)")
dat$DV <- factor(dat$DV, levels=vars, labels=lab[vars])

# ---- 증가/감소 방향 계산(각 ID-DV 단위) ----
dir_tbl <- dat %>%
  pivot_wider(names_from=Time, values_from=Value) %>%
  mutate(delta = `6m` - `0m`,
         Direction = case_when(
           is.na(delta) ~ NA_character_,
           delta > 0    ~ "Increase",
           delta < 0    ~ "Decrease",
           TRUE         ~ "No change"
         )) %>%
  select(ID, DV, Direction)

dat2 <- dat %>% left_join(dir_tbl, by=c("ID","DV"))

# ---- 전체 평균(패널별 Time 평균) ----
mean_df <- dat2 %>%
  group_by(DV, Time) %>%
  summarise(Mean = mean(Value, na.rm=TRUE), .groups="drop")

# ---- 플롯 ----
p <- ggplot(dat2, aes(x=Time, y=Value, group=ID, color=Direction)) +
  geom_line(alpha=0.45, linewidth=0.7) +
  geom_point(alpha=0.65, size=1.4) +
  # 평균 경향선(굵게)
  geom_line(data=mean_df, aes(x=Time, y=Mean, group=1),
            inherit.aes=FALSE, linewidth=1.2, color="black") +
  geom_point(data=mean_df, aes(x=Time, y=Mean),
             inherit.aes=FALSE, size=2.2, color="black") +
  facet_wrap(~DV, scales="free_y") +
  scale_color_manual(values=c("Decrease"="blue", "Increase"="red", "No change"="grey60"),
                     breaks=c("Decrease","Increase","No change")) +
  theme_classic() +
  theme(legend.position="bottom") +
  labs(x=NULL, y=NULL, color=NULL)

print(p)

# ---- PNG 저장 ----
ggsave("spaghetti_0m_6m_overall.png", p, width=10, height=8, dpi=300)
