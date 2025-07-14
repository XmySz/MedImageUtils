# ---------------- 0. 环境 ----------------
library(readxl)    
library(rmda)      
library(dplyr)
library(tidyr)
library(ggplot2)


# ---------------- 1. 读取 & 整理数据 ----------------
file_path <- "F:/Data/Jmzxyy/职称申报骨质疏松中文课题/最终表格.xlsx"
raw_df    <- read_excel(file_path, sheet = "Sheet3")

outcome_col <- "Label"
pred_cols   <- c("CM", "DLM", "FM")

df <- raw_df %>%
  select(all_of(c(outcome_col, pred_cols))) %>%
  mutate(across(all_of(outcome_col), as.numeric))

# --- 2. 计算 DCA 曲线 --------------------------------------------------
# 建议的阈值范围（可按实际场景调整）
thresholds <- seq(0, 1, by = 0.01)

# 分别为 3 个模型生成 decision_curve 对象
dca_cm  <- decision_curve(Label ~ CM,
                          data       = df,
                          thresholds = thresholds,
                          confidence.intervals = 0.95,   # 生成 95% CI
                          study.design = "cohort")       # 若为病例-对照，请改为 "case-control"

dca_dlm <- decision_curve(Label ~ DLM,
                          data = df,
                          thresholds = thresholds,
                          confidence.intervals = 0.95,
                          study.design = "cohort")

dca_fm  <- decision_curve(Label ~ FM,
                          data = df,
                          thresholds = thresholds,
                          confidence.intervals = 0.95,
                          study.design = "cohort")

# --- 3. 绘制 DCA 曲线 --------------------------------------------------
dca_list <- list(CM  = dca_cm,
                 DLM = dca_dlm,
                 FM  = dca_fm)

plot_decision_curve(
  dca_list,
  curve.names          = names(dca_list),   # 图例名称
  cost.benefit.axis    = TRUE,              # 同时显示 cost-benefit 轴
  confidence.intervals = FALSE,             # 如需显示 CI 改为 TRUE
  standardize          = TRUE,              # 标准化净受益
  legend.position      = "topright"      # 图例位置
)

title("Decision Curve Analysis for Osteoporosis Risk Models")
