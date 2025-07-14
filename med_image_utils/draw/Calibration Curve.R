# ---------------- 0. 环境 ----------------
library(readxl)   # 读 Excel
library(dplyr)
library(tidyr)    # pivot_longer
library(ggplot2)

# ---------------- 1. 读取数据 ----------------
# 改成你的文件路径 / sheet 名
file_path  <- "F:/Data/Jmzxyy/职称申报骨质疏松中文课题/最终表格.xlsx"
raw_df <- read_excel(file_path)
raw_df <- subset(raw_df, split_fold_1 == 'test2')


# ---------------- 2. 整理成长表格式 ----------------
pred_cols <- c("CM", "DLM", "FM")  


df_long <- raw_df %>%
  select(all_of(c("Label", pred_cols))) %>%   # 只保留标签列 + 概率列
  pivot_longer(
    cols   = all_of(pred_cols),
    names_to  = "model",
    values_to = "pred"
  )

# ---------------- 3. 计算校准曲线 ----------------
compute_curve <- function(df, n_bins = 10) {
  df %>%
    mutate(
      bin = cut(pred, breaks = seq(0, 1, length.out = n_bins + 1),
                include.lowest = TRUE)
    ) %>%
    group_by(model, bin) %>%
    summarise(
      mean_pred = mean(pred,  na.rm = TRUE),
      obs_rate  = mean(Label, na.rm = TRUE),
      .groups   = "drop"
    ) %>%
    arrange(model, mean_pred)
}

curve_df <- compute_curve(df_long, n_bins = 10)

# ---------------- 4. 绘图 ----------------
# 自定义颜色 & 节点形状
model_palette <- c(
  CM    = "red",
  DLM    = "navy",
  FM  = "deepskyblue"
)

ggplot(curve_df,
       aes(x = mean_pred, y = obs_rate,
           colour = model, group = model)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +  # 理想校准线
  geom_line(size = 1.2) +
  geom_point(size = 3, shape = 22, fill = "white") +            # 方形节点
  scale_colour_manual(values = model_palette) +
  labs(title = "Calibration Curve",
       x = "Mean predicted value",
       y = "Fraction of positives",
       colour = NULL) +
  theme_bw(base_size = 14) +
  theme(
    plot.title      = element_text(size = 18, face = "bold"),
    legend.position = c(0.85, 0.15),
    legend.background = element_rect(fill = "white", colour = "grey80")
  )
