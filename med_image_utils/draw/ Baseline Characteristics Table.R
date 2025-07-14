library(readxl)
library(dplyr)
library(stringr)
library(gtsummary)
library(gt)
library(svglite)
library(grid)

preprocess_data <- function(file_path) {
  dat <- read_xlsx(file_path) |>
    mutate(
      年龄 = as.numeric(年龄),
      性别 = factor(性别, levels = c("male", "female")),
      类别 = factor(类别, levels = c("Normal", "Osteopenia", "Osteoporosis")),
      身高 = as.numeric(身高),
      体重 = as.numeric(体重),
      BMD = as.numeric(BMD),
      T值 = as.numeric(T值),
      BMI = as.numeric(BMI),
    ) |>
    filter(if_all(everything(), ~ !is.na(.)))
  return(dat)
}

# 1. 读取和预处理数据 ----------------------------------------------------
path_train <- "E:\\Dataset\\江门市中心医院_新\\data\\Labels.xlsx"
path_test1 <- "E:\\Dataset\\广州市第一人民医院_新\\data\\Labels.xlsx"
path_test2 <- "E:\\Dataset\\新人医_新\\data\\Labels.xlsx"

dat_train <- preprocess_data(path_train)
dat_test_1 <- preprocess_data(path_test1)
dat_test_2 <- preprocess_data(path_test2)

# 数据质量检查
cat("训练集样本量:", nrow(dat_train), "\n")
cat("测试集1样本量:", nrow(dat_test_1), "\n")
cat("测试集2样本量:", nrow(dat_test_2), "\n")

# 检查各组样本量
print("训练集各组样本量:")
table(dat_train$类别)
print("测试集1各组样本量:")
table(dat_test_1$类别)
print("测试集2各组样本量:")
table(dat_test_2$类别)

# 2. 为训练集生成 Table 1 -------------------------------------------------
tbl_train <- tbl_summary(
  dat_train,
  by = 类别,
  statistic = list(
    all_continuous()  ~ "{mean} ± {sd}",
    all_categorical() ~ "{n} ({p}%)"
  ),
  digits  = all_continuous() ~ 1,
  label = list(
    年龄 ~ "Age (years)",
    身高 ~ "Height (cm)",
    体重 ~ "Weight (kg)",
    BMD ~ "BMD (g/cm²)",
    T值 ~ "T-score",
    性别 ~ "Sex",
    BMI ~ "BMI (kg/m²)"
  ),
  missing = "no"
) |>
  add_overall(last = FALSE) |>
  add_p(test = list(
    all_continuous()  ~ "kruskal.test",  # 改用非参数检验
    all_categorical() ~ "fisher.test"    # 改用Fisher精确检验
  )) |>
  bold_labels()

# 3. 为外部测试集1生成 Table 1 ---------------------------------------------
include_vars_test <- c("年龄", "性别", "身高", "体重", "BMI", "BMD", "T值", "类别")
include_vars_test <- intersect(include_vars_test, names(dat_test_1))
include_vars_test_for_summary <- setdiff(include_vars_test, "类别")

tbl_test_1 <- tbl_summary(
  dat_test_1,
  by = 类别,
  include = all_of(include_vars_test_for_summary),
  statistic = list(
    all_continuous()  ~ "{mean} ± {sd}",
    all_categorical() ~ "{n} ({p}%)"
  ),
  digits  = all_continuous() ~ 1,
  label = list(
    年龄 ~ "Age (years)",
    身高 ~ "Height (cm)",
    体重 ~ "Weight (kg)",
    BMD ~ "BMD (g/cm²)",
    T值 ~ "T-score",
    性别 ~ "Sex",
    BMI ~ "BMI (kg/m²)"
  ),
  missing = "no"
) |>
  add_overall(last = FALSE) |>
  add_p(test = list(
    all_continuous()  ~ "kruskal.test",  # 改用非参数检验
    all_categorical() ~ "fisher.test"    # 改用Fisher精确检验
  )) |>
  bold_labels()

# 4. 为外部测试集2生成 Table 2 ---------------------------------------------
include_vars_test <- c("年龄", "性别", "身高", "体重", "BMI", "BMD", "T值", "类别")
include_vars_test <- intersect(include_vars_test, names(dat_test_2))
include_vars_test_for_summary <- setdiff(include_vars_test, "类别")

tbl_test_2 <- tbl_summary(
  dat_test_2,
  by = 类别,
  include = all_of(include_vars_test_for_summary),
  statistic = list(
    all_continuous()  ~ "{mean} ± {sd}",
    all_categorical() ~ "{n} ({p}%)"
  ),
  digits  = all_continuous() ~ 1,
  label = list(
    年龄 ~ "Age (years)",
    身高 ~ "Height (cm)",
    体重 ~ "Weight (kg)",
    BMD ~ "BMD (g/cm²)",
    T值 ~ "T-score",
    性别 ~ "Sex",
    BMI ~ "BMI (kg/m²)"
  ),
  missing = "no"
) |>
  add_overall(last = FALSE) |>
  add_p(test = list(
    all_continuous()  ~ "kruskal.test",  # 改用非参数检验
    all_categorical() ~ "fisher.test"    # 改用Fisher精确检验
  )) |>
  bold_labels()

# 5. 合并表格 -------------------------------------------------------------
tbl_merged <- tbl_merge(
  tbls = list(tbl_train, tbl_test_1, tbl_test_2),
  tab_spanner = c("**Training Set**", "**External Test Set 1**", "**External Test Set 2**")
)

# 6. 转成 gt_tbl ------------------------------------------------------------
tbl_gt <- gtsummary::as_gt(tbl_merged)   # ★核心修正：显式指定命名空间

# 7. 导出 Word/HTML ---------------------------------------------------------
gt::gtsave(tbl_gt, filename = "F:/baseline_table_combined.docx")
gt::gtsave(tbl_gt, filename = "F:/baseline_table_combined.html")

# 8. 生成 SVG ---------------------------------------------------------------
svglite::svglite("F:/baseline_table_combined.svg", width = 25, height = 8)
grid::grid.draw(gt::as_gtable(tbl_gt))
dev.off()


