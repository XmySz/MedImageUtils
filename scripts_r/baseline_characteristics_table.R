library(readxl)
library(dplyr)
library(stringr)
library(gtsummary)
library(gt)
library(svglite)
library(grid)

# -------------------------------
# 统一指定"分析变量"名单（含分组变量）
# -------------------------------
analysis_vars <- c(
  "年龄", "性别", "身高", "体重", "BMI", "BMD", "T值", "类别"
)

id_like_vars <- c("住院号", "姓名", "就诊号", "检查号", "病历号")

# -------------------------------
# 预处理（类型矫正 + 排除ID类变量 + 温和缺失过滤）
# -------------------------------
preprocess_data <- function(file_path) {
  dat <- read_xlsx(file_path)
  
  has <- function(v) intersect(v, names(dat))
  
  if ("年龄" %in% names(dat)) {
    dat <- dat |>
      mutate(年龄 = suppressWarnings(as.numeric(str_remove(as.character(年龄), "岁"))))
  }
  
  if ("性别" %in% names(dat)) {
    dat <- dat |>
      mutate(性别 = factor(性别,
                         levels = c("male", "female"),
                         labels = c("男性", "女性")))
  }
  
  if ("类别" %in% names(dat)) {
    dat <- dat |>
      mutate(类别 = factor(类别,
                         levels = c("Normal", "Osteopenia", "Osteoporosis"),
                         labels = c("正常", "骨量减少", "骨质疏松"))) |>
      mutate(类别 = case_when(
        as.character(类别) %in% c("正常", "骨量减少") ~ "正常",
        as.character(类别) == "骨质疏松" ~ "骨质疏松",
        TRUE ~ as.character(类别)
      )) |>
      mutate(类别 = factor(类别, levels = c("正常", "骨质疏松")))
  }
  
  numeric_candidates <- c("身高", "体重", "BMI", "BMD", "T值")
  dat <- dat |>
    mutate(across(all_of(has(numeric_candidates)),
                  ~ suppressWarnings(as.numeric(as.character(.)))))
  
  dat <- dat |>
    select(-any_of(has(id_like_vars)))
  
  used_for_summary <- setdiff(intersect(analysis_vars, names(dat)), "类别")
  if (length(used_for_summary) > 0) {
    dat <- dat |>
      filter(if_all(all_of(used_for_summary), ~ !is.na(.)))
  }
  
  dat
}

# ---------------------------------
# 自定义函数：提取检验统计量 (已修复统计学逻辑)
# ---------------------------------
calc_test_stat <- function(data, variable, by, ...) {
  dat_clean <- data[complete.cases(data[, c(variable, by)]), ]
  
  if (is.numeric(dat_clean[[variable]])) {
    # 连续变量（假设正态）：独立样本 t 检验
    result <- t.test(reformulate(by, variable), data = dat_clean)
    stat_val <- sprintf("%.3f", result$statistic)
    stat_val
  } else {
    # 分类变量：Pearson 卡方检验
    tbl <- table(dat_clean[[variable]], dat_clean[[by]])
    if (min(dim(tbl)) < 2) return(NA_character_)
    result <- suppressWarnings(chisq.test(tbl, correct = FALSE))
    stat_val <- sprintf("%.3f", result$statistic)
    stat_val
  }
}

# ---------------------------------
# 自定义函数：提取检验方法名称 (已修复名称匹配)
# ---------------------------------
calc_test_name <- function(data, variable, by, ...) {
  dat_clean <- data[complete.cases(data[, c(variable, by)]), ]
  
  if (is.numeric(dat_clean[[variable]])) {
    "t" # t检验的统计量名称为 t
  } else {
    "\u03C7\u00B2" # 卡方检验统计量名称为 χ²
  }
}

# ---------------------------------
# 读取数据
# ---------------------------------
path_train <- "Jmszxyy.xlsx"
path_test1 <- "Gz.xlsx"
path_test2 <- "Xry.xlsx"

dat_train  <- preprocess_data(path_train)
dat_test_1 <- preprocess_data(path_test1)
dat_test_2 <- preprocess_data(path_test2)

# ---------------------------------
# 表格函数（含统计量列） (已修复方法设定)
# ---------------------------------
# ---------------------------------
# 表格函数（含统计量列）(已修复 add_stat 报错)
# ---------------------------------
create_chinese_table <- function(data, title) {
  include_vars <- setdiff(intersect(analysis_vars, names(data)), "类别")
  
  tbl <- tbl_summary(
    data,
    by = 类别,
    include = all_of(include_vars),
    statistic = list(
      all_continuous()  ~ "{mean} ± {sd}",
      all_categorical() ~ "{n} ({p}%)"
    ),
    digits = all_continuous() ~ 1,
    label = list(
      年龄 ~ "年龄 (岁)",
      身高 ~ "身高 (cm)",
      体重 ~ "体重 (kg)",
      BMD ~ "骨密度 (g/cm²)",
      T值 ~ "T值",
      性别 ~ "性别",
      BMI ~ "体质指数 (kg/m²)"
    ),
    missing = "no"
  ) |>
    add_overall(last = FALSE, col_label = "**总体**") |>
    
    # ✅ 1. 添加统计量名称列 (默认列名自动生成为 add_stat_1)
    add_stat(
      fns = everything() ~ calc_test_name,
      location = everything() ~ "label"
    ) |>
    
    # ✅ 2. 添加统计量数值列 (默认列名自动生成为 add_stat_2)
    add_stat(
      fns = everything() ~ calc_test_stat,
      location = everything() ~ "label"
    ) |>
    
    # ✅ 3. 添加 P 值列
    add_p(
      test = list(
        all_continuous()  ~ "t.test",
        all_categorical() ~ "chisq.test"
      ),
      pvalue_fun = ~ style_pvalue(.x, digits = 3)
    ) |>
    bold_labels() |>
    
    # ✅ 4. 集中修改所有表头的显示标签
    modify_header(
      label = "**特征**",
      stat_0 = "**总体**, N = {N}",
      add_stat_1 = "**检验方法**",  # 将第一个自定义列重命名
      add_stat_2 = "**统计量**",    # 将第二个自定义列重命名
      p.value = "**P值**"
    ) |>
    modify_header(all_stat_cols() ~ "**{level}**, N = {n}") |>
    # 移除自动脚注
    modify_table_styling(columns = everything(), footnote = "")
  
  tbl
}

# ---------------------------------
# 生成三张表并合并
# ---------------------------------
tbl_train  <- create_chinese_table(dat_train,  "训练集")
tbl_test_1 <- create_chinese_table(dat_test_1, "测试集1")
tbl_test_2 <- create_chinese_table(dat_test_2, "测试集2")

tbl_merged <- tbl_merge(
  tbls = list(tbl_train, tbl_test_1, tbl_test_2),
  tab_spanner = c("**训练集**", "**外部测试集1**", "**外部测试集2**")
)

# 转 gt_tbl
tbl_gt <- gtsummary::as_gt(tbl_merged)

# 完全自定义的中文格式 (已修复脚注说明)
tbl_gt_final <- tbl_gt |>
  gt::tab_header(
    title = "基线特征比较表",
    subtitle = "按骨密度状态分组的患者特征"
  ) |>
  gt::tab_footnote(
    footnote = "连续变量：均值 ± 标准差；分类变量：例数 (百分比)",
    locations = gt::cells_column_labels(columns = 2)
  ) |>
  gt::tab_footnote(
    footnote = "t：独立样本t检验统计量；χ²：卡方检验统计量",
    locations = gt::cells_column_labels(columns = contains("p.value"))
  ) |>
  gt::tab_source_note(
    source_note = "说明：连续变量以均值 ± 标准差表示，组间比较采用独立样本t检验；分类变量以例数 (百分比) 表示，组间比较采用Pearson卡方检验。"
  ) |>
  gt::tab_style(
    style = gt::cell_text(weight = "bold"),
    locations = gt::cells_column_spanners()
  ) |>
  gt::tab_style(
    style = gt::cell_text(size = "small"),
    locations = gt::cells_footnotes()
  ) |>
  gt::tab_style(
    style = gt::cell_text(size = "small", style = "italic"),
    locations = gt::cells_source_notes()
  )

# ---------------------------------
# 导出
# ---------------------------------
gt::gtsave(tbl_gt_final, filename = "基线特征表_纯中文.docx")
gt::gtsave(tbl_gt_final, filename = "基线特征表_纯中文.html")

svglite::svglite("基线特征表_纯中文.svg", width = 28, height = 8)
grid::grid.draw(gt::as_gtable(tbl_gt_final))
dev.off()

print(tbl_gt_final)