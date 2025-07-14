# ------------------- 0. 准备环境 -------------------
library(readxl)
library(ROCR)

# ------------------- 1. 读取数据 -------------------
file_path <- "F:\\Data\\Jmzxyy\\职称申报骨质疏松中文课题\\最终表格.xlsx"
data <- read_excel(file_path)

# ------------------- 2. 清理数据 -------------------
clean_data <- function(pred, label) na.omit(data.frame(pred = pred, label = label))

data <- subset(data, split_fold_1 == "val")

data1 <- clean_data(data$Predicted_Probability_Fold1, data$Label)
data2 <- clean_data(data$Predicted_Probability_Fold2, data$Label)
data3 <- clean_data(data$Predicted_Probability_Fold3, data$Label)
data4 <- clean_data(data$Predicted_Probability_Fold4, data$Label)
data5 <- clean_data(data$Predicted_Probability_Fold5, data$Label)

# ------------------- 3. ROC & AUC -------------------
get_perf <- function(df) {
  pred <- prediction(df$pred, df$label)
  list(roc = performance(pred, "tpr", "fpr"),
       auc = performance(pred, "auc")@y.values[[1]])
}

perf1 <- get_perf(data1)
perf2 <- get_perf(data2)
perf3 <- get_perf(data3)
perf4 <- get_perf(data4)
perf5 <- get_perf(data5)

# ------------------- 4. 绘图 -------------------
par(
  mar = c(5, 5, 4, 8),
  cex = 1.2,       # 全局基准字号 ↑10%
  cex.main = 1.6,  # 主标题更大
  cex.lab  = 1.4,  # 坐标轴标题
  cex.axis = 1.2   # 刻度数字
)

plot(NULL, NULL, xlim = c(0,1), ylim = c(0,1),
     xlab = "False positive rate",
     ylab = "True positive rate",
     main = "Internal Valid",
     )

curve_cols <- c("red","green","blue","black")
for(i in 1:5) {
  perf <- get(paste0("perf", i))$roc
  lines(perf@x.values[[1]], perf@y.values[[1]], col = curve_cols[i], lwd = 3)
}

abline(0,1,lty = 2, col = "grey")

legend("bottomright",
       legend = sprintf("FOLD %d (AUC = %.3f)",
                        1:5,
                        sapply(1:5, function(i) get(paste0("perf", i))$auc)),
       col = curve_cols, lwd = 3, bty = "n")
# grid()
