library(readxl)
library(ggplot2)
library(dplyr)

plot_pcr_confusion_matrix <- function(y_true, y_pred, title = "Confusion Matrix") {
  class_levels <- c("Recurrence", "Recurrence-Free")
  y_true <- factor(y_true, levels = class_levels)
  y_pred <- factor(y_pred, levels = class_levels)
  
  # 创建混淆矩阵数据
  cm_data <- as.data.frame(table(Actual = y_true, Predicted = y_pred))
  colnames(cm_data) <- c("Actual", "Predicted", "Count")
  
  # 计算行内百分比
  cm_data <- cm_data %>%
    group_by(Actual) %>%
    mutate(Percentage = Count / sum(Count) * 100) %>%
    ungroup()
  
  p <- ggplot(cm_data, aes(x = Predicted, y = Actual, fill = Percentage)) +
    geom_tile(color = "white", linewidth = 1.2) +
    geom_text(aes(label = Count),  color = ifelse(cm_data$Percentage > 50, "white", "black"),  size = 10, fontweight = "bold") +
    scale_y_discrete(limits = rev) +
    scale_fill_gradient(
      low = "#E3F2FD",     
      high = "#0D47A1",    
      name = "",
      labels = function(x) paste0(round(x), "%"),
      limits = c(0, 100),
      guide = guide_colorbar(
        barheight = grid::unit(0.71, "npc"),
        barwidth = grid::unit(0.02, "npc"),
        frame.colour = "black",
        ticks.colour = "black",
        title.position = "top",
        title.hjust = 0.5
      )
    ) +
    labs(
      title = title,
      x = "OC-MCAT Prediction",
      y = "Pathologist evaluation"
    ) +
    coord_fixed() +
    theme_minimal(base_size = 14) +
    theme(
      # 标题样式
      plot.title = element_text(
        hjust = 0.5, 
        size = 18, 
        face = "bold",
        margin = margin(b = -10)
      ),
      
      # 坐标轴标题
      axis.title.x = element_text(
        size = 16, 
        face = "bold",
        margin = margin(t = 15)
      ),
      axis.title.y = element_text(
        size = 16, 
        face = "bold", 
        margin = margin(r = 15)
      ),
      
      # 坐标轴文本
      axis.text.x = element_text(
        size = 14, 
        color = "black", 
        face = "bold",
        margin = margin(t = -10) 
        
      ),
      
      # Y轴文本垂直显示
      axis.text.y = element_text(
        size = 14, 
        color = "black", 
        face = "bold",
        angle = 90,
        hjust = 0.5,
        margin = margin(r = -10) 
      ),
      
      # 去除网格线
      panel.grid = element_blank(),
      
      # 图例位置和样式
      legend.position = "right",
      legend.margin = margin(l=10, b=20),
      legend.text = element_text(size = 12, face = "bold"),
      
      # 调整整体边距
      plot.margin = margin(20, 20, 20, 20)
    )
  
  return(p)
}

df <- read_excel("D:\\Data\\OvarianCancer\\Materials\\Label.xlsx")
val = subset(df, split_1 == "test")
y_true_labels <- ifelse(val$Label == 1, "Recurrence", "Recurrence-Free")
y_pred_binary <- ifelse(val$Ours > 0.5, 1, 0) 
y_pred_labels <- ifelse(y_pred_binary == 1, "Recurrence", "Recurrence-Free") 

p = plot_pcr_confusion_matrix(y_true_labels, y_pred_labels, title = "External Test Set\n Confusion Matrix")
print(p)
