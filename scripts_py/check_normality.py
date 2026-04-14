import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pathlib import Path

def plot_normality(df: pd.DataFrame, column_name: str, save_path: str = None):
    """
    对表格中的指定连续型变量列进行正态分布的图示法检验（直方图、KDE以及Q-Q图）。
    
    参数:
        df: pandas DataFrame 数据表
        column_name: 需要进行检验的列名
        save_path: 若提供，则将绘制的图片保存到该路径
    """
    # 过滤掉缺失值
    data = df[column_name].dropna()
    
    if len(data) == 0:
        raise ValueError(f"列 '{column_name}' 没有有效数据！")

    # 设置绘图风格
    sns.set_theme(style="whitegrid")
    # 支持中文显示（如果系统存在对应字体）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. 直方图 + 核密度估计 (KDE) + 正态分布拟合曲线
    ax1 = axes[0]
    sns.histplot(data, kde=True, stat="density", ax=ax1, color="skyblue", bins=30, label='直方图与 KDE')
    
    # 获取该列数据的均值和标准差
    mu, std = stats.norm.fit(data)
    xmin, xmax = ax1.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    
    # 叠加理想正态分布的曲线
    ax1.plot(x, p, 'k', linewidth=2, label=f'标准正态分布\n(mu={mu:.2f}, std={std:.2f})')
    ax1.set_title(f'"{column_name}" 密度分布与理论正态分布对比', fontsize=12)
    ax1.set_xlabel('数值')
    ax1.set_ylabel('密度')
    ax1.legend()

    # 2. Q-Q 图 (Quantile-Quantile Plot)
    ax2 = axes[1]
    stats.probplot(data, dist="norm", plot=ax2)
    ax2.set_title(f'"{column_name}" 的 Q-Q 图', fontsize=12)
    ax2.set_xlabel('理论分位数 (Theoretical Quantiles)')
    ax2.set_ylabel('实际样本分位数 (Ordered Values)')
    
    # 大标题
    plt.suptitle(f'图示法正态分布检验 - {column_name}', fontsize=16)
    plt.tight_layout()

    # 显示或保存
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # ==========================
    # 在这里直接配置文件路径和列名
    # ==========================
    EXCEL_PATH = r"D:\Data\Jmszxyy\职称申报骨质疏松中文课题\投稿\20260414返修\Jmszxyy.xlsx"
    COLUMN_NAME = "T值"
    # 如果需要将绘制的图表保存为文件，在这里指定路径，如 r"D:\Data\output.png"。留空则直接弹出窗口展示。
    SAVE_PATH = None  
    
    if Path(EXCEL_PATH).exists():
        print(f"正在读取表格: {EXCEL_PATH}")
        # 读取 Excel 文件 (需要环境中安装了 openpyxl)
        df = pd.read_excel(EXCEL_PATH)
        
        if COLUMN_NAME in df.columns:
            plot_normality(df, COLUMN_NAME, save_path=SAVE_PATH)
        else:
            print(f"错误: 表格中找不到名为 '{COLUMN_NAME}' 的列。该表格现有的表头包含: {list(df.columns)}")
    else:
        print(f"提示: 未能在该路径找到表格 -> {EXCEL_PATH}")
        print("⬇️ 下面为您演示一份随机生成的虚拟数据正态检验效果...\n")
        
        # 跑一段自带测试 Demo
        np.random.seed(42)
        test_df = pd.DataFrame({
            "年龄": np.random.normal(loc=50, scale=10, size=500), # 模拟服从正态分布的年龄数据
            "病灶体积": np.random.exponential(scale=2, size=500)  # 模拟右偏分布的体积数据
        })
        
        print(">> 展示 [符合] 正态分布的数据图示:")
        plot_normality(test_df, "年龄")
        
        print(">> 展示 [不符合] 正态分布(右偏)的数据图示:")
        plot_normality(test_df, "病灶体积")
