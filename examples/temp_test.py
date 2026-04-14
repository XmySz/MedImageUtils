import os
import pandas as pd

# 路径配置
CSV_PATH = r"D:\Data\Jmszxyy\骨松四分类\Dataset\Swin_MAE_Pretrain\swin_mae_metadata.csv"
PATCH_DIR = r"D:\Data\Jmszxyy\骨松四分类\Dataset\Swin_MAE_Pretrain\Patches"
CLEAN_CSV_PATH = r"D:\Data\Jmszxyy\骨松四分类\Dataset\Swin_MAE_Pretrain\swin_mae_metadata_clean.csv"


def clean_and_align_dataset():
    print("1. 正在读取原始 CSV 表格...")
    df = pd.read_csv(CSV_PATH)
    initial_count = len(df)
    print(f"   初始记录数: {initial_count} 行")

    # 【关键修复】：如果 Patch_ID 结尾没有 .png，我们自动帮它加上
    # 这样既修复了比对问题，也方便后续 DataLoader 直接拼接路径读取图片
    if not df['Patch_ID'].iloc[0].endswith('.png'):
        print("   检测到表格中缺失 .png 后缀，正在自动补全...")
        df['Patch_ID'] = df['Patch_ID'].astype(str) + '.png'

    print("\n2. 正在清除 CSV 中的重复记录...")
    df = df.drop_duplicates(subset=['Patch_ID'], keep='last')
    dedup_count = len(df)
    print(f"   去重后剩余: {dedup_count} 行")

    print("\n3. 正在交叉验证硬盘上的实际图片文件...")
    actual_files = set(os.listdir(PATCH_DIR))
    print(f"   硬盘实际读取到文件数: {len(actual_files)} 个")

    # 过滤比对
    df_clean = df[df['Patch_ID'].isin(actual_files)]

    final_count = len(df_clean)
    print(f"\n4. 清洗完成！")
    print(f"   最终保留有效记录: {final_count} 行")

    # 强制保存为 CSV 格式以保证深度学习读取速度
    df_clean.to_csv(CLEAN_CSV_PATH, index=False)
    print(f"   已生成绝对对齐的干净 CSV: {CLEAN_CSV_PATH}")


if __name__ == '__main__':
    clean_and_align_dataset()