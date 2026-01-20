import glob
import os
import shutil
from pathlib import Path
from typing import Optional, Union, List, Tuple, Literal
from pypinyin import lazy_pinyin

import pandas as pd


def StandardizeNaming(dir_path, mapping_excel=None, train=True, prefix="breast", start_index=1, reverse=False):
    """
        用于批量、标准化地重命名指定目录下的文件（特别是.nii.gz格式的医学影像），使其符合统一的命名规范（如prefix_XXX_0000.nii.gz），同时能生成一个映射文件用于追溯和撤销操作。
    """
    if reverse and mapping_excel:
        mapping_df = pd.read_excel(mapping_excel)
        for _, row in mapping_df.iterrows():
            old_path = os.path.join(dir_path, row['New_Name'])
            new_path = os.path.join(dir_path, row['Original_Name'])
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
        return

    files = sorted(glob.glob(dir_path + '/*'))
    mapping = []

    for index, file in enumerate(files, start_index):
        old_name = os.path.basename(file)
        new_name = f"{prefix}_{index:>03}_0000.{file[file.index('.') + 1:]}" if train else f"{prefix}_{index:>03}.{file[file.index('.') + 1:]}"
        os.rename(file, os.path.join(dir_path, new_name))
        mapping.append([old_name, new_name])

    if mapping_excel:
        pd.DataFrame(mapping, columns=['Original_Name', 'New_Name']).to_excel(mapping_excel, index=False)


def find_files_to_table(root_dir: str, file_extensions: Optional[Union[str, List[str], Tuple[str, ...]]] = None):
    """
    遍历指定的根目录及其所有子目录，查找指定后缀名或所有文件，输出文件路径映射到表格中

    参数:
    root_dir (str): 要开始搜索的根目录的路径。
    file_extensions (str, list, tuple, optional):
        - 单个格式: '.txt'
        - 多种格式: ['.jpg', '.png']
        - 所有文件: None (默认)
        - 格式匹配不区分大小写。
    返回:
    pandas.DataFrame: 一个包含'文件名'和'路径'两列的DataFrame。
    """
    if not os.path.isdir(root_dir):
        print(f"错误：目录 '{root_dir}' 不存在或不是一个有效的目录。")
        return pd.DataFrame(columns=['文件名', '路径'])

    search_all_files = False
    processed_extensions = None

    if file_extensions is None or not file_extensions:
        search_all_files = True
        search_message = "所有文件"
    elif isinstance(file_extensions, str):
        processed_extensions = (file_extensions.lower(),)
        search_message = f"'{file_extensions}' 文件"
    elif isinstance(file_extensions, (list, tuple)):
        processed_extensions = tuple(ext.lower() for ext in file_extensions)
        search_message = f"{', '.join(file_extensions)} 文件"
    else:
        print(f"错误：'file_extensions' 参数类型无效。")
        return pd.DataFrame(columns=['文件名', '路径'])

    found_files_data = []
    print(f"开始在 '{root_dir}' 目录中搜索 {search_message}...")

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if search_all_files or filename.lower().endswith(processed_extensions):
                full_path = os.path.join(dirpath, filename)
                found_files_data.append({
                    '文件名': filename,
                    '路径': full_path
                })

    if found_files_data:
        print(f"搜索完成！共找到 {len(found_files_data)} 个匹配的文件。")
        pd.DataFrame(found_files_data).to_excel(r"文件列表.xlsx", index=False, engine='openpyxl')
    else:
        print("搜索完成！未找到任何匹配的文件。")


def move_matching_files(dir1, dir2, dir3):
    """
        读取目录1下所有的文件 如果这个文件在目录2也存在的话 就移动到目录3
    Args:
        dir1:
        dir2:
        dir3:

    Returns:

    """
    for file1 in os.listdir(dir1):
        name1 = os.path.splitext(file1)[0]
        for file2 in os.listdir(dir2):
            name2 = os.path.splitext(file2)[0]
            if name1 == name2:
                shutil.move(os.path.join(dir1, file1), os.path.join(dir3, file1))
                break


def transfer_files_from_excel(
        excel_path: str,
        column_name: str,
        src_dir: str,
        dst_dir: str,
        action: Literal["copy", "move"] = "copy",
) -> None:
    df = pd.read_excel(excel_path)

    # Excel里可能写“abc”或“abc.svs”，统一取不带后缀名
    stems = {Path(str(x)).stem for x in df[column_name].dropna()}

    # 全部转为小写
    stems = [i.lower() for i in stems]

    os.makedirs(dst_dir, exist_ok=True)
    op = shutil.copy2 if action == "copy" else shutil.move

    for name in os.listdir(src_dir):
        print(name)
        p = Path(src_dir) / name
        if p.is_file() and p.stem.lower() in stems:
            op(str(p), str(Path(dst_dir) / p.name))


def compare_directories(dir1: str, dir2: str) -> tuple[set[str], set[str]]:
    """
    比较两个目录的文件（忽略扩展名）

    参数:
        dir1: 目录1的路径
        dir2: 目录2的路径

    返回:
        (只在目录1中的文件集合, 只在目录2中的文件集合)
    """
    files1 = {f.stem for f in Path(dir1).iterdir() if f.is_file()}
    files2 = {f.stem for f in Path(dir2).iterdir() if f.is_file()}

    only_in_dir1 = files1 - files2
    only_in_dir2 = files2 - files1

    return only_in_dir1, only_in_dir2


def rename_files_from_excel(directory: str, excel_file: str, col_old: str, col_new: str) -> dict[str, str]:
    """
    根据Excel中的映射关系重命名目录下的文件

    参数:
        directory: 目标目录路径
        excel_file: Excel文件路径
        col_old: 原文件名列名
        col_new: 新文件名列名

    返回:
        重命名成功的文件映射字典 {原文件名: 新文件名}
    """
    df = pd.read_excel(excel_file)
    dir_path = Path(directory)

    # 创建映射字典（去掉后缀）
    name_map = {}
    for _, row in df.iterrows():
        old_name = Path(str(row[col_old])).stem
        new_name = Path(str(row[col_new])).stem
        name_map[old_name] = new_name

    renamed = {}

    for file in dir_path.iterdir():
        if not file.is_file():
            continue

        file_stem = file.stem

        if file_stem in name_map:
            new_name = name_map[file_stem]
            new_file = dir_path / f"{new_name}{file.suffix}"

            # 检查目标文件是否已存在
            if new_file.exists():
                print(f"跳过：{file.name} -> {new_file.name}（目标文件已存在）")
                continue

            # 检查是否会造成重复
            if new_name in renamed.values():
                print(f"跳过：{file.name} -> {new_file.name}（会造成重复）")
                continue

            file.rename(new_file)
            renamed[file.name] = new_file.name
            print(f"重命名：{file.name} -> {new_file.name}")

    return renamed


def export_files_to_excel(directory: str, output_file: str) -> None:
    """
    将目录下所有文件的名称和路径导出到Excel

    参数:
        directory: 要扫描的目录路径
        output_file: 输出的Excel文件路径
    """
    dir_path = Path(directory)

    files_data = []
    for file in dir_path.rglob('*'):
        if file.is_file():
            files_data.append({
                '文件名': file.name,
                '完整路径': str(file.absolute()),
            })

    df = pd.DataFrame(files_data)
    df.to_excel(output_file, index=False)
    print(f"已导出 {len(files_data)} 个文件到 {output_file}")


def process_excel_pinyin(input_path: str, output_path: str, col_name: str = '姓名') -> None:
    """
    读取Excel文件，为指定列添加对应的拼音列（首字母大写，空格分隔），并保存结果。

    Args:
        input_path: 输入Excel文件的路径
        output_path: 输出Excel文件的路径
        col_name: 需要转换的姓名列列名，默认为'姓名'
    """
    df = pd.read_excel(input_path)

    # 将指定列转换为拼音，处理非字符串情况
    df['拼音'] = df[col_name].apply(lambda x: ' '.join(lazy_pinyin(str(x))).title())

    df.to_excel(output_path, index=False)


def filter_table_by_filenames(table_path, col_name, dir_path, output_path):
    """
    读取表格，筛选出指定列的值与目录下文件名（去除后缀）匹配的行，并保存。

    Args:
        table_path (str): 输入表格路径 (.xlsx, .xls 或 .csv)
        col_name (str): 表格中用于匹配的列名
        dir_path (str): 包含文件的文件夹路径
        output_path (str): 输出表格的保存路径
    """
    # 读取数据，强制将匹配列转为字符串，防止数字/字符串类型不匹配
    if table_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(table_path, dtype={col_name: str})
    else:
        df = pd.read_csv(table_path, dtype={col_name: str}, )

    # 获取目录下所有文件名的主名（去除后缀），使用 set 加速查找
    # 例如：'abc.jpg' -> 'abc'
    file_names = {os.path.splitext(f)[0].lower() for f in os.listdir(dir_path)}

    # 使用 isin 筛选出列值存在于文件名集合中的行
    df_filtered = df[df[col_name].str.lower().isin(file_names)]

    # 保存结果
    if output_path.endswith(('.xlsx', '.xls')):
        df_filtered.to_excel(output_path, index=False)
    else:
        df_filtered.to_csv(output_path, index=False)


def filter_table_by_reference_table(target_table_path, ref_table_path, col_name, output_path):
    """
    读取目标表格，筛选出指定列的值存在于参考表格同一列名中的行，并保存。

    Args:
        target_table_path (str): 需要筛选的主表格路径 (.xlsx, .xls 或 .csv)
        ref_table_path (str): 提供筛选标准的参考表格路径 (.xlsx, .xls 或 .csv)
        col_name (str): 两个表格中用于匹配的共同列名
        output_path (str): 输出表格的保存路径
    """
    try:
        # 1. 读取两个表格
        # 强制转换为字符串类型，避免 "001" 和 1 无法匹配的问题
        df_target = pd.read_excel(target_table_path, dtype={col_name: str}, sheet_name='CenterA(jm)')
        df_ref = pd.read_excel(ref_table_path, dtype={col_name: str}, sheet_name='仅张博')

        # 2. 获取参考表中的所有唯一值
        # 预处理：转字符串 -> 去除首尾空格 -> 转小写 (根据需求可去掉 .lower()) -> 存入 set 加速查找
        ref_values = set(
            df_ref[col_name]
            .dropna()  # 去除空值
            .astype(str)  # 确保是字符串
            .str.strip()  # 去除首尾空格
            .str.lower()  # 统一转小写进行匹配（忽略大小写）
        )

        # 3. 筛选目标表
        # 对目标表的列也做相同的预处理（转小写、去空格），然后判断是否在 ref_values 中
        # 注意：这里我们只用处理后的值来生成布尔索引，不修改原数据
        mask = (
            df_target[col_name]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(ref_values)
        )
        print(mask)

        df_filtered = df_target[mask]

        # 4. 保存结果
        print(f"筛选完成：原数据 {len(df_target)} 行，筛选后 {len(df_filtered)} 行。")

        if output_path.endswith(('.xlsx', '.xls')):
            df_filtered.to_excel(output_path, index=False)
        else:
            df_filtered.to_csv(output_path, index=False)

    except KeyError as e:
        print(f"错误：列名 {e} 在某个表格中不存在，请检查列名拼写。")
    except Exception as e:
        print(f"发生错误：{e}")


def delete_unmatched_files(table_path, col_name, dir_path):
    """
    读取表格指定列的值，遍历目录下所有文件，删除文件名（不含后缀）不在表格列中的文件。

    Args:
        table_path (str): 表格路径 (.xlsx, .xls 或 .csv)
        col_name (str): 表格中包含保留文件名的列名
        dir_path (str): 目标文件夹路径
    """
    # 读取表格，强制将指定列读取为字符串以确保匹配准确
    if table_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(table_path, dtype={col_name: str})
    else:
        df = pd.read_csv(table_path, dtype={col_name: str})

    # 获取白名单集合（去除空值）
    valid_names = set(df[col_name].dropna().astype(str))

    # 遍历目录
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)

        # 确保是文件而不是子文件夹
        if os.path.isfile(file_path):
            # 获取不带后缀的文件名
            name_stem = os.path.splitext(filename)[0]

            # 如果文件名不在白名单中，则删除
            if name_stem not in valid_names:
                os.remove(file_path)
                print(f"已删除: {filename}")


def rename_files_to_pinyin(dir_path: str) -> None:
    """
    遍历指定目录，将文件名中的中文转换为空格分隔的拼音，并保留原后缀。

    Args:
        dir_path (str): 目标文件夹路径
    """
    for filename in os.listdir(dir_path):
        name, ext = os.path.splitext(filename)

        # 将文件名转换为拼音列表并用空格拼接 (非中文部分保持原样)
        pinyin_name = ' '.join(lazy_pinyin(name)) + ext

        old_path = os.path.join(dir_path, filename)
        new_path = os.path.join(dir_path, pinyin_name)

        # 仅当文件名发生变化时执行重命名
        if old_path != new_path and not os.path.exists(new_path):
            os.rename(old_path, new_path)
        else:
            print(f'{old_path} 未完成重命名')


def rename_files_to_title_case(dir_path: str) -> None:
    """
    遍历指定目录，将文件名中的单词转为首字母大写 (Title Case)。
    例如: 'LIANG HUA.png' -> 'Liang Hua.png'

    Args:
        dir_path (str): 目标文件夹路径
    """
    for filename in os.listdir(dir_path):
        name, ext = os.path.splitext(filename)

        # 将文件名转为每个单词首字母大写
        new_name = name.title() + ext

        old_path = os.path.join(dir_path, filename)
        new_path = os.path.join(dir_path, new_name)

        if old_path != new_path:
            os.rename(old_path, new_path)


def merge_tables(table1, table2, key_column, new_column_name, save_path):
    """
    根据指定列名匹配两个表格,给表格1添加新列

    参数:
        table1: DataFrame 或文件路径 - 第一个表格
        table2: DataFrame 或文件路径 - 第二个表格
        key_column: str - 用于匹配的列名
        new_column_name: str - 要从表格2添加到表格1的列名

    返回:
        DataFrame - 添加了新列的表格1
    """
    df1 = pd.read_excel(table1, dtype={key_column: str}, sheet_name='仅张博')
    df2 = pd.read_excel(table2, dtype={key_column: str})

    # 只选择需要的列进行合并
    df2_subset = df2[[key_column, new_column_name]]

    # 使用左连接,保留表格1的所有行
    result = df1.merge(df2_subset, on=key_column, how='left')

    result.to_excel(save_path, index=False)


def create_folders_from_excel(excel_path: str, target_dir: str, col_name: str) -> None:
    """
    读取Excel指定列的内容，并在目标目录下创建对应名称的文件夹。

    Args:
        excel_path: Excel文件路径。
        target_dir: 目标文件夹路径（如果不存在会自动创建）。
        col_name: 用于生成文件夹名的列标题。
    """
    df = pd.read_excel(excel_path)

    # 提取列数据：去空值 -> 转字符串 -> 去首尾空格 -> 去重
    names = df[col_name].dropna().astype(str).str.strip().unique()

    for name in names:
        if name:  # 确保不是空字符串
            os.makedirs(os.path.join(target_dir, name), exist_ok=True)


if __name__ == '__main__':
    # rename_files_from_excel(r'D:\Data\Jmszxyy\骨松四分类\Dataset\新补充_筛选\英文名\已匹配到',
    #                         r'D:\Data\Jmszxyy\骨松四分类\Dataset\新补充_筛选\中文名.xlsx', '姓名拼音', '姓名')

    # export_files_to_excel(r'E:\胃癌\胃癌SVS文件\表格无对应', r'E:\胃癌\胃癌SVS文件\表格无对应.xlsx')

    # StandardizeNaming(r'C:\Users\Administrator\Desktop\OCT\外部验证组（D）\图', prefix='oct')

    # transfer_files_from_excel(r'D:\Data\Jmszxyy\骨松四分类\Dataset\工作表_1.xlsx', column_name='姓名拼音',
    #                           src_dir=r'D:\Data\Jmszxyy\骨松四分类\Dataset\新补充_筛选\英文名\未匹配到', dst_dir=r'D:\Data\Jmszxyy\骨松四分类\Dataset\新补充_筛选\英文名\已匹配到1', action='move')

    # process_excel_pinyin(r"D:\Data\Jmszxyy\骨松四分类\Dataset\工作表.xlsx", r"D:\Data\Jmszxyy\骨松四分类\Dataset\工作表_1.xlsx")

    # print(compare_directories(r'F:\EC\WSI\最终可用', r'Z:\Zyn\PyCharmProjects\Jmzxyy\WSI_Segmenter-master\WSIs'))

    # delete_unmatched_files(r'D:\Data\Jmszxyy\骨松四分类\Dataset\新补充_筛选\英文名.xlsx', '姓名拼音',
    #                           r'D:\Data\Jmszxyy\骨松四分类\Dataset\新补充_筛选\英文名\已匹配到')

    # filter_table_by_filenames(r'C:\Users\Administrator\Desktop\manifest-1754069116660\clinical.xlsx',
    #                           'cases.submitter_id',
    #                           r'C:\Users\Administrator\Desktop\manifest-1754069116660\CPTAC-LSCC\CT',
    #                           r'C:\Users\Administrator\Desktop\manifest-1754069116660\CPTAC-LSCC\基因临床表.xlsx')

    # filter_table_by_reference_table(
    #                                 r'D:\Data\Jmszxyy\骨松四分类\Dataset\江门市中心医院\临床信息表汇总.xlsx',r'D:\Data\Jmszxyy\骨松四分类\Dataset\江门市中心医院\Labels.xlsx',
    #                                 '住院号',
    #                                 r'D:\Data\Jmszxyy\骨松四分类\Dataset\江门市中心医院\Labels1.xlsx')

    merge_tables(r'D:\Data\Jmszxyy\骨松四分类\Dataset\江门市中心医院\Labels.xlsx',
                 r'D:\Data\Jmszxyy\骨松四分类\Dataset\江门市中心医院\Labels1.xlsx',
                 '住院号',
                 r'是否骨折',
                 r'D:\Data\Jmszxyy\骨松四分类\Dataset\江门市中心医院\labels2.xlsx')
