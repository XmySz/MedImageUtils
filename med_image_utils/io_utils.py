import glob
import os
from typing import Optional, Union, List, Tuple

import pandas as pd


def StandardizeNaming(dir_path, mapping_excel=None, train=True, prefix="breast", start_index=1, reverse=False):
    """
        用于批量、标准化地重命名指定目录下的文件（特别是.nii.gz格式的医学影像），使其符合统一的命名规范（如prefix_XXX_0000.nii.gz），同时能生成一个映射文件用于追溯和撤销操作。
    :param dir_path:
    :param mapping_excel:
    :param train:
    :param prefix:
    :param start_index:
    :param reverse:
    :return:
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
        new_name = f"{prefix}_{index:>03}_0000.nii.gz" if train else f"{prefix}_{index:>03}.nii.gz"
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


if __name__ == '__main__':
    pass
