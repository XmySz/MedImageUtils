import glob
import os

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


if __name__ == '__main__':
    pass
