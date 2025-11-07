import nibabel as nib
import numpy as np
from pathlib import Path

input_path = r'Z:\Zyn\PyCharmProjects\OvarianCancer\OvarianCancerRecurrencePrediction\Datasets\MRI_Normalized\T2\Images'  # 单个文件路径或目录路径
threshold = 10  # 小于此值的体素将被设为0


def normalize_nifti(nifti_path, threshold=0):
    """对单个nifti文件进行强度归一化"""
    img = nib.load(nifti_path)
    data = img.get_fdata()

    # 将小于阈值的体素设为0
    data[data < threshold] = 0

    # 在非0区域计算均值和标准差
    mask = data != 0
    if mask.sum() > 0:
        mean = data[mask].mean()
        std = data[mask].std()

        # 归一化：(data - mean) / std
        if std > 0:
            data[mask] = (data[mask] - mean) / std

    # 保存
    nib.save(nib.Nifti1Image(data, img.affine, img.header), nifti_path)
    print(f'已处理: {nifti_path}')


# 主程序
if __name__ == "__main__":
    path = Path(input_path)
    if path.is_file():
        normalize_nifti(path, threshold)
    elif path.is_dir():
        for nifti_file in path.glob('**/*.nii*'):
            normalize_nifti(nifti_file, threshold)
    else:
        print('路径不存在')
