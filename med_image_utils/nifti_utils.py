import glob
import gzip
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable

import SimpleITK as sitk
import nibabel as nib


def check_and_adjust_properties(imagesTr, labelsTr, save_dir=None):
    """
    检查并调整图像和标签的属性（原点、方向、spacing）

    Args:
        imagesTr (str): 图像目录路径
        labelsTr (str): 标签目录路径
        save_dir (str, optional): 调整后的标签保存目录，如果为None则覆盖原文件
    """
    images = sorted(glob.glob(str(Path(imagesTr) / '*.nii.gz')))
    labels = sorted(glob.glob(str(Path(labelsTr) / '*.nii.gz')))

    if len(images) != len(labels):
        raise ValueError(f"图像数量({len(images)})和标签数量({len(labels)})不匹配")

    if save_dir: os.makedirs(save_dir, exist_ok=True)

    for i, (image_path, label_path) in enumerate(zip(images, labels), 1):
        print(f"\n处理第{i}对图像和标签...")
        print(f"图像: {image_path}")
        print(f"标签: {label_path}")

        try:
            image = sitk.ReadImage(image_path)
            label = sitk.ReadImage(label_path)

            img_origin = image.GetOrigin()
            img_direction = image.GetDirection()
            img_spacing = image.GetSpacing()

            label_origin = label.GetOrigin()
            label_direction = label.GetDirection()
            label_spacing = label.GetSpacing()

            # 检查是否需要调整
            need_adjustment = False

            if img_origin != label_origin:
                print(f"原点不匹配:")
                print(f"图像原点: {img_origin}")
                print(f"标签原点: {label_origin}")
                need_adjustment = True

            if img_direction != label_direction:
                print(f"方向不匹配:")
                print(f"图像方向: {img_direction}")
                print(f"标签方向: {label_direction}")
                need_adjustment = True

            if img_spacing != label_spacing:
                print(f"像素间距不匹配:")
                print(f"图像间距: {img_spacing}")
                print(f"标签间距: {label_spacing}")
                need_adjustment = True

            if need_adjustment:
                print("调整标签属性...")
                resample = sitk.ResampleImageFilter()
                resample.SetReferenceImage(image)  # 使用图像作为参考

                # 设置插值方法（对于标签图像使用最近邻插值）
                resample.SetInterpolator(sitk.sitkNearestNeighbor)

                # 执行重采样
                adjusted_label = resample.Execute(label)

                # 保存调整后的标签
                if save_dir:
                    output_path = str(Path(save_dir) / Path(label_path).name)
                else:
                    output_path = label_path

                sitk.WriteImage(adjusted_label, output_path)
                print(f"已保存调整后的标签到: {output_path}")
            else:
                print("属性匹配，无需调整")

        except Exception as e:
            print(f"处理失败: {str(e)}")
            raise


def repair_nifti_files(directory):
    """
    检查并修复目录下所有.nii.gz文件的压缩问题。

    该函数遍历指定目录下的所有.nii.gz文件。它首先尝试使用nibabel
    加载每个文件以验证其完整性。如果加载失败，函数会假定文件可能
    存在Gzip压缩层面的损坏或不兼容问题，并尝试通过解压再重新压缩
    的方式来修复它。

    """
    dir_path = Path(directory)

    nifti_files = list(dir_path.glob("*.nii.gz"))

    print(f"开始处理目录 '{directory}' 中的 {len(nifti_files)} 个文件...")

    for file_path in nifti_files:
        try:
            _ = nib.load(file_path)
            print(f"✅ 成功读取: {file_path.name}")

        except Exception as e:
            print(f"⚠️ 处理文件出错 {file_path.name}: {e}")
            print(f"   ...尝试重新压缩文件...")

            temp_path = file_path.with_suffix('.nii.temp')

            try:
                with gzip.open(file_path, 'rb') as f_in:
                    with open(temp_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                with open(temp_path, 'rb') as f_in:
                    with gzip.open(file_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                print(f"   ✔️ 已重新压缩: {file_path.name}")

            except Exception as e_recompress:
                print(f"   ❌ 重新压缩失败 {file_path.name}: {e_recompress}")

            finally:
                # 清理临时文件，无论成功与否都要执行
                if temp_path.exists():
                    temp_path.unlink()


def Nii2Nifti() -> None:
    """
        nii格式转为nii.gz格式, 注意会删除原本的nii格式文件, 替代为新的nii.gz格式文件
    """
    def iterate_nii_files(root: Path) -> Iterable[Path]:
        """递归获取所有扩展名为 .nii 的文件，忽略已压缩的 .nii.gz。"""
        yield from (p for p in root.rglob("*.nii") if not p.name.endswith(".nii.gz"))

    def compress_nii(nii_path: Path) -> None:
        """将 .nii 压缩为 .nii.gz，成功后删除原文件。"""
        gz_path = nii_path.with_suffix(".nii.gz")
        if gz_path.exists():
            print(f"[跳过] 目标文件已存在: {gz_path}", file=sys.stderr)
            return

        print(f"[开始] 压缩 {nii_path} -> {gz_path}")
        try:
            with nii_path.open("rb") as src, gzip.open(gz_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            nii_path.unlink()  # 压缩成功再删除原文件
            print(f"[完成] {gz_path}")
        except Exception as exc:
            if gz_path.exists():
                gz_path.unlink(missing_ok=True)
            print(f"[错误] 压缩失败 {nii_path}: {exc}", file=sys.stderr)

    ROOT_DIR = Path(r"D:\Data\OvarianCancer\Datasets\CA_hos_Label\Original")

    if not ROOT_DIR.exists():
        print(f"[致命] 指定根目录不存在: {ROOT_DIR}", file=sys.stderr)
        return

    for nii in iterate_nii_files(ROOT_DIR):
        compress_nii(nii)


if __name__ == '__main__':
    check_and_adjust_properties(r'D:\Data\OvarianCancer\Datasets\CA_hos_Label\T2\Images',
                                r'D:\Data\OvarianCancer\Datasets\CA_hos_Label\T2\Labels')
