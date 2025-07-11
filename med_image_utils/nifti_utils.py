import glob
import gzip
import os
import shutil
from pathlib import Path

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


if __name__ == '__main__':
    pass
