import os
import shutil
from pathlib import Path

import numpy as np
import cv2


def find_top_n_lymphocyte_patches(
    input_dir,
    output_dir,
    n=50,
    mask_suffix="_mask.png",
    image_exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    copy_masks=True,
):
    """
    从 input_dir 中找到含“淋巴细胞”(mask非零像素)最多的前 n 个 patch，
    将对应的原图（以及可选的 mask）复制到 output_dir。

    参数
    ----
    input_dir : str or Path
        存放 patch 图像和 *_mask.png 的目录
    output_dir : str or Path
        要复制到的新目录（不存在会自动创建）
    n : int
        选出前 n 张含淋巴细胞最多的 patch
    mask_suffix : str
        mask 文件名后缀，默认 "_mask.png"
    image_exts : tuple
        可能的原图扩展名列表
    copy_masks : bool
        是否同时复制对应的 mask
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 遍历 mask 文件，统计“淋巴细胞面积”（非零像素数）
    stats = []  # 每个元素: (非零像素数, mask_path, base_name)

    for mask_path in input_dir.glob(f"*{mask_suffix}"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"警告：无法读取 mask 文件：{mask_path}")
            continue

        lymph_area = int(np.count_nonzero(mask))
        base_name = mask_path.name.replace(mask_suffix, "")  # 对应原图的“主名”

        stats.append((lymph_area, mask_path, base_name))

    if not stats:
        print("在输入目录中没有找到任何 mask 文件。")
        return

    # 2. 按淋巴细胞面积从大到小排序，取前 n 个
    stats.sort(key=lambda x: x[0], reverse=True)
    top_stats = stats[:n]

    print(f"在目录 {input_dir} 中共找到 {len(stats)} 个 mask 文件。")
    print(f"将选出含淋巴细胞最多的前 {len(top_stats)} 个 patch。")

    # 3. 复制对应原图及 mask
    num_copied = 0

    for lymph_area, mask_path, base_name in top_stats:
        # 查找对应原图 patch
        src_image_path = None
        for ext in image_exts:
            candidate = input_dir / f"{base_name}{ext}"
            if candidate.exists():
                src_image_path = candidate
                break

        if src_image_path is None:
            print(f"警告：未找到与 {mask_path.name} 对应的原图 patch，跳过。")
            continue

        # 目标路径
        dst_image_path = output_dir / src_image_path.name
        shutil.copy2(src_image_path, dst_image_path)

        if copy_masks:
            dst_mask_path = output_dir / mask_path.name
            shutil.copy2(mask_path, dst_mask_path)

        num_copied += 1
        print(
            f"已复制：{src_image_path.name}  (淋巴细胞像素数={lymph_area})"
        )

    print(
        f"\n完成！共复制 {num_copied} 个 patch 到 {output_dir} "
        f"(每个附带 mask={copy_masks})."
    )


if __name__ == "__main__":
    # ======= 根据你的实际路径修改下面这几行 =======
    input_dir = r"D:\Data\Temp\test"        # 之前 wsi_to_patch 的输出目录
    output_dir = r"D:\Data\Temp\top_lymph"   # 想保存“淋巴细胞最多”的图像的目录
    n = 50                                  # 例如取前 100 张

    find_top_n_lymphocyte_patches(
        input_dir=input_dir,
        output_dir=output_dir,
        n=n,
        mask_suffix="_mask.png",  # 和 wsi_to_patch 里保持一致
        copy_masks=True,          # 如果只想复制原图，可以改为 False
    )
