import os
import sys

import h5py
import openslide
from tqdm import tqdm

original_stderr = sys.stderr  # 抑制警告信息
sys.stderr = open(os.devnull, 'w')


def export_patches_from_h5(wsi_path, h5_path, output_dir, patch_format='png',
                           force_patch_size=None, force_patch_level=None):
    """
    根据使用CLAM生成的h5坐标文件生成patch图像。

    Args:
        wsi_path (str): 原始 WSI 文件的路径 (.svs, .tif, etc.).
        h5_path (str): 包含 patch 坐标的 H5 文件的路径.
        output_dir (str): 保存导出 patch 图像的目标目录.
        patch_format (str): 保存 patch 的图像格式 ('png' 或 'jpg').
        force_patch_size (int, optional): 如果 H5 文件中没有 patch_size 信息，
                                          或者需要覆盖 H5 中的值，则手动指定 patch 大小.
        force_patch_level (int, optional): 如果 H5 文件中没有 patch_level 信息，
                                           或者需要覆盖 H5 中的值，则手动指定提取层级.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"将在以下目录中保存 patches: {output_dir}")

    try:
        slide = openslide.OpenSlide(wsi_path)

        with h5py.File(h5_path, 'r') as hf:
            if 'coords' not in hf:
                print(f"错误：在 H5 文件中未找到 'coords' 数据集: {h5_path}")
                slide.close()
                return

            coords = hf['coords'][:]  # 加载所有坐标到内存
            print(f"从 H5 文件加载了 {len(coords)} 个坐标。")

            # --- 获取 patch_size 和 patch_level ---
            patch_size = force_patch_size
            patch_level = force_patch_level

            # 优先尝试从 H5 属性读取 (如果存在且未被强制覆盖)
            if patch_size is None and 'patch_size' in hf['coords'].attrs:
                patch_size = int(hf['coords'].attrs['patch_size'])
                print(f"从 H5 属性读取 patch_size: {patch_size}")
            if patch_level is None and 'patch_level' in hf['coords'].attrs:
                patch_level = int(hf['coords'].attrs['patch_level'])
                print(f"从 H5 属性读取 patch_level: {patch_level}")

            # 如果仍然没有 patch_size 或 patch_level，则报错退出
            if patch_size is None:
                print("错误：无法确定 patch_size。请在 H5 文件中存储它，或使用 --patch-size 参数指定。")
                slide.close()
                return
            if patch_level is None:
                print("错误：无法确定 patch_level。请在 H5 文件中存储它，或使用 --patch-level 参数指定。")
                slide.close()
                return

            print(f"将使用 patch_size={patch_size}, patch_level={patch_level} 进行提取。")

            wsi_basename = os.path.splitext(os.path.basename(wsi_path))[0]
            num_exported = 0
            for i in tqdm(range(len(coords)), desc="导出 Patches"):
                x, y = int(coords[i, 0]), int(coords[i, 1])

                try:
                    patch_img = slide.read_region((x, y), patch_level, (patch_size, patch_size))

                    patch_img_rgb = patch_img.convert('RGB')

                    patch_filename = f"x{x}_y{y}.{patch_format}"
                    output_path = os.path.join(output_dir, patch_filename)

                    patch_img_rgb.save(output_path, format=patch_format.upper())
                    num_exported += 1

                except openslide.OpenSlideError as e:
                    print(f"\n警告：读取坐标 (x={x}, y={y}), level={patch_level} 时出错: {e}。跳过此 patch。")
                except Exception as e:
                    print(f"\n错误：处理坐标 (x={x}, y={y}) 时发生意外错误: {e}。跳过此 patch。")

        slide.close()
        print(f"\n完成！成功导出 {num_exported} / {len(coords)} 个 patches 到 {output_dir}")

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        if 'slide' in locals() and slide:
            slide.close()


if __name__ == '__main__':
    wsi_path = r"F:\Data\OvarianCancer\Data\temp\p.svs"
    h5_path = r"F:\Data\OvarianCancer\Data\Temp\116_p_level1\patches\p.h5"
    output_dir = r"F:\Data\OvarianCancer\Data\Temp\116_p_level1\images"

    export_patches_from_h5(
        wsi_path=wsi_path,
        h5_path=h5_path,
        output_dir=output_dir
    )
