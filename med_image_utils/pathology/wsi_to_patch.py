import os
import sys
import json

import openslide
import numpy as np
import cv2
from PIL import Image

# 屏蔽 openslide 的 stderr 噪声
original_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')


# ======================
# 基础工具函数（原有）
# ======================

def _get_base_magnification(slide):
    props = slide.properties
    for key in ['aperio.AppMag', 'openslide.objective-power']:
        if key in props:
            try:
                return float(props[key])
            except Exception:
                pass
    return None


def _choose_level_for_magnification(slide, target_magnification):
    """
    根据目标物镜倍率选择最合适的 level
    """
    base_mag = _get_base_magnification(slide)

    level_downsamples = [float(ds) for ds in slide.level_downsamples]
    if base_mag is None:
        # 如果读不到物镜倍率，就简单选与 level 0 尺寸最接近的 downsample
        diffs = [abs(ds - (base_mag or level_downsamples[0])) for ds in level_downsamples]
        best_level = int(np.argmin(diffs))
        print(f"警告：无法获取基础物镜倍率，导出 level 退化为 {best_level}")
        return best_level

    level_mags = [base_mag / ds for ds in level_downsamples]
    diffs = [abs(m - target_magnification) for m in level_mags]
    best_level = int(np.argmin(diffs))

    print("基础物镜倍率: {:.2f}x".format(base_mag))
    print("各 level 对应倍率:", ["{:.2f}x".format(m) for m in level_mags])
    print(f"目标倍率: {target_magnification}x -> 选择 level {best_level} (~{level_mags[best_level]:.2f}x)")

    return best_level


def _choose_level_for_intersection(slide, target_magnification=10):
    """
    为肿瘤 mask 与 tissue mask 求交集选择一个合适的 level。
    尽量选择倍率 <= target_magnification 且尽可能接近 target_magnification。
    """
    base_mag = _get_base_magnification(slide)
    level_downsamples = [float(ds) for ds in slide.level_downsamples]

    if base_mag is None:
        # 退化策略：用倒数第二层
        inter_level = max(slide.level_count - 2, 0)
        print(f"警告：无法获取基础物镜倍率，交集 level 退化为 {inter_level}")
        return inter_level

    level_mags = [base_mag / ds for ds in level_downsamples]

    candidates = [(i, m) for i, m in enumerate(level_mags) if m <= target_magnification]
    if candidates:
        inter_level, inter_mag = max(candidates, key=lambda x: x[1])
    else:
        inter_level, inter_mag = min(enumerate(level_mags), key=lambda x: x[1])

    print(
        f"交集 mask 使用 level {inter_level}，约 {inter_mag:.2f}x "
        f"(目标为 {target_magnification}x)"
    )
    return inter_level


def _prepare_tumor_mask(mask_path, slide):
    """
    读取整张肿瘤 mask，并匹配到最接近的 level。
    这里只负责读取和匹配，不做遍历逻辑。
    """
    if mask_path is None or not os.path.isfile(mask_path):
        print("提示：未使用肿瘤 mask。")
        return None

    try:
        mask_img = Image.open(mask_path).convert("L")
    except Exception as e:
        print(f"错误：无法读取肿瘤 mask 文件: {mask_path}，错误信息: {e}")
        return None

    mask_w, mask_h = mask_img.size
    level_dims = slide.level_dimensions

    dists = []
    for i, (w, h) in enumerate(level_dims):
        d = abs(w - mask_w) + abs(h - mask_h)
        dists.append(d)
    mask_level = int(np.argmin(dists))
    level_w, level_h = level_dims[mask_level]

    scale_x = mask_w / level_w
    scale_y = mask_h / level_h
    ds_mask_level = float(slide.level_downsamples[mask_level])

    mask_arr = np.array(mask_img)

    print(
        f"肿瘤 mask 读取成功: {mask_w}x{mask_h}，匹配到 level {mask_level} "
        f"({level_w}x{level_h}, downsample={ds_mask_level:.2f})"
    )

    return {
        "mask": mask_arr,
        "level": mask_level,
        "ds": ds_mask_level,
        "w": mask_w,
        "h": mask_h,
        "scale_x": scale_x,
        "scale_y": scale_y,
    }


# ======================
# GeoJSON 相关新函数
# ======================

def _load_geojson_features(geojson_path):
    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("features", [])


def _filter_polygons_by_type(features, target_cell_type, use_type_str=False):
    """
    从 GeoJSON features 中筛选指定类型细胞的 polygon。
    结构参考 QuPath 导出的：
    properties:
        classification: {color: [r,g,b], name: 'epithelial'}
        type_str: 'epithelial'
    geometry:
        type: 'Polygon' / 'MultiPolygon'
        coordinates: [...]
    返回：list[np.ndarray]，每个元素为 (N,2) 的 (x, y) 顶点坐标（level 0）
    """
    polygons = []

    for feat in features:
        geom = feat.get("geometry", {})
        props = feat.get("properties", {})

        # 1) 判断细胞类型
        if use_type_str:
            cell_type = props.get("type_str", None)
        else:
            cls = props.get("classification", {})
            cell_type = cls.get("name", None)

        if cell_type != target_cell_type:
            continue

        # 2) 读取 geometry
        gtype = geom.get("type", None)
        coords = geom.get("coordinates", [])

        if gtype == "Polygon":
            if not coords:
                continue
            outer_ring = coords[0]
            poly = np.array(outer_ring, dtype=np.float32)
            polygons.append(poly)

        elif gtype == "MultiPolygon":
            for poly_coords in coords:
                if not poly_coords:
                    continue
                outer_ring = poly_coords[0]
                poly = np.array(outer_ring, dtype=np.float32)
                polygons.append(poly)

        # 其他类型（Point/LineString）忽略

    return polygons


def _prepare_geojson_polygons_for_level(slide, patch_level, geojson_path, target_cell_type,
                                        use_type_str=False):
    """
    把指定细胞类型的 polygon 从 level 0 缩放到 patch_level。
    返回结构：list[{"poly": np.ndarray (N,2), "bbox": (xmin, ymin, xmax, ymax)}]
    """
    if geojson_path is None or target_cell_type is None:
        return None

    print(f"加载 GeoJSON: {geojson_path}")
    features = _load_geojson_features(geojson_path)
    base_polygons = _filter_polygons_by_type(
        features, target_cell_type=target_cell_type, use_type_str=use_type_str
    )

    print(f"找到 {len(base_polygons)} 个 '{target_cell_type}' 的 polygon (level 0 坐标)。")

    if not base_polygons:
        return []

    base_w, base_h = slide.level_dimensions[0]
    low_w, low_h = slide.level_dimensions[patch_level]

    # 从 level 0 -> patch_level 的缩放比例
    scale_x = low_w / base_w
    scale_y = low_h / base_h

    polygons_level = []
    for poly in base_polygons:
        poly_low = poly.copy()
        poly_low[:, 0] = poly_low[:, 0] * scale_x
        poly_low[:, 1] = poly_low[:, 1] * scale_y

        xmin = float(poly_low[:, 0].min())
        xmax = float(poly_low[:, 0].max())
        ymin = float(poly_low[:, 1].min())
        ymax = float(poly_low[:, 1].max())

        polygons_level.append(
            {
                "poly": poly_low,
                "bbox": (xmin, ymin, xmax, ymax),
            }
        )

    print(f"已将 polygon 缩放到 level {patch_level}，用于 patch 级别 mask 生成。")
    return polygons_level


# ======================
# 主函数：导出 patch & 可选细胞 mask
# ======================

def export_patches_from_wsi(
        wsi_path,
        output_dir,
        patch_size=256,
        patch_format='png',
        target_magnification=20,
        tissue_threshold=0.5,
        thumb_output_dir=None,
        tumor_mask_path=None,
        # ---- 新增参数 ----
        geojson_path=None,
        target_cell_type=None,
        use_type_str=False,
):
    """
    从 WSI 导出 patch。
    - 如果提供 geojson_path 和 target_cell_type，则同时导出该类细胞的 mask patch。
    """
    try:
        slide = openslide.OpenSlide(wsi_path)

        # 1. 缩略图 / 前景检测（原逻辑）
        thumb_level = slide.level_count - 2
        thumb_w, thumb_h = slide.level_dimensions[thumb_level]
        print(f"前景检测使用 level {thumb_level}，尺寸: {thumb_w}x{thumb_h}")

        thumb_img = slide.read_region((0, 0), thumb_level, (thumb_w, thumb_h)).convert("RGB")
        thumb_np = np.array(thumb_img)

        gray = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        tissue_mask = 255 - mask

        kernel = np.ones((3, 3), np.uint8)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        if thumb_output_dir is not None:
            os.makedirs(thumb_output_dir, exist_ok=True)

            thumb_save_path = os.path.join(thumb_output_dir, f"thumbnail_level{thumb_level}.png")
            thumb_img.save(thumb_save_path)

            mask_img = Image.fromarray(tissue_mask)
            mask_save_path = os.path.join(thumb_output_dir, f"tissue_mask_level{thumb_level}.png")
            mask_img.save(mask_save_path)

            color_mask = cv2.applyColorMap(tissue_mask, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(thumb_np, 0.7, color_mask, 0.3, 0)
            overlay_img = Image.fromarray(overlay)
            overlay_save_path = os.path.join(
                thumb_output_dir, f"thumbnail_mask_overlay_level{thumb_level}.png"
            )
            overlay_img.save(overlay_save_path)

            print(f"缩略图及前景检测结果已保存到: {thumb_output_dir}")

        # 2. 读取肿瘤 mask（原逻辑）
        tumor_info = _prepare_tumor_mask(tumor_mask_path, slide)

        # 3. 选择 patch 的 level（原逻辑）
        patch_level = _choose_level_for_magnification(slide, target_magnification)
        level_w, level_h = slide.level_dimensions[patch_level]
        print(f"导出 patch 使用 level {patch_level}，尺寸: {level_w}x{level_h}")

        downsample_patch = float(slide.level_downsamples[patch_level])
        downsample_thumb = float(slide.level_downsamples[thumb_level])

        patch_size_0 = int(patch_size * downsample_patch)

        print(f"patch_size (level {patch_level}): {patch_size} 像素")
        print(f"对应在 level 0 上尺寸: {patch_size_0} 像素")

        W0, H0 = slide.level_dimensions[0]
        print(f"level 0 尺寸: {W0}x{H0}")

        # 4. 肿瘤 mask 与 tissue mask 求交集（原逻辑）
        intersection_mask = None
        intersection_level = None
        downsample_intersection = None
        inter_w = inter_h = None

        if tumor_info is not None:
            print("提示：存在肿瘤 mask，将在约 10x 的倍率下与前景 mask 求交集。")

            intersection_level = _choose_level_for_intersection(slide, target_magnification=10)
            inter_w, inter_h = slide.level_dimensions[intersection_level]
            downsample_intersection = float(slide.level_downsamples[intersection_level])

            # 上采样 tissue mask 到 intersection_level 尺度
            tissue_mask_inter = cv2.resize(
                tissue_mask,
                (inter_w, inter_h),
                interpolation=cv2.INTER_NEAREST
            )

            # 上采样 tumor mask 到 intersection_level 尺度
            tumor_mask_raw = (tumor_info["mask"] > 0).astype(np.uint8) * 255
            tumor_mask_inter = cv2.resize(
                tumor_mask_raw,
                (inter_w, inter_h),
                interpolation=cv2.INTER_NEAREST
            )

            # 取交集
            intersection_mask = cv2.bitwise_and(tissue_mask_inter, tumor_mask_inter)

            if thumb_output_dir is not None:
                inter_color = cv2.applyColorMap(intersection_mask, cv2.COLORMAP_JET)
                inter_img = Image.fromarray(inter_color)
                inter_save_path = os.path.join(
                    thumb_output_dir,
                    f"intersection_mask_level{intersection_level}.png"
                )
                inter_img.save(inter_save_path)
                print(f"肿瘤与前景交集 mask 已保存到: {inter_save_path}")
        else:
            print("提示：无肿瘤 mask，仅使用前景 (tissue) 掩膜进行筛选。")

        # 5. 如果需要细胞 mask，预处理 GeoJSON polygon -> patch_level
        polygons_for_patches = None
        if geojson_path is not None and target_cell_type is not None:
            polygons_for_patches = _prepare_geojson_polygons_for_level(
                slide,
                patch_level=patch_level,
                geojson_path=geojson_path,
                target_cell_type=target_cell_type,
                use_type_str=use_type_str,
            )
            if polygons_for_patches is None:
                polygons_for_patches = []
        else:
            print("提示：未提供 geojson_path 或 target_cell_type，不导出细胞 mask。")

        num_exported = 0
        patch_idx = 0

        y_range = range(0, H0 - patch_size_0 + 1, patch_size_0)
        x_range = range(0, W0 - patch_size_0 + 1, patch_size_0)

        total_candidates = len(y_range) * len(x_range)
        print(f"总共需要扫描 {total_candidates} 个候选位置")

        os.makedirs(output_dir, exist_ok=True)

        # 6. 遍历 level 0 网格，筛选 patch，并导出图像 + 可选 mask
        for y0 in y_range:
            for x0 in x_range:

                # --- 若 intersection_mask 存在，在交集 mask 上判断 ---
                if intersection_mask is not None:
                    mx0 = int(x0 / downsample_intersection)
                    my0 = int(y0 / downsample_intersection)
                    mx1 = int((x0 + patch_size_0) / downsample_intersection)
                    my1 = int((y0 + patch_size_0) / downsample_intersection)

                    mx1 = min(mx1, inter_w)
                    my1 = min(my1, inter_h)

                    if mx0 >= inter_w or my0 >= inter_h:
                        continue

                    mask_patch = intersection_mask[my0:my1, mx0:mx1]
                    if mask_patch.size == 0:
                        continue

                    valid_ratio = np.mean(mask_patch > 0)
                    if valid_ratio < tissue_threshold:
                        continue

                # --- 否则，使用 thumb_level 的 tissue_mask（原逻辑） ---
                else:
                    mx0 = int(x0 / downsample_thumb)
                    my0 = int(y0 / downsample_thumb)
                    mx1 = int((x0 + patch_size_0) / downsample_thumb)
                    my1 = int((y0 + patch_size_0) / downsample_thumb)

                    mx1 = min(mx1, thumb_w)
                    my1 = min(my1, thumb_h)

                    if mx0 >= thumb_w or my0 >= thumb_h:
                        continue

                    mask_patch = tissue_mask[my0:my1, mx0:mx1]
                    if mask_patch.size == 0:
                        continue

                    tissue_ratio = np.mean(mask_patch > 0)
                    if tissue_ratio < tissue_threshold:
                        continue

                # 满足条件，真正读 patch 图像
                try:
                    patch_img = slide.read_region(
                        (x0, y0),
                        patch_level,
                        (patch_size, patch_size)
                    ).convert("RGB")

                    patch_filename_base = f"patch_{patch_idx:06d}_x{x0}_y{y0}"
                    img_filename = f"{patch_filename_base}.{patch_format}"
                    img_output_path = os.path.join(output_dir, img_filename)

                    patch_img.save(img_output_path, format=patch_format.upper())
                    num_exported += 1

                    # ---------- 新功能：导出细胞 mask patch ----------
                    if polygons_for_patches is not None and len(polygons_for_patches) > 0:
                        # 当前 patch 在 patch_level 的坐标范围
                        x0_low = x0 / downsample_patch
                        y0_low = y0 / downsample_patch
                        x1_low = x0_low + patch_size
                        y1_low = y0_low + patch_size

                        cell_mask_patch = np.zeros((patch_size, patch_size), dtype=np.uint8)

                        # 遍历所有 polygon，选出 bbox 与当前 patch 有交集的
                        for item in polygons_for_patches:
                            poly = item["poly"]
                            xmin, ymin, xmax, ymax = item["bbox"]

                            # bbox 不相交则跳过
                            if xmax < x0_low or xmin > x1_low or ymax < y0_low or ymin > y1_low:
                                continue

                            # 把 polygon 坐标变到 patch 内局部坐标
                            poly_local = poly.copy()
                            poly_local[:, 0] = poly_local[:, 0] - x0_low
                            poly_local[:, 1] = poly_local[:, 1] - y0_low

                            poly_local_int = np.round(poly_local).astype(np.int32)

                            cv2.fillPoly(cell_mask_patch, [poly_local_int], 255)

                        # 保存细胞 mask patch
                        mask_filename = f"{patch_filename_base}_mask.png"
                        mask_output_path = os.path.join(output_dir, mask_filename)
                        cv2.imwrite(mask_output_path, cell_mask_patch)

                    patch_idx += 1

                except openslide.OpenSlideError as e:
                    print(f"警告：读取位置 (x0={x0}, y0={y0}), level={patch_level} 时出错: {e}。跳过此 patch。")
                except Exception as e:
                    print(f"错误：处理位置 (x0={x0}, y0={y0}) 时发生意外错误: {e}。跳过此 patch。")

        slide.close()
        print(f"\n完成！成功导出 {num_exported} 个 patches 到 {output_dir}")

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        if 'slide' in locals() and slide:
            slide.close()


if __name__ == '__main__':
    # ====== 示例用法，根据你自己的路径修改 ======
    wsi_path = r"F:\内膜\内膜癌svs文件\201121909002.svs"
    output_dir = r"D:\Data\Temp\test"
    thumb_output_dir = r"D:\Data\Temp\test_mask"
    tumor_mask_path = r"F:\内膜\201121909002.png"  # 可为 None

    # 新增：GeoJSON + target_cell_type（可选）
    geojson_path = r"D:\Data\Temp\WSIout\json\201121909002.geojson"  # 如果不需要细胞 mask，可设为 None
    target_cell_type = "lymphocyte"  # 例如 "epithelial" / "lymphocyte" 等

    export_patches_from_wsi(
        wsi_path=wsi_path,
        output_dir=output_dir,
        patch_size=512,
        patch_format='png',
        target_magnification=20,
        tissue_threshold=0.5,
        thumb_output_dir=thumb_output_dir,
        tumor_mask_path=tumor_mask_path,
        geojson_path=geojson_path,
        target_cell_type=target_cell_type,
        use_type_str=False,  # 若想用 properties['type_str'] 匹配，就设 True
    )
