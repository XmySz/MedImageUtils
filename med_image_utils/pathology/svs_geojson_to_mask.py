import json
import numpy as np
import cv2
import openslide
from pathlib import Path


def load_geojson(geojson_path):
    with open(geojson_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("features", [])


def filter_polygons_by_type(features, target_cell_type, use_type_str=False):
    """
    根据 features，筛选指定类型细胞的 polygon。

    target_cell_type: 比如 'epithelial' / 'CD8' 等
    use_type_str: 如果想用 properties['type_str'] 来判断，就设 True
    返回：list[np.ndarray]，每个元素形状为 (N, 2)，代表一个 polygon 顶点集合（x, y）
    """
    polygons = []

    for feat in features:
        geom = feat.get("geometry", {})
        props = feat.get("properties", {})

        # 1) 先判断细胞类型
        if use_type_str:
            cell_type = props.get("type_str", None)
        else:
            cls = props.get("classification", {})
            cell_type = cls.get("name", None)

        if cell_type != target_cell_type:
            continue

        # 2) 再看 geometry 类型
        gtype = geom.get("type", None)
        coords = geom.get("coordinates", [])

        if gtype == "Polygon":
            # coords: [ [ [x1, y1], [x2, y2], ... ] ]
            if not coords:
                continue
            outer_ring = coords[0]  # 一般第 0 个是外轮廓
            poly = np.array(outer_ring, dtype=np.float32)  # (N, 2)
            polygons.append(poly)

        elif gtype == "MultiPolygon":
            # 如果有 MultiPolygon，就把每个 polygon 都加入
            for poly_coords in coords:
                if not poly_coords:
                    continue
                outer_ring = poly_coords[0]
                poly = np.array(outer_ring, dtype=np.float32)
                polygons.append(poly)

        # 如果还有其他类型（Point/LineString），这里就直接忽略了

    return polygons


def create_mask_for_cells(
        wsi_path,
        geojson_path,
        target_cell_type,
        level=2,
        output_path="mask.png",
        use_type_str=False
):
    slide = openslide.OpenSlide(str(wsi_path))

    base_w, base_h = slide.level_dimensions[0]
    if level >= slide.level_count:
        raise ValueError(f"指定的 level={level} 超出范围，WSI 只有 {slide.level_count} 层")

    low_w, low_h = slide.level_dimensions[level]
    scale_x = low_w / base_w
    scale_y = low_h / base_h

    print(f"Base level size: {base_w} x {base_h}")
    print(f"Target level {level} size: {low_w} x {low_h}")
    print(f"Scale factors: x={scale_x:.4f}, y={scale_y:.4f}")

    # 1. 读取并筛选 polygon
    features = load_geojson(geojson_path)
    cell_polygons = filter_polygons_by_type(
        features,
        target_cell_type=target_cell_type,
        use_type_str=use_type_str
    )

    print(f"Found {len(cell_polygons)} polygons of type '{target_cell_type}'.")

    # 2. 创建 mask
    mask = np.zeros((low_h, low_w), dtype=np.uint8)

    # 3. 把 polygon 从 base level 缩放到低倍率层，并填充
    for poly in cell_polygons:
        # poly: (N, 2) -> 缩放
        poly_low = poly.copy()
        poly_low[:, 0] = poly_low[:, 0] * scale_x
        poly_low[:, 1] = poly_low[:, 1] * scale_y

        poly_low_int = np.round(poly_low).astype(np.int32)

        # cv2.fillPoly 需要 [poly] 的形式
        cv2.fillPoly(mask, [poly_low_int], 255)

    cv2.imwrite(str(output_path), mask)
    print(f"Mask saved to: {output_path}")


if __name__ == "__main__":
    # ==== 根据你自己的情况修改这里 ====
    wsi_path = Path(r"F:\内膜\内膜癌svs文件\201121909002.svs")  # HE 染色的 WSI
    geojson_path = Path(r"D:\Data\Temp\WSIout\json\201121909002.geojson")  # 对应的 GeoJSON
    target_cell_type = "lymphocyte"  # 想要生成 mask 的免疫细胞类型
    level = 0  # 低倍率层（可以改成 1、2、3 等看看尺寸）
    output_path = "mask_level2.png"  # 输出文件名

    create_mask_for_cells(
        wsi_path=wsi_path,
        geojson_path=geojson_path,
        target_cell_type=target_cell_type,
        level=level,
        output_path=output_path,
    )