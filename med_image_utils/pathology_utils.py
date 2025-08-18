import os
import cv2
import staintools
from tqdm import tqdm
from natsort import natsorted
import argparse, os, json, numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
import geopandas as gpd
import openslide
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely import affinity
from rasterio import features


def wsi_stain_normalization(reference_image_path: str,
                            source_images_dir: str,
                            normalized_images_dir: str,
                            method: str = 'macenko'):
    """
    Batch stain-normalization for whole-slide images (WSI).

    Parameters
    ----------
    reference_image_path : str
        Path to the reference patch/slide used to fit the normalizer.
    source_images_dir : str
        Directory containing images to be normalized.
    normalized_images_dir : str
        Output directory for normalized images (will be created if absent).
    method : str, optional
        Stain-normalization algorithm: 'macenko', 'vahadane', or 'reinhard'
        (default 'macenko').

    Returns
    -------
    failed_files : list[tuple[str, str]]
        List of (image_path, error_message) for images that failed to process;
        empty list if all succeed.
    """

    # ---------- helpers ----------
    def _load_rgb(path: str):
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _save_rgb(img_rgb, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    def _build_normalizer(ref_rgb, alg: str):
        alg = alg.lower()
        if alg == 'macenko':
            norm = staintools.StainNormalizer(method='macenko')
        elif alg == 'vahadane':
            norm = staintools.StainNormalizer(method='vahadane')
        elif alg == 'reinhard':
            norm = staintools.ReinhardColorNormalizer()
        else:
            raise ValueError(f"Unsupported method: {alg}")
        norm.fit(ref_rgb)
        return norm

    # ---------- pipeline ----------
    ref_rgb = _load_rgb(reference_image_path)
    normalizer = _build_normalizer(ref_rgb, method)

    os.makedirs(normalized_images_dir, exist_ok=True)
    src_paths = [os.path.join(source_images_dir, f)
                 for f in natsorted(os.listdir(source_images_dir))]

    failed = []
    for img_path in tqdm(src_paths,
                         desc=f"Normalizing images with {method}"):
        try:
            base = os.path.basename(img_path)
            out_path = os.path.join(
                normalized_images_dir,
                os.path.splitext(base)[0] + "_normalized.png")

            if os.path.exists(out_path):  # skip if done
                continue

            img_rgb = _load_rgb(img_path)
            norm_rgb = normalizer.transform(img_rgb)
            _save_rgb(norm_rgb, out_path)
        except Exception as e:
            failed.append((img_path, str(e)))

    print("\nStain normalization finished.")
    print(f"Failed files: {len(failed)}" if failed else "All images processed successfully.")
    return failed


def geojson_to_mask(wsi_path, geo_path, out_path, level=8, use_rasterio=True):
    slide = openslide.OpenSlide(wsi_path)
    level = min(level, slide.level_count - 1)
    w, h = slide.level_dimensions[level]
    down_factor = slide.level_downsamples[level]

    gdf = gpd.read_file(geo_path)
    polys = [geom for geom in gdf.geometry if isinstance(geom, (Polygon, MultiPolygon))]
    if not polys:
        print(f'[{os.path.basename(wsi_path)}] No polygon, skip.')
        return

    def scale_geom(geom):
        return affinity.scale(
            geom,
            xfact=1.0 / down_factor,
            yfact=1.0 / down_factor,
            origin=(0, 0)
        )

    scaled_polys = [scale_geom(p) for p in polys]

    if use_rasterio:
        mask = features.rasterize(
            ((g, 255) for g in scaled_polys),
            out_shape=(h, w),
            fill=0,
            dtype=np.uint8
        )
    else:
        img = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(img)
        for g in scaled_polys:
            if isinstance(g, Polygon):
                draw.polygon(list(g.exterior.coords), outline=255, fill=255)
            else:
                for sub in g:
                    draw.polygon(list(sub.exterior.coords), outline=255, fill=255)
        mask = np.array(img, dtype=np.uint8)

    Image.fromarray(mask, mode="L").save(out_path, compress_level=1)
    slide.close()


def batch_convert(wsi_dir, geojson_dir, mask_dir, level=8):
    os.makedirs(mask_dir, exist_ok=True)
    wsi_files = [f for f in os.listdir(wsi_dir) if f.lower().endswith(('.svs', '.tif', '.tiff', '.ndpi'))]
    for fname in tqdm(wsi_files, desc="Converting"):
        stem = os.path.splitext(fname)[0]
        wsi = os.path.join(wsi_dir, fname)
        geo = os.path.join(geojson_dir, stem + '.geojson')
        out = os.path.join(mask_dir, stem + '.png')
        if os.path.exists(out):
            print(f'Exist {out} Skip')
        if not os.path.exists(geo):
            print(f'[WARN] {geo} not found, skip.')
            continue
        geojson_to_mask(wsi, geo, out, level=level)


if __name__ == "__main__":
    # wsi_dir =     r"\\wsl.localhost\Ubuntu-22.04\home\zyn\PycharmProjects\hover_net\testWSI"
    # geojson_dir = r"\\wsl.localhost\Ubuntu-22.04\home\zyn\PycharmProjects\hover_net\testGeoJson"
    # mask_dir =    r"\\wsl.localhost\Ubuntu-22.04\home\zyn\PycharmProjects\hover_net\testWSImasks"
    # batch_convert(wsi_dir, geojson_dir, mask_dir, level=4)

    wsi_dir = r"\\172.23.3.8\yxyxlab\Zyn\PyCharmProjects\hover_net\testWSI_1"
    geojson_dir = r"\\172.23.3.8\yxyxlab\Zyn\PyCharmProjects\hover_net\testGeoJson"
    mask_dir = r"\\172.23.3.8\yxyxlab\Zyn\PyCharmProjects\hover_net\testWSImasks"
    batch_convert(wsi_dir, geojson_dir, mask_dir, level=4)
