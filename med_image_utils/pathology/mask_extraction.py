"""
组织区域 Mask 提取工具
======================
提供三个函数：
  1. extract_mask_from_png  —— 从 PNG 图像中提取组织 mask
  2. extract_mask_from_svs  —— 从 SVS 病理全切片图像中提取组织 mask
  3. apply_mask_to_png      —— 根据 mask 对原始 PNG 图像作遮罩

已修复：所有读写均兼容中文路径（Windows / Linux 通用）。

依赖：
  pip install opencv-python numpy openslide-python Pillow
"""

import cv2
import numpy as np
from pathlib import Path


# ────────────────────────────────────────────────────────────
#  中文路径兼容的图像读写辅助函数
# ────────────────────────────────────────────────────────────
def _imread_unicode(image_path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """
    替代 cv2.imread，支持中文 / 非 ASCII 路径。
    通过 np.fromfile 读取字节再用 cv2.imdecode 解码。
    """
    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(data, flags)
        return image
    except Exception:
        return None


def _imwrite_unicode(output_path: str, image: np.ndarray, ext: str = ".png") -> bool:
    """
    替代 cv2.imwrite，支持中文 / 非 ASCII 路径。
    通过 cv2.imencode 编码再用 ndarray.tofile 写入。
    """
    try:
        # 确保输出目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        success, buf = cv2.imencode(ext, image)
        if success:
            buf.tofile(output_path)
            return True
        return False
    except Exception:
        return False


# ────────────────────────────────────────────────────────────
#  函数 1：从 PNG 图像提取组织 Mask
# ────────────────────────────────────────────────────────────
def extract_mask_from_png(
    image_path: str,
    output_path: str | None = None,
    gaussian_ksize: int = 5,
    morph_ksize: int = 7,
    morph_iterations: int = 3,
) -> np.ndarray:
    """
    读取 PNG 图像，使用 Otsu 算法提取组织区域的二值 mask 并保存为 PNG。

    Parameters
    ----------
    image_path : str
        输入 PNG 图像路径（支持中文路径）。
    output_path : str | None
        输出 mask 的保存路径。若为 None，则自动在同目录下生成
        ``<原文件名>_mask.png``。
    gaussian_ksize : int
        高斯模糊的核大小（奇数），用于预处理降噪。
    morph_ksize : int
        形态学操作的核大小，用于后处理填补空洞、去除小噪点。
    morph_iterations : int
        形态学闭运算 + 开运算的迭代次数。

    Returns
    -------
    mask : np.ndarray
        二值 mask，组织区域为 255，背景为 0。
    """
    # 1. 读取图像（兼容中文路径）
    image = _imread_unicode(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    # 2. 转灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. 高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, (gaussian_ksize, gaussian_ksize), 0)

    # 4. Otsu 二值化（背景通常偏白，组织偏深）
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 5. 形态学后处理：闭运算填补空洞，开运算去除小噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)

    # 6. 保存 mask（兼容中文路径）
    if output_path is None:
        p = Path(image_path)
        output_path = str(p.parent / f"{p.stem}_mask.png")
    _imwrite_unicode(output_path, mask)
    print(f"[PNG] Mask 已保存至: {output_path}")

    return mask


# ────────────────────────────────────────────────────────────
#  函数 2：从 SVS 病理全切片图像提取组织 Mask
# ────────────────────────────────────────────────────────────
def extract_mask_from_svs(
    svs_path: str,
    target_magnification: float = 1.25,
    output_path: str | None = None,
    gaussian_ksize: int = 5,
    morph_ksize: int = 7,
    morph_iterations: int = 3,
) -> np.ndarray:
    """
    读取 SVS 格式的病理全切片图像，在指定放大倍率下使用 Otsu 算法
    提取组织区域的二值 mask 并保存为 PNG。

    Parameters
    ----------
    svs_path : str
        输入 SVS 文件路径（支持中文路径）。
    target_magnification : float
        目标放大倍率，例如 1.25、2.5、5、10、20、40。
    output_path : str | None
        输出 mask 的保存路径。若为 None，则自动在同目录下生成
        ``<原文件名>_mask.png``。
    gaussian_ksize : int
        高斯模糊的核大小（奇数），用于预处理降噪。
    morph_ksize : int
        形态学操作的核大小，用于后处理。
    morph_iterations : int
        形态学闭运算 + 开运算的迭代次数。

    Returns
    -------
    mask : np.ndarray
        二值 mask，组织区域为 255，背景为 0。
    """
    try:
        from openslide import OpenSlide
    except ImportError:
        raise ImportError(
            "需要安装 openslide-python：pip install openslide-python\n"
            "同时需要系统安装 OpenSlide C 库，详见 https://openslide.org"
        )
    from PIL import Image

    # 1. 打开 SVS 文件
    slide = OpenSlide(svs_path)

    # 2. 获取最高倍率
    try:
        base_magnification = float(
            slide.properties.get("openslide.objective-power", 40)
        )
    except (ValueError, TypeError):
        base_magnification = 40.0
        print(f"[SVS] 未找到物镜倍率信息，默认使用 {base_magnification}x")

    # 3. 计算目标降采样因子，并选择最佳金字塔层级
    target_downsample = base_magnification / target_magnification
    level = slide.get_best_level_for_downsample(target_downsample)
    actual_downsample = slide.level_downsamples[level]
    actual_magnification = base_magnification / actual_downsample
    level_dims = slide.level_dimensions[level]

    print(
        f"[SVS] 基础倍率: {base_magnification}x | "
        f"目标倍率: {target_magnification}x | "
        f"实际使用: {actual_magnification:.2f}x (level {level}, "
        f"尺寸 {level_dims[0]}×{level_dims[1]})"
    )

    # 4. 读取该层级的全图（PIL Image → numpy array）
    pil_image: Image.Image = slide.read_region(
        location=(0, 0), level=level, size=level_dims
    )
    pil_image = pil_image.convert("RGB")
    image = np.array(pil_image)

    slide.close()

    # 5. 转灰度
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 6. 高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, (gaussian_ksize, gaussian_ksize), 0)

    # 7. Otsu 二值化
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 8. 形态学后处理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)

    # 9. 保存 mask（兼容中文路径）
    if output_path is None:
        p = Path(svs_path)
        output_path = str(p.parent / f"{p.stem}_mask.png")
    _imwrite_unicode(output_path, mask)
    print(f"[SVS] Mask 已保存至: {output_path}")

    return mask


# ────────────────────────────────────────────────────────────
#  函数 3：根据 Mask 对原始 PNG 作遮罩
# ────────────────────────────────────────────────────────────
def apply_mask_to_png(
    image_path: str,
    mask_path: str,
    output_path: str | None = None,
    background_color: tuple[int, int, int] = (255, 255, 255),
    save_with_alpha: bool = False,
) -> np.ndarray:
    """
    根据 mask 对原始 PNG 图像作遮罩，非组织区域填充指定背景色
    或设为透明。

    Parameters
    ----------
    image_path : str
        原始 PNG 图像路径（支持中文路径）。
    mask_path : str
        二值 mask 图像路径，组织区域为 255，背景为 0。
    output_path : str | None
        输出遮罩后图像的保存路径。
    background_color : tuple[int, int, int]
        非组织区域的填充颜色（BGR 格式），默认白色。
    save_with_alpha : bool
        若为 True，输出 RGBA 图像，非组织区域完全透明。

    Returns
    -------
    result : np.ndarray
        遮罩后的图像。
    """
    # 1. 读取原图和 mask（兼容中文路径）
    image = _imread_unicode(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    mask = _imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"无法读取 mask: {mask_path}")

    # 2. 尺寸适配：mask 可能来自低倍率 SVS，和原图大小不同
    h, w = image.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # 3. 确保 mask 是严格二值
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 4. 应用遮罩
    if save_with_alpha:
        # RGBA 模式：非组织区域透明
        b, g, r = cv2.split(image)
        result = cv2.merge([b, g, r, mask])
        ext = ".png"
    else:
        # BGR 模式：非组织区域填充背景色
        background = np.full_like(image, background_color, dtype=np.uint8)
        mask_3ch = cv2.merge([mask, mask, mask])
        result = np.where(mask_3ch == 255, image, background)
        ext = ".png"

    # 5. 保存结果（兼容中文路径）
    if output_path is None:
        p = Path(image_path)
        output_path = str(p.parent / f"{p.stem}_masked.png")
    _imwrite_unicode(output_path, result, ext)
    print(f"[Mask] 遮罩图像已保存至: {output_path}")

    return result


# ────────────────────────────────────────────────────────────
#  使用示例
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 示例 1：处理 PNG 图像
    # mask = extract_mask_from_png(r"D:\Data\Jmszxyy\组织分割\转换后类别\testImages\20264181-原发灶-HE_0000.png")

    # 示例 2：处理 SVS 病理图像（指定 2.5x 倍率）
    # mask = extract_mask_from_svs("病理数据/切片_01.svs", target_magnification=2.5)

    # 示例 3：对原始图像应用 mask 遮罩
    result = apply_mask_to_png(r"D:\Data\Jmszxyy\组织分割\转换后类别\Predicts_color_mask\20264181-原发灶-HE.png",
                               r"D:\Data\Jmszxyy\组织分割\转换后类别\testImages\20264181-原发灶-HE_0000_mask.png")


