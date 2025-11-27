import os
from typing import List, Tuple, Union

import cv2
import staintools
from tqdm import tqdm
from natsort import natsorted


def load_image(image_path: str):
    """读取图像并转为 RGB。"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb


def save_image(image_rgb, output_path: str):
    """将 RGB 图像保存为指定路径（自动创建目录）。"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image_bgr)


def get_stain_normalizer(reference_image_rgb, method: str = "macenko"):
    """根据参考图像和方法创建并拟合染色标准化器。"""
    method = method.lower()
    if method == "macenko":
        normalizer = staintools.StainNormalizer(method="macenko")
    elif method == "vahadane":
        normalizer = staintools.StainNormalizer(method="vahadane")
    elif method == "reinhard":
        normalizer = staintools.ReinhardColorNormalizer()
    else:
        raise ValueError(f"不支持的标准化方法: {method}")

    normalizer.fit(reference_image_rgb)
    return normalizer


def _normalize_suffix_list(input_suffixes: Union[str, List[str]]) -> List[str]:
    """
    把用户传入的后缀参数统一整理成标准列表形式：
    - 支持 ".jpg" / "jpg" / [".jpg", ".png"] / ["jpg", "png"] 等
    """
    if isinstance(input_suffixes, str):
        suffix_list = [input_suffixes]
    else:
        suffix_list = list(input_suffixes)

    if not suffix_list:
        raise ValueError("input_suffixes 不能为空，请至少指定一种输入格式，例如 '.jpg' 或 ['.jpg', '.png'].")

    # 统一成以 '.' 开头的小写
    normalized = []
    for s in suffix_list:
        s = s.strip()
        if not s:
            continue
        if not s.startswith("."):
            s = "." + s
        normalized.append(s.lower())

    if not normalized:
        raise ValueError("处理 input_suffixes 后为空，请检查传入的格式设置。")

    return normalized


def collect_image_paths(
    input_path: str,
    input_suffixes: Union[str, List[str]],
) -> List[str]:
    """
    收集需要处理的图像路径：
    - 如果 input_path 是文件：只处理这一张（如果后缀满足条件）
    - 如果 input_path 是文件夹：处理该目录下所有满足后缀条件的文件（不递归）
    只使用用户指定的 input_suffixes，不再使用默认格式列表。
    """
    input_suffixes = _normalize_suffix_list(input_suffixes)

    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in input_suffixes:
            return [input_path]
        else:
            print(
                f"输入的是单个文件，但扩展名 {ext} 不在指定的输入格式 {input_suffixes} 中，将不会处理。"
            )
            return []

    if os.path.isdir(input_path):
        image_paths = []
        for filename in natsorted(os.listdir(input_path)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in input_suffixes:
                image_paths.append(os.path.join(input_path, filename))
        return image_paths

    raise ValueError(f"input_path 既不是有效文件也不是有效文件夹: {input_path}")


def build_output_path(
    image_path: str,
    output_dir: str,
    output_ext: str,
) -> str:
    """根据输入路径和目标扩展名生成输出路径。"""
    if not output_ext.startswith("."):
        output_ext = f".{output_ext}"
    output_ext = output_ext.lower()

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = base_name + output_ext
    return os.path.join(output_dir, output_filename)


def normalize_and_save_single_image(
    image_path: str,
    normalizer,
    output_dir: str,
    output_ext: str,
    overwrite: bool = False,
) -> Tuple[str, str]:
    """
    对单张图像进行染色标准化并保存。
    返回 (图像路径, 处理状态字符串)。
    """
    output_path = build_output_path(image_path, output_dir, output_ext)

    if (not overwrite) and os.path.exists(output_path):
        return image_path, "跳过（输出文件已存在）"

    target_image_rgb = load_image(image_path)
    normalized_image_rgb = normalizer.transform(target_image_rgb)
    save_image(normalized_image_rgb, output_path)
    return image_path, "Success"


def run_stain_normalization(
    reference_image_path: str,
    input_path: str,
    output_dir: str,
    method: str = "macenko",
    input_suffixes: Union[str, List[str]] = None,
    output_ext: str = ".jpeg",
    overwrite: bool = False,
):
    """
    主流程：
    - 用参考图拟合 normalizer
    - 收集单个文件或目录下的所有待处理图像（只按 input_suffixes 检索）
    - 批量进行染色标准化并保存
    """
    if input_suffixes is None:
        raise ValueError(
            "必须显式指定 input_suffixes，例如 '.jpg' 或 ['.jpg', '.png']，"
            "当前设置为 None。"
        )

    # 1. 加载参考图像并拟合 normalizer
    print(f"加载参考图像: {reference_image_path}")
    reference_image_rgb = load_image(reference_image_path)
    normalizer = get_stain_normalizer(reference_image_rgb, method=method)

    # 2. 收集待处理图像路径（只按指定格式）
    image_paths = collect_image_paths(input_path, input_suffixes)
    if not image_paths:
        print("没有找到符合要求的图像文件，程序结束。")
        return

    # 3. 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 4. 批量归一化
    failed_files = []
    skipped = 0

    desc = f"使用 {method} 方法进行染色标准化"
    for img_path in tqdm(image_paths, desc=desc):
        try:
            _, status = normalize_and_save_single_image(
                img_path,
                normalizer,
                output_dir=output_dir,
                output_ext=output_ext,
                overwrite=overwrite,
            )
            if "跳过" in status:
                skipped += 1
        except Exception as e:
            failed_files.append((img_path, str(e)))

    # 5. 输出统计信息
    print("\n染色标准化完成")
    print(f"总文件数: {len(image_paths)}")
    if skipped > 0:
        print(f"跳过（已存在且未覆盖）: {skipped}")
    if failed_files:
        print(f"处理失败的文件数量: {len(failed_files)}")
        for path, err in failed_files:
            print(f"  - {path} -> {err}")
    else:
        print("所有图像处理成功")


if __name__ == "__main__":
    # 参考图像路径（用于拟合染色标准化器）
    REFERENCE_IMAGE_PATH = r"REFERENCE_IMAGE/img.png"

    # 输入路径：可以是单个图像文件，也可以是一个文件夹
    INPUT_PATH = r"D:\Data\Temp\CellProfiler\patch_000029_x15360_y6144.png"

    # 输出目录：归一化后的图像会保存在这里
    OUTPUT_DIR = r"D:\Data\Temp\CellProfiler"

    # 染色标准化方法: 'macenko', 'vahadane', 'reinhard'
    METHOD = "vahadane"

    # 需要处理的输入图像格式（只按你指定的来检索）
    # 支持写成 ".jpg" 或 "jpg"，也可以写成列表例如 [".jpg", ".png"]
    INPUT_SUFFIXES = ".png"

    # 输出图像格式（'.jpg' / 'jpg' / '.png' / 'png' / '.jpeg' 等都可以）
    OUTPUT_EXT = ".jpg"

    # 如果输出文件已存在，是否覆盖
    OVERWRITE = True

    run_stain_normalization(
        reference_image_path=REFERENCE_IMAGE_PATH,
        input_path=INPUT_PATH,
        output_dir=OUTPUT_DIR,
        method=METHOD,
        input_suffixes=INPUT_SUFFIXES,
        output_ext=OUTPUT_EXT,
        overwrite=OVERWRITE,
    )
