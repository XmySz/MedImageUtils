#!/usr/bin/env python3
import SimpleITK as sitk
from pathlib import Path

# ===== 需按实际环境调整的路径配置 =====
INPUT_PATH = Path(r"D:\Data\OvarianCancer\Datasets\CA_hos_Label\T2\Images")  # 可以是单个文件或目录
OUTPUT_PATH = None  # 若为目录则批量输出到该目录；若为文件则仅限单文件模式；None 时自动生成路径
MASK_PATH = Path(r"D:\Data\OvarianCancer\Datasets\CA_hos_Label\T2\Labels")  # 可选：单个掩膜文件、掩膜目录或 None
# =====================================

def n4_bias_correction(input_image_path, output_image_path, mask_image_path=None):
    input_image = sitk.ReadImage(str(input_image_path), sitk.sitkFloat32)
    if mask_image_path:
        mask_image = sitk.ReadImage(str(mask_image_path), sitk.sitkUInt8)
    else:
        mask_image = sitk.OtsuThreshold(input_image, 0, 1, 200)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    output_image = corrector.Execute(input_image, mask_image)
    output_image.CopyInformation(input_image)
    sitk.WriteImage(output_image, str(output_image_path))

def run():
    input_path = Path(INPUT_PATH).resolve()
    output_hint = Path(OUTPUT_PATH).resolve() if OUTPUT_PATH else None
    mask_hint = Path(MASK_PATH).resolve() if MASK_PATH else None

    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    if output_hint and output_hint.is_file() and input_path.is_dir():
        raise ValueError("批量模式下 OUTPUT_PATH 不能是文件，请改为目录或置为 None。")

    # 准备输出目录（目录模式或显式指定目录时）
    if input_path.is_dir():
        output_dir = output_hint if output_hint else input_path.with_name(f"{input_path.name}_n4")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[信息] 批量输出目录: {output_dir}")
    else:
        output_dir = output_hint if (output_hint and output_hint.is_dir()) else None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

    def build_output_name(img_path):
        name = img_path.name
        if name.endswith(".nii.gz"):
            base = name[:-7]
            return f"{base}_n4.nii.gz"
        if name.endswith(".nii"):
            base = name[:-4]
            return f"{base}_n4.nii"
        return f"{img_path.stem}_n4{''.join(img_path.suffixes)}"

    def pick_mask(image_file):
        if not mask_hint:
            return None
        if mask_hint.is_file():
            return mask_hint
        if mask_hint.is_dir():
            candidates = [mask_hint / image_file.name]
            if image_file.name.endswith("_0000.nii.gz"):
                stripped = image_file.name.replace("_0000", "")
                candidates.append(mask_hint / stripped)
            if image_file.suffixes[-2:] == [".nii", ".gz"]:
                base = image_file.stem  # 去掉最后一层后缀 => xxx.nii
                candidates.append(mask_hint / f"{base}.gz")
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            print(f"[提示] 未在 {mask_hint} 中找到 {image_file.name} 对应掩膜，改用 Otsu。")
        return None

    files_to_process = []
    if input_path.is_file():
        files_to_process.append(input_path)
    else:
        for item in sorted(input_path.iterdir()):
            if not item.is_file():
                continue
            suffix_combo = "".join(item.suffixes)
            if suffix_combo in (".nii.gz", ".nii"):
                files_to_process.append(item)
        if not files_to_process:
            print(f"[警告] 在 {input_path} 中未找到任何 nii/nii.gz 文件。")
            return

    for image_file in files_to_process:
        mask_file = pick_mask(image_file)
        if input_path.is_file():
            if output_hint and not output_hint.is_dir():
                target = output_hint
            elif output_dir:
                target = output_dir / build_output_name(image_file)
            else:
                target = image_file.with_name(build_output_name(image_file))
        else:
            target = output_dir / build_output_name(image_file)

        print(f"\n[处理] {image_file.name}")
        if mask_file:
            print(f"  掩膜: {mask_file.name}")
        else:
            print("  掩膜: Otsu 阈值自动生成")

        n4_bias_correction(image_file, target, mask_file)
        print(f"  输出: {target}")

if __name__ == "__main__":
    run()
