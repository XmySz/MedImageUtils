import os

import cv2
import numpy as np
from PIL import Image


def keep_largest_n_component(input_path: str, output_path: str, n: int = 2) -> None:
    """
    保留黑白图像中的最大n个连通域

    参数:
        input_path: 输入图像路径
        output_path: 输出图像路径
        n: 保留的连通域数量，默认为1
    """
    # 禁用PIL的图像大小限制
    Image.MAX_IMAGE_PIXELS = None

    img = Image.open(input_path).convert('L')
    img = np.array(img)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 按面积从大到小排序，保留前n个
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:n]

    result = np.zeros_like(img)
    cv2.drawContours(result, contours, -1, 255, thickness=cv2.FILLED)

    Image.fromarray(result).save(output_path)


def dilate_mask(input_path: str, output_path: str, kernel_size: int) -> None:
    """
    对黑白mask图像进行膨胀操作，连接零散的小区域

    Args:
        input_path: 输入图像路径
        output_path: 输出图像路径
        kernel_size: 膨胀核大小（正整数）
    """
    # 禁用PIL的图像大小限制
    Image.MAX_IMAGE_PIXELS = None

    img = Image.open(input_path).convert('L')
    img = np.array(img)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(img, kernel, iterations=1)

    Image.fromarray(dilated).save(output_path)


def downsample_image(input_path: str, output_path: str, scale: int) -> None:
    """
    将图像分辨率降低指定倍率

    Args:
        input_path: 输入图像路径
        output_path: 输出图像路径
        scale: 降低倍率（例如scale=2表示宽高各缩小为原来的1/2）
    """
    # 禁用PIL的图像大小限制
    Image.MAX_IMAGE_PIXELS = None

    img = Image.open(input_path)
    new_size = (img.width // scale, img.height // scale)
    img_downsampled = img.resize(new_size, Image.LANCZOS)
    img_downsampled.save(output_path)


if __name__ == '__main__':
    # keep_largest_n_component(r'D:\Data\PycharmProjects\WSI_Segmenter-master\tumor_mask_continuous.png',
    #                          r'D:\Data\PycharmProjects\WSI_Segmenter-master\tumor_mask_continuous2.png')

    # dilate_mask(r'F:\内膜\EC\TumorMask_PNG\2423347-5-6-HE-病理科_mask_d4.png',
    #             r'F:\内膜\EC\TumorMask_PNG\2423347-5-6-HE-病理科_mask_d4_100.png', 200)

    # downsample_image(r'F:\内膜\EC\TumorMask_PNG\2423347-5-6-HE-病理科_mask.png',
    #                 r'F:\内膜\EC\TumorMask_PNG\2423347-5-6-HE-病理科_mask_d4.png', 4)

    for f in os.listdir(r'F:\内膜\EC\TumorMask_PNG\Original'):
        print(f)
        keep_largest_n_component(os.path.join(r'F:\内膜\EC\TumorMask_PNG\Original_D4_Dilate', f), os.path.join(r'F:\内膜\EC\TumorMask_PNG\Original_D4_Dilate_2_Component', f), 2)