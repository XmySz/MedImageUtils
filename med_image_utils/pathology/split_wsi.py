"""
    将一张WSI上的两个患者切割开
"""
import os

VIPS_BIN = r"D:\Softwares\vips-dev-8.18\bin"
os.add_dll_directory(VIPS_BIN)

import pyvips

image = pyvips.Image.new_from_file(r"E:\胃癌\胃癌SVS文件\表格无对应\201534410.svs", access='sequential')

# 自动获取整张WSI的宽高
w, h = image.width, image.height
half_w = w // 2

# 两个区域：全高 + 半宽（左半、右半）
region1 = (0,      0, half_w,     h)          # 左半
region2 = (half_w, 0, w - half_w, h)          # 右半（兼容奇数宽度）

# 裁剪出区域 1
crop1 = image.extract_area(*region1)
crop1.write_to_file(r"C:\Users\Administrator\Desktop\正常\201534410.tif",
                    tile=True,
                    pyramid=True,
                    compression='jpeg',
                    bigtiff=True,
                    )
print("病人1 处理完成")

# 裁剪出区域 2
crop2 = image.extract_area(*region2)
crop2.write_to_file(r"C:\Users\Administrator\Desktop\正常\201535037.tif",
                    tile=True,
                    pyramid=True,
                    compression='jpeg',
                    bigtiff=True
                    )
print("病人2 处理完成")
