from collections import OrderedDict

import SimpleITK as sitk
import numpy as np
from batchgenerators.augmentations.utils import resize_segmentation
from scipy.ndimage import map_coordinates
from skimage.transform import resize


def resample_data_or_seg(data, new_shape, is_seg, axis=None, order=3, do_separate_z=False, order_z=0):
    """
    对数据或分割图进行重采样到新的形状。

    separate_z=True 会沿着z轴使用order_z（默认为0，即最近邻）进行重采样。

    :param data: numpy 数组，输入的原始数据或分割图。
                 必须是4维数组，格式为 (c, x, y, z)，其中 c 是通道数。
                 (numpy array, the data to be resampled. Must be (c, x, y, z))
    :param new_shape: 元组或列表，指定重采样后的目标空间形状 (x, y, z)。
                      (tuple/list, the spatial shape the data should be resampled to (x, y, z))
    :param is_seg: 布尔值，指示输入数据是否为分割图。
                   如果为 True，则使用适合分割图的重采样方法（通常是最近邻或特殊处理以保持标签值）。
                   如果为 False，则使用标准的插值方法。
                   (bool, determines whether the input is a segmentation map (uses special resizing) or regular data)
    :param axis: 列表或元组，当 do_separate_z=True 时，指定需要分开处理的轴（通常是代表z轴的索引，如[2]）。
                 当前实现只支持一个各向异性轴。
                 (list/tuple, specifies the axis to be treated separately when do_separate_z is True (usually the z-axis index, e.g., [2]). Only one axis is supported.)
    :param order: 整数，指定插值的阶数。
                  对于 is_seg=False（原始数据），常用的值有：
                      0: 最近邻插值 (Nearest neighbor)
                      1: 线性插值 (Linear)
                      3: 三次样条插值 (Cubic spline)
                  对于 is_seg=True，batchgenerators 的 resize_segmentation 会内部处理，通常倾向于最近邻。
                  当 do_separate_z=True 时，此参数用于面内（in-plane）插值。
                  (int, the order of interpolation for resampling. 0=nearest, 1=linear, 3=cubic, etc. For seg, handled by resize_segmentation. If do_separate_z=True, this is the in-plane order.)
    :param do_separate_z: 布尔值，是否对指定的 `axis` 进行特殊的分离处理。
                          如果为 True，则先在非指定轴构成的平面（如XY平面）上使用 `order` 进行插值，
                          然后在指定的 `axis` 方向上使用 `order_z` 进行插值。这常用于处理各向异性的数据（如不同层厚的CT/MRI）。
                          (bool, whether to handle the 'axis' separately. If True, resamples in-plane first with 'order', then along 'axis' with 'order_z'. Useful for anisotropic data.)
    :param order_z: 整数，仅当 do_separate_z=True 时生效。
                    指定在分离处理的轴（`axis`）方向上使用的插值阶数。
                    默认为 0 (最近邻插值)，这对于保持分割标签的离散性或避免在低分辨率轴上引入过多模糊是常见的选择。
                    (int, interpolation order along the separately handled 'axis', only used if do_separate_z is True. Defaults to 0 (nearest neighbor).)
    :return: 重采样后的 numpy 数组，形状为 (c, new_shape[0], new_shape[1], new_shape[2])。
             (numpy array, the resampled data with shape (c, new_shape[0], new_shape[1], new_shape[2]))
    """
    # 断言检查：确保输入数据是4维 (通道, x, y, z)
    assert len(data.shape) == 4, "data must be (c, x, y, z)"
    # 断言检查：确保目标形状是3维 (x, y, z)
    assert len(new_shape) == len(data.shape) - 1

    # 根据 is_seg 选择不同的重采样函数和参数
    if is_seg:
        # 分割图使用 batchgenerators 的 resize_segmentation
        resize_fn = resize_segmentation
        kwargs = OrderedDict()  # resize_segmentation 可能不需要额外参数
    else:
        # 原始数据使用 skimage 的 resize
        resize_fn = resize
        # 设置 skimage.resize 的参数：边界模式为'edge'，关闭抗锯齿（通常为了速度或特定效果）
        kwargs = {'mode': 'edge', 'anti_aliasing': False}

    # 记录原始数据类型，以便最后转换回去
    dtype_data = data.dtype
    # 获取原始数据的空间形状 (x, y, z)
    shape = np.array(data[0].shape)
    # 将目标形状转换为 numpy 数组，方便比较
    new_shape = np.array(new_shape)

    # 检查原始空间形状和目标形状是否不同，如果不同才需要执行重采样
    if np.any(shape != new_shape):
        # 将数据类型转换为 float 进行插值计算，避免整数运算导致的问题
        data = data.astype(float)

        # 如果需要分开处理 Z 轴 (或其他指定轴)
        if do_separate_z:
            print("separate z, order in z is", order_z, "order inplane is", order)
            # 断言检查：确保 axis 参数被提供且只包含一个轴索引
            assert axis is not None and len(axis) == 1, "only one anisotropic axis supported"
            axis = axis[0]  # 获取轴索引

            # 确定2D切片的形状 new_shape_2d
            if axis == 0:  # 如果分离轴是 x 轴
                new_shape_2d = new_shape[1:]  # 目标形状是 (y, z)
            elif axis == 1:  # 如果分离轴是 y 轴
                new_shape_2d = new_shape[[0, 2]]  # 目标形状是 (x, z)
            else:  # 如果分离轴是 z 轴 (axis == 2)
                new_shape_2d = new_shape[:-1]  # 目标形状是 (x, y)

            # 存储最终结果的列表
            reshaped_final_data = []
            # 遍历每个通道
            for c in range(data.shape[0]):
                # 存储当前通道处理后的2D切片
                reshaped_data = []
                # 遍历指定轴的每个切片
                for slice_id in range(shape[axis]):
                    # 根据分离轴的不同，提取2D切片并进行面内（in-plane）重采样
                    if axis == 0:
                        # 提取 (y, z) 切片 data[c, slice_id, :, :]
                        reshaped_slice = resize_fn(data[c, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data)
                    elif axis == 1:
                        # 提取 (x, z) 切片 data[c, :, slice_id, :]
                        reshaped_slice = resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs).astype(
                            dtype_data)
                    else:  # axis == 2
                        # 提取 (x, y) 切片 data[c, :, :, slice_id]
                        reshaped_slice = resize_fn(data[c, :, :, slice_id], new_shape_2d, order, **kwargs).astype(
                            dtype_data)
                    # 将重采样后的2D切片添加到列表中
                    reshaped_data.append(reshaped_slice)

                # 将处理完的2D切片列表沿着指定轴堆叠起来，形成一个初步重采样（仅面内）的3D体积
                # 注意：此时体积的形状是 (new_shape_2d[0], new_shape_2d[1], shape[axis]) 或类似排列，但轴顺序与原始不同
                # np.stack 会将新的轴插入到指定位置，所以结果的轴顺序是正确的
                reshaped_data = np.stack(reshaped_data, axis=axis)

                # 检查指定轴的长度是否也需要改变
                if shape[axis] != new_shape[axis]:
                    # 如果指定轴的长度也需要改变，则在此方向上进行第二次重采样

                    # --- 这部分代码使用 scipy.ndimage.map_coordinates 进行沿轴插值 ---
                    # 获取面内重采样后的数据形状
                    orig_rows, orig_cols, orig_dim = reshaped_data.shape  # 注意：这里的rows, cols, dim对应的是 reshaped_data 的轴
                    # 获取最终目标形状
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]

                    # 计算每个维度上的缩放因子
                    # 注意：这里假设 reshaped_data 的轴顺序已经是 (x, y, z) 或与 new_shape 一致
                    # 缩放因子 = 原始尺寸 / 目标尺寸
                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim  # 特别是这个，沿分离轴的缩放

                    # 创建目标坐标网格 (使用 np.mgrid)
                    # map_rows, map_cols, map_dims 的形状都是 (rows, cols, dim)
                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]

                    # 将目标坐标网格映射回原始（reshaped_data）坐标系
                    # +0.5 和 -0.5 是为了使采样点位于像素/体素中心
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5  # 沿分离轴的映射

                    # 组合成 map_coordinates 需要的坐标数组，形状为 (3, rows, cols, dim)
                    coord_map = np.array([map_rows, map_cols, map_dims])

                    # 如果不是分割图 或者 分割图也要求使用最近邻插值(order_z=0)
                    if not is_seg or order_z == 0:
                        # 使用 map_coordinates 进行插值
                        # order=order_z 指定沿分离轴的插值阶数
                        # mode='nearest' 处理边界外的坐标
                        reshaped_final_data.append(map_coordinates(reshaped_data, coord_map, order=order_z,
                                                                   mode='nearest')[None].astype(dtype_data))
                    else:
                        # 如果是分割图且 order_z 非0 (例如要做线性或更高阶插值后阈值化 - 不推荐但代码实现了)
                        # 这种方式比较复杂且可能效果不好，通常分割图用最近邻(order=0)
                        # 获取唯一的标签值
                        unique_labels = np.unique(reshaped_data)
                        # 创建一个空的输出数组
                        reshaped = np.zeros(new_shape, dtype=dtype_data)

                        # 对每个标签分别处理
                        for i, cl in enumerate(unique_labels):
                            # 创建一个布尔掩码，表示当前标签 cl 的位置，并转为 float
                            label_mask = (reshaped_data == cl).astype(float)
                            # 对这个二值掩码进行插值
                            reshaped_multihot = map_coordinates(label_mask, coord_map, order=order_z,
                                                                mode='nearest')
                            # 四舍五入插值结果，并找到大于0.5的位置（认为是该标签）
                            # 将这些位置在最终输出 reshaped 中赋值为当前标签 cl
                            reshaped[np.round(reshaped_multihot) > 0.5] = cl
                        # 将处理完的单通道结果添加到最终列表
                        reshaped_final_data.append(reshaped[None].astype(dtype_data))
                else:
                    # 如果指定轴的长度不需要改变，则面内重采样后的 reshaped_data 就是最终结果
                    reshaped_final_data.append(reshaped_data[None].astype(dtype_data))

            # 将所有通道的结果堆叠起来，形成最终的4D数组 (c, x, y, z)
            reshaped_final_data = np.vstack(reshaped_final_data)

        # 如果不需要分开处理 Z 轴 (do_separate_z=False)
        else:
            print("no separate z, order", order)
            reshaped = []
            # 直接对每个通道的整个3D体积进行一次性重采样
            for c in range(data.shape[0]):
                reshaped_channel = resize_fn(data[c], new_shape, order, **kwargs)
                # 添加通道维度 [None] 并转换回原始数据类型
                reshaped.append(reshaped_channel[None].astype(dtype_data))
            # 将所有通道的结果堆叠起来
            reshaped_final_data = np.vstack(reshaped)

        # 返回重采样后的数据，确保数据类型是原始类型
        return reshaped_final_data.astype(dtype_data)
    else:
        print("no resampling necessary")
        return data


if __name__ == '__main__':
    data_xyz = sitk.GetArrayFromImage(sitk.ReadImage(r"C:\Users\Administrator\Desktop\6056.nii.gz"))

    data = np.transpose(data_xyz, (2, 1, 0))
    data_with_channel = data[np.newaxis, :]

    print("\nResampling with separate z:")
    original_shape_spatial = data_with_channel.shape[1:]
    resampled_sep_z = resample_data_or_seg(data_with_channel,
                                           (360, 274, 94),
                                           is_seg=True,
                                           axis=[2],
                                           order=3,
                                           do_separate_z=True,
                                           order_z=1)
    print(f"Resampled shape (separate z): {resampled_sep_z.shape}")
    resampled_sep_z = np.transpose(resampled_sep_z, (0, 3, 2, 1))
    sitk.WriteImage(sitk.GetImageFromArray(resampled_sep_z[0, :]), r"C:\Users\Administrator\Desktop\6056_1.nii.gz")
