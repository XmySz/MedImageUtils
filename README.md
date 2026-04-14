# MedImageUtils 🏥

**MedImageUtils** 是一个面向日常工作与学习的**医学图像处理与临床数据分析**个人百宝箱。该项目旨在收集、整理并沉淀日常高频使用的医学图像算法与科研统计代码，将日常繁琐的数据预处理、指标评价和统计绘图固化为规范的、开箱即用的工具库。

---

## 📂 核心目录架构

当前项目的架构经过了合理设计，核心按照 **Python 工程包** 与 **独立分析脚本** 进行了语言级和功能级的分离，以提升代码的干净度和可复用性。

```text
MedImageUtils/
├── 📦 med_image_utils/            # 👉 Python 核心库 (支持通过 from med_image_utils... 规范导入)
│   ├── io/                        # 【输入输出】所有 DICOM / Nifti 的读写解析模块
│   ├── core/                      # 【核心计算】通用图像基础操作 (image_utils) 与精度评测 (metrics)
│   ├── radiology/                 # 【放射影像】MRI/CT数据对齐、N4偏置场校正、Spacing重采样及灰度归一化
│   └── pathology/                 # 【数字病理】WSI切片处理、Patch分割、组织区域掩膜提取与染色标准化
│
├── 📊 scripts_py/                 # 👉 Python 数据处理独立脚本工具
│   └── check_normality.py         # 包含临床表格数据的正态性检验等独立制图脚本
│
├── 📈 scripts_r/                  # 👉 R 语言的临床统计与绘图脚本
│   ├── baseline_characteristics.R # 患者基线特征统计 (Baseline table)
│   ├── auc_curve.R                # ROC曲线 / AUC计算 
│   ├── calibration_curve.R        # 模型校准度曲线
│   ├── dca.R                      # 临床决策曲线 (Decision Curve Analysis)
│   └── confusion_matrix.R         # 混淆矩阵绘制
│
├── 🧪 examples/                   # 存放调用核心库的示例代码或平时测试的临时草稿
├── 📁 resources/                  # 存放算法所依赖的静态全局数据（例如用于染色的参考图像）
└── 📄 requirements.txt            # 项目所需的 Python 第三方依赖库汇总
```

---

## 🛠️ 主要功能与工具速览

### 1. 放射影像处理 (`radiology`)
专注三维结构数据（MRI, CT等）的清洗与规范：
- **`n4_bias_correction.py`**: 自动处理 MRI 因设备伪影导致的非均匀灰度偏置场。
- **`resample.py` / `nnunet_resampler.py`**: 基于医学设备物理 Spacing 的真实空间像素重采样，保持空间结构一致性。
- **`intensity_norm.py`**: 保障跨设备影像间的像素灰度数值处于同一量级。

### 2. 病理图像分析 (`pathology`)
专注超大规模多层级全切片 (WSI) 的轻量化拆解与染色分析：
- **组织提取**: `mask_extraction.py`, `svs_geojson_to_mask.py` —— 从 SVS 原始格式与人工标注框提取关键的组织轮廓与掩膜。
- **分块(Patch)处理**: `wsi_to_patch.py`, `h5_to_patch.py`, `split_wsi.py` —— 对巨大分辨率的高倍镜影像按设定尺寸进行切块与特征留存。
- **色彩标准化**: `dye_standardization.py` —— 解决不同的数字化切片扫描仪及染色试剂造成的颜色散乱，对齐到标准参考图色池。

### 3. 数据分析与检验脚本 (`scripts_py` & `scripts_r`)
针对课题设计阶段与后期投稿所需的可视化和检验工作：
- R 语言版包含 **基线特征提取**、**各种医学常用评价曲线(ROC/DCA 等)** 绘图代码。
- Python 版包含 `check_normality.py`：使用直方图、KDE 和 Q-Q 图，一键完成读取 `.xlsx` 临床电子表格并可视化检验数据的正态分布特性。

---

## 🚀 快速上手

### 依赖安装
首先，确保您环境所需的第三方图像处理与统计计算相关包已配齐：
```bash
# 建议在 Conda 或虚拟环境内执行
pip install -r requirements.txt
```
*(主要包含 `numpy`, `SimpleITK`, `nibabel`, `openslide-python`, `staintools`, `openpyxl`, `seaborn` 等)*

### 代码导入引用原则
如果您在其他工程中或 Jupyter Notebook 下调用本项目核心方法，仅需确保本项目根目录位于环境变量的扫描路径下，即可享受丝滑的导入体验：
```python
# 示例：导入病理染色相关的工具
from med_image_utils.pathology.dye_standardization import run_stain_normalization

# 示例：导入 Nifti 的读写
from med_image_utils.io.nifti_utils import load_nifti
```

### 独立代码直接运行
对于 `scripts_py` 中的统计检验代码，您可根据文件底部的 `__name__ == "__main__"` 区块修改 Excel 路径等硬编码变量配置，即可利用 IDE 的 Run 功能轻松获得可视化结果。
