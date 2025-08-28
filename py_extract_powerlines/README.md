# Python版电力线提取系统

本目录包含了从MATLAB代码完整转换的Python版本电力线提取算法，保持了所有原始参数和处理逻辑。

## 🚀 快速开始

### 1. 安装依赖
```bash
cd project-development
pip install -r requirements.txt
```

### 2. 快速体验
```bash
python quick_start.py
```

### 3. 完整运行
```bash
python demo_extract_powerline.py
```

## 📁 文件说明

| 文件 | 功能 | 对应MATLAB文件 |
|------|------|---------------|
| `demo_extract_powerline.py` | 主程序，完整处理流程 | `demo_extract_powerline.m` |
| `utils.py` | 核心算法函数 | `extractPLs.m`, `getPCA.m`, `getDist.m`, `insert_3D.m` |
| `clustering.py` | 聚类和合并函数 | `merge.m`, `findMerge.m` |
| `quick_start.py` | 快速演示程序 | - |
| `requirements.txt` | Python依赖包 | - |

## 🔧 算法参数（与MATLAB完全一致）

- **radius = 0.5**: 邻域搜索半径
- **angle_thr = 10**: 角度阈值（度）
- **l_thr = 0.98**: 线性度阈值
- **min_distance_1 = 2.0**: 第一次聚类距离
- **min_distance_2 = 0.3**: 第二次聚类距离
- **min_cluster_size = 15**: 最小聚类点数
- **min_length = 1.5**: 最小电力线长度
- **insert_resolution = 0.1**: 点插入分辨率

## 📊 处理流程

1. **地面滤波**: 使用直方图分析去除地面点
2. **候选点提取**: 基于PCA线性特征分析提取候选电力线点
3. **欧几里得聚类**: 对候选点进行空间聚类
4. **电力线建模**: 合并、优化和密化电力线模型

## 🎯 输出结果

程序会生成以下可视化图像：
- `f1_nonGroundPoints.png`: 非地面点
- `f2_candidate_powerline_points.png`: 候选电力线点
- `f3_candidate_powerline_points_clusters.png`: 候选点聚类
- `f4_powerline_points_clusters.png`: 过滤后聚类
- `f5_colorization_clusters.png`: 彩色聚类显示
- `f6_powerLines_clusters.png`: 合并后电力线
- `f7_Power_line_model.png`: 最终电力线模型

## 💡 使用注意事项

1. **数据路径**: 确保点云文件位于 `../pointcloud_files/` 目录下
2. **文件格式**: 支持MATLAB .mat格式的点云文件
3. **内存要求**: 建议8GB以上内存用于处理大规模点云
4. **可视化**: 程序会自动显示中间和最终结果的3D可视化

## 🔍 参数调优指南

### 如果提取效果不佳，可尝试调整以下参数：

**噪声较多的情况:**
- 增大 `angle_thr` (如15-20)
- 增大 `l_thr` (如0.99)
- 减小 `radius` (如0.3-0.4)

**稀疏点云:**
- 减小 `min_distance_1` (如1.5)
- 增大 `radius` (如0.7-1.0)
- 减小 `min_cluster_size` (如10)

**复杂场景:**
- 适当降低 `l_thr` (如0.95-0.97)
- 调整 `angle_thr` 范围

## 🐛 故障排除

### 常见问题：

1. **找不到文件**: 检查点云文件是否在正确路径
2. **内存不足**: 减少处理的点云数量或增加系统内存
3. **没有检测到电力线**: 调整算法参数，特别是 `l_thr` 和 `angle_thr`
4. **可视化不显示**: 确保图形界面可用，或检查matplotlib配置

### 调试技巧：
- 运行 `quick_start.py` 进行快速测试
- 检查中间结果的点数量是否合理
- 使用较小的数据集进行参数调试

## 📚 算法详解

详细的算法逻辑和处理流程说明请参考：[算法逻辑与处理流程.md](算法逻辑与处理流程.md)

## 📖 引用

如果使用本代码，请引用原始论文：

Zhenwei Shi, Yi Lin, and Hui Li. "Extraction of urban power lines and potential hazard analysis from mobile laser scanning point clouds." International Journal of Remote Sensing 41, no. 9 (2020): 3411-3428.