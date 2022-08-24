# AD-Toolbox

基于MMDetection和MMRotate的遥感目标检测的工具箱

主要方便自己使用，无需修改MMDet和MMRotate的代码

[English](./README.md) | 简体中文

## 支持数据集

|                        数据集                         | 水平框检测 | 旋转框检测 | 实例分割 |
| :---------------------------------------------------: | :--------: | :--------: | :------: |
| [DOTA](https://captain-whu.github.io/DOTA/index.html) |     ×      |    ✓ \*    |    -     |
|                    [HRSC2016](<>)                     |     ×      |    ✓ \*    |    -     |
|    [DIOR](https://gcheng-nwpu.github.io/#Datasets)    |     ✓      |     ✓      |    -     |
|   [VHR-10](https://gcheng-nwpu.github.io/#Datasets)   |     ✓      |     ×      |    ✓     |
|   [MAR20](https://gcheng-nwpu.github.io/#Datasets)    |     ✓      |     ✓      |    -     |

**注意:**

- `*` : `DOTA` and `HRSC2016` 由 MMRotate 官方支持.
- 所有配置文件还需要进行测试
-

## 安装

需要安装我 fork 的 mmrotate 分支 https://github.com/liuyanyi/mmrotate/tree/ryolox

等待其合并到主分支。

## 模型

- Rotated YOLOX (仍在调整)
- MMdetection 和 MMRotate 中的所有模型.

## 依赖

- [MMCV-Full](https://github.com/open-mmlab/mmcv)
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [MMRotate](https://github.com/open-mmlab/mmrotate)
